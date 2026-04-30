import json
import tarfile
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

HF_PATH_DATASET = "nRuaif-reseach-lab/Danbooru-2026"
REMOTE_DATA_DIRS = ("data", "data_1", "data_2", "data_3", "data_4", "data_6", "data_9")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class DanbooruProcessor:
    def __init__(self, cfg_path="config.yaml"):
        self.cfg_path = Path(cfg_path)
        self.cfg = self._load_yaml(self.cfg_path)

        paths_cfg = self.cfg.get("paths", {})
        processing_cfg = self.cfg.get("processing", {})
        output_cfg = self.cfg.get("output", {})
        resume_cfg = self.cfg.get("resume", {})
        filter_cfg = self.cfg.get("filter", {})

        self.cache_dir = Path(paths_cfg.get("cache", "./data"))
        self.output_dir = Path(
            paths_cfg.get("output", output_cfg.get("path", "./output"))
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.repo_id = HF_PATH_DATASET
        self.prefetch_shards = int(processing_cfg.get("prefetch_shards", 4))
        self.progress_path = Path(
            paths_cfg.get("progress", self.cache_dir / ".progress.yaml")
        )
        self.manifest_path = Path(
            paths_cfg.get("manifest", self.cache_dir / ".shards.txt")
        )
        self.delete_shards = bool(processing_cfg.get("delete_completed_shards", True))

        self.resume_enabled = bool(resume_cfg.get("enabled", True))
        self.last_completed_shard = self._load_progress().get(
            "last_completed_shard"
        ) or resume_cfg.get("last_completed_shard")

        self.include_tags = set(filter_cfg.get("include_tags") or [])
        self.exclude_tags = set(filter_cfg.get("exclude_tags") or [])

        self._remote_shards = None

    @staticmethod
    def _load_yaml(path):
        with Path(path).open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def _load_progress(self):
        if not self.resume_enabled or not self.progress_path.exists():
            return {}

        with self.progress_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def _save_progress(self, shard_path):
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with self.progress_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump({"last_completed_shard": shard_path}, file, sort_keys=False)

    def _remote_shard_ids(self):
        if self._remote_shards is None:
            if self.manifest_path.exists():
                files = self.manifest_path.read_text(encoding="utf-8").splitlines()
                files = [file.strip() for file in files if file.strip()]
            else:
                files = HfApi().list_repo_files(self.repo_id, repo_type="dataset")
                files = [file for file in files if file.endswith(".tar")]
                self.manifest_path.write_text(
                    "\n".join(sorted(files)) + "\n", encoding="utf-8"
                )

            self._remote_shards = sorted(files)

        return self._remote_shards

    def _pending_shards(self):
        for shard_path in self._remote_shard_ids():
            if (
                self.resume_enabled
                and self.last_completed_shard
                and shard_path <= self.last_completed_shard
            ):
                continue

            yield shard_path

    def _ensure_prefetch(self, current_shard_path):
        shards = self._remote_shard_ids()
        current_index = shards.index(current_shard_path)

        for shard_path in shards[current_index : current_index + self.prefetch_shards]:
            self._download_shard(shard_path)

    def _download_shard(self, shard_path):
        local_path = self.cache_dir / shard_path
        if local_path.exists():
            return local_path

        print(f"downloading {shard_path}")
        hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=shard_path,
            local_dir=self.cache_dir,
        )
        return local_path

    def _mark_completed(self, shard_path):
        self.last_completed_shard = shard_path
        if self.resume_enabled:
            self._save_progress(shard_path)

    def _process_shard(self, shard_path, local_path):
        kept = 0
        skipped = 0

        with tarfile.open(local_path, "r") as tar:
            samples = self._index_tar_members(tar)

            for key in sorted(samples):
                files = samples[key]
                image = files["image"]
                meta_json = files[".json"]

                metadata = self._read_json_member(tar, meta_json)
                if not self._sample_allowed(metadata):
                    skipped += 1
                    continue

                self._write_sample(shard_path, key, image, metadata, tar)
                kept += 1

        return kept, skipped

    @staticmethod
    def _index_tar_members(tar):
        samples = {}

        for member in tar.getmembers():
            if not member.isfile():
                continue

            path = Path(member.name)
            suffix = path.suffix.lower()
            key = path.with_suffix("").as_posix()
            sample = samples.setdefault(key, {})

            if suffix == ".webp":
                sample["image"] = member
            elif suffix == ".json":
                sample["meta"] = member

        return samples

    def _write_sample(self, shard_path, key, image_member, metadata, tar):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shard_name = shard_path.removesuffix(".tar").replace("/", "_")
        output_name = f"{shard_name}_{Path(key).name}"
        image_path = self.output_dir / f"{output_name}.webp"
        json_path = self.output_dir / f"{output_name}.json"

        image = tar.extractfile(image_member)
        if image is None:
            return

        image_path.write_bytes(image.read())
        json_path.write_text(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def _read_json_member(self, tar, member):
        file = tar.extractfile(member)
        if file is None:
            return {}

        return json.loads(file.read().decode("utf-8"))

    def _sample_allowed(self, metadata):
        if not self.include_tags and not self.exclude_tags:
            return True

        tags = set((metadata.get("tag_string") or "").split())

        if self.include_tags and not self.include_tags.issubset(tags):
            return False

        return tags.isdisjoint(self.exclude_tags)
