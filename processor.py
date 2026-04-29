from pathlib import Path

import yaml

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
        self.last_completed_shard = self._clean_shard_id(
            self._load_progress().get("last_completed_shard")
            or resume_cfg.get("last_completed_shard")
        )

        self.include_tags = set(filter_cfg.get("include_tags") or [])
        self.exclude_tags = set(filter_cfg.get("exclude_tags") or [])

        self._remote_shards = None

    @staticmethod
    def _load_yaml(path):
        with Path(path).open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    @staticmethod
    def _clean_shard_id(shard_id):
        if not shard_id:
            return None

        shard_id = Path(str(shard_id)).as_posix().strip("/")
        if shard_id.endswith(".tar"):
            shard_id = shard_id[:-4]

        return shard_id

    def _load_progress(self):
        if not self.resume_enabled or not self.progress_path.exists():
            return {}

        with self.progress_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def _save_progress(self, shard_id):
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with self.progress_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump({"last_completed_shard": shard_id}, file, sort_keys=False)
