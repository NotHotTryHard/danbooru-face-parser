from pathlib import Path

import yaml
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class DanbooruDataset(IterableDataset):
    def __init__(self, cfg_path="config.yaml", delete_shards=True):
        self.cfg = self._load_yaml(cfg_path)

        dataset_cfg = self.cfg.get("dataset", {})
        resume_cfg = self.cfg.get("resume", {})
        filter_cfg = self.cfg.get("filter", {})

        self.root = Path(dataset_cfg.get("path", "./data"))
        self.root.mkdir(parents=True, exist_ok=True)

        self.data_dirs = resume_cfg.get("data_dirs")
        self.delete_shards = delete_shards
        self.resume_enabled = bool(resume_cfg.get("enabled", False))
        self.last_shard = self._clean_shard_id(resume_cfg.get("last_shard"))

        self.include_tags = set(filter_cfg.get("include_tags") or [])
        self.exclude_tags = set(filter_cfg.get("exclude_tags") or [])

    def __iter__(self):
        shards = self._split_between_workers(self._shards())

        for shard_id, shard_path in shards:
            yield None

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

    def _shards(self):
        shards = sorted(self._find_shards())

        if not self.resume_enabled or self.last_shard is None:
            return shards

        return [(name, path) for name, path in shards if name > self.last_shard]

    def _find_shards(self):
        for data_dir in self.data_dirs:
            base = self.root / data_dir
            if not base.exists():
                continue

            for shard_path in base.glob("shard-*"):
                if shard_path.is_file():
                    yield f"{data_dir}/{shard_path.name}", shard_path
