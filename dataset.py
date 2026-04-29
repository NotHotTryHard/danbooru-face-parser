from pathlib import Path

import yaml
from torch.utils.data import DataLoader, IterableDataset


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
