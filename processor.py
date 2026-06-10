import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import yaml
from cheesechaser.datapool import Danbooru2024WebpDataPool
from huggingface_hub import hf_hub_download
from imgutils.detect import detect_faces
from PIL import Image

DEFAULT_REPO_ID = "deepghs/danbooru2024-webp-4Mpixel"

PAD_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "transparent": (0, 0, 0, 0),
}
SAVE_FORMATS = {"webp": "WEBP", "jpg": "JPEG", "jpeg": "JPEG", "png": "PNG"}


class DanbooruFaceParser:
    def __init__(self, cfg_path="config.yaml"):
        self.cfg = self._load_yaml(cfg_path)

        dataset_cfg = self.cfg.get("dataset", {})
        paths_cfg = self.cfg.get("paths", {})
        filter_cfg = self.cfg.get("filter", {})
        download_cfg = self.cfg.get("download", {})
        face_cfg = self.cfg.get("face_detection", {})
        output_cfg = self.cfg.get("output", {})
        image_cfg = output_cfg.get("image", {})

        self.repo_id = dataset_cfg.get("repo_id", DEFAULT_REPO_ID)

        self.cache_dir = Path(paths_cfg.get("cache", "./data"))
        self.images_dir = self.cache_dir / "images"
        self.output_dir = Path(paths_cfg.get("output", "./output"))
        self.processed_path = self.cache_dir / ".processed.txt"
        for directory in (self.images_dir, self.output_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.include_tags = sorted(set(filter_cfg.get("include_tags") or []))
        self.exclude_tags = sorted(set(filter_cfg.get("exclude_tags") or []))
        self.limit = int(filter_cfg.get("limit", 0))

        self.max_workers = int(download_cfg.get("max_workers", 8))
        self.keep_originals = bool(download_cfg.get("keep_originals", True))

        self.detect_level = str(face_cfg.get("level", "s"))
        self.confidence_threshold = float(face_cfg.get("confidence_threshold", 0.8))
        self.bbox_scale = float(face_cfg.get("bbox_scale", 1.2))

        self.target_width = int(image_cfg.get("width", 256))
        self.target_height = int(image_cfg.get("height", 256))
        self.pad = str(image_cfg.get("pad", "none"))
        self.image_format = str(image_cfg.get("format", "webp")).lower()
        self.quality = int(image_cfg.get("quality", 95))
        self.save_metadata = bool(output_cfg.get("save_metadata", True))

        if self.image_format not in SAVE_FORMATS:
            raise ValueError(f"unsupported output format: {self.image_format}")
        if self.pad == "transparent" and self.image_format in ("jpg", "jpeg"):
            raise ValueError("transparent padding requires webp or png output")

    def run(self):
        posts = self._query_posts()
        print(f"matched {len(posts)} posts for tags {self.include_tags}")

        self._download_images(posts)
        self._process_images(posts)

    @staticmethod
    def _load_yaml(path):
        with Path(path).open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    # --- stage 1: tag query over the dataset metadata index ---

    def _query_posts(self):
        query_key = hashlib.sha1(
            json.dumps(
                [self.repo_id, self.include_tags, self.exclude_tags, self.limit]
            ).encode()
        ).hexdigest()[:12]
        cache_path = self.cache_dir / f"query_{query_key}.json"

        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        # remote duckdb queries fail on HF's Xet-backed large files,
        # so fetch the metadata index once and query it locally
        print("fetching metadata index (~7 GB on first run)...")
        metadata_path = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename="metadata.parquet",
            local_dir=self.cache_dir,
        )

        print("querying metadata index...")
        con = duckdb.connect()
        sql = """
            SELECT id, tag_string, rating
            FROM read_parquet(?)
            WHERE list_has_all(string_split(tag_string, ' '), ?)
              AND NOT list_has_any(string_split(tag_string, ' '), ?)
            ORDER BY id
        """
        params = [metadata_path, self.include_tags, self.exclude_tags]
        if self.limit > 0:
            sql += " LIMIT ?"
            params.append(self.limit)

        rows = con.execute(sql, params).fetchall()
        posts = [
            {"id": post_id, "tag_string": tag_string, "rating": rating}
            for post_id, tag_string, rating in rows
        ]
        cache_path.write_text(
            json.dumps(posts, ensure_ascii=False), encoding="utf-8"
        )
        return posts

    # --- stage 2: download matched images by id ---

    def _download_images(self, posts):
        existing = {path.stem for path in self.images_dir.glob("*.webp")}
        processed = self._load_processed()
        missing = [
            post["id"]
            for post in posts
            if str(post["id"]) not in existing and post["id"] not in processed
        ]
        if not missing:
            return

        print(f"downloading {len(missing)} images")
        Danbooru2024WebpDataPool().batch_download_to_directory(
            resource_ids=missing,
            dst_dir=str(self.images_dir),
            max_workers=self.max_workers,
        )

    # --- stage 3: face detection and cropping ---

    def _process_images(self, posts):
        processed = self._load_processed()
        pending = [post for post in posts if post["id"] not in processed]
        total_faces = 0

        for index, post in enumerate(pending, start=1):
            image_path = self.images_dir / f"{post['id']}.webp"
            if not image_path.exists():
                print(f"missing image for post {post['id']}, skipping")
                continue

            faces = self._extract_faces(image_path, post)
            total_faces += faces
            self._mark_processed(post["id"])

            if not self.keep_originals:
                image_path.unlink(missing_ok=True)
            if index % 50 == 0 or index == len(pending):
                print(f"processed {index}/{len(pending)}, faces so far: {total_faces}")

        print(f"done: {len(pending)} images processed, {total_faces} faces saved")

    def _extract_faces(self, image_path, post):
        with Image.open(image_path) as image:
            image.load()
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")

            detections = detect_faces(
                image,
                level=self.detect_level,
                conf_threshold=self.confidence_threshold,
            )

            saved = 0
            for face_index, (bbox, _, confidence) in enumerate(detections):
                crop = self._crop_face(image, bbox)
                self._save_face(crop, post, face_index, bbox, confidence)
                saved += 1

        return saved

    def _crop_face(self, image, bbox):
        x0, y0, x1, y1 = bbox
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        width = (x1 - x0) * self.bbox_scale
        height = (y1 - y0) * self.bbox_scale

        # grow the region to match the target aspect ratio
        aspect = self.target_width / self.target_height
        if width / height < aspect:
            width = height * aspect
        else:
            height = width / aspect

        if self.pad == "none":
            # shrink to fit inside the image, then shift the window back in-bounds
            scale = min(1.0, image.width / width, image.height / height)
            width, height = width * scale, height * scale
            left = min(max(center_x - width / 2, 0), image.width - width)
            top = min(max(center_y - height / 2, 0), image.height - height)
            region = image.crop(
                (round(left), round(top), round(left + width), round(top + height))
            )
        else:
            region = self._crop_with_padding(image, center_x, center_y, width, height)

        resample = Image.Resampling.LANCZOS
        return region.resize((self.target_width, self.target_height), resample)

    def _crop_with_padding(self, image, center_x, center_y, width, height):
        left = round(center_x - width / 2)
        top = round(center_y - height / 2)
        right, bottom = left + round(width), top + round(height)

        if self.pad == "transparent":
            image = image.convert("RGBA")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        visible = image.crop(
            (
                max(left, 0),
                max(top, 0),
                min(right, image.width),
                min(bottom, image.height),
            )
        )
        pad_widths = (
            (max(0, -top), max(0, bottom - image.height)),
            (max(0, -left), max(0, right - image.width)),
            (0, 0),
        )

        pixels = np.asarray(visible)
        if self.pad == "edge":
            padded = np.pad(pixels, pad_widths, mode="edge")
        else:
            color = PAD_COLORS[self.pad][: pixels.shape[2]]
            padded = np.stack(
                [
                    np.pad(
                        pixels[:, :, channel],
                        pad_widths[:2],
                        constant_values=color[channel],
                    )
                    for channel in range(pixels.shape[2])
                ],
                axis=2,
            )

        return Image.fromarray(padded)

    def _save_face(self, crop, post, face_index, bbox, confidence):
        if self.image_format in ("jpg", "jpeg") and crop.mode != "RGB":
            crop = crop.convert("RGB")

        stem = f"{post['id']}_face{face_index}"
        crop.save(
            self.output_dir / f"{stem}.{self.image_format}",
            format=SAVE_FORMATS[self.image_format],
            quality=self.quality,
        )

        if self.save_metadata:
            metadata = {
                "post_id": post["id"],
                "rating": post.get("rating"),
                "tags": (post.get("tag_string") or "").split(),
                "face_index": face_index,
                "bbox": [round(value) for value in bbox],
                "confidence": round(float(confidence), 4),
            }
            (self.output_dir / f"{stem}.json").write_text(
                json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    # --- resume bookkeeping ---

    def _load_processed(self):
        if not self.processed_path.exists():
            return set()
        lines = self.processed_path.read_text(encoding="utf-8").splitlines()
        return {int(line) for line in lines if line.strip()}

    def _mark_processed(self, post_id):
        with self.processed_path.open("a", encoding="utf-8") as file:
            file.write(f"{post_id}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    DanbooruFaceParser(args.config).run()


if __name__ == "__main__":
    main()
