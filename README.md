# danbooru-face-parser

A pipeline that collects cropped anime faces from Danbooru by tag:

1. **Filter by tags** — queries the local copy of `metadata.parquet` from
   [deepghs/danbooru2024-webp-4Mpixel](https://huggingface.co/datasets/deepghs/danbooru2024-webp-4Mpixel)
   with duckdb (~8M posts, downloaded once, ~7 GB).
2. **Download only matching images** — by post id via
   [cheesechaser](https://github.com/deepghs/cheesechaser), which uses the
   dataset's tar index and HTTP range requests, so no full shards are ever fetched.
3. **Detect & crop faces** — [dghs-imgutils](https://github.com/deepghs/imgutils)
   (`deepghs/anime_face_detection`), then resize/pad each face to the configured
   resolution and format.

## Setup

```bash
uv sync
uv run hf auth login   # token needed: the dataset is gated (auto-approved)
```

Accept the gate once at the
[dataset page](https://huggingface.co/datasets/deepghs/danbooru2024-webp-4Mpixel)
(or any first request will fail with a 403 telling you to do so).

## Usage

Edit `config.yaml` (tags, limit, output size/format/padding), then:

```bash
uv run python processor.py --config config.yaml
```

Outputs land in `output/` as `<post_id>_face<N>.<ext>` plus a `.json` sidecar
with the post tags, rating, bbox and detection confidence. Progress is tracked
in `data/.processed.txt`, so reruns resume where they left off; tag query
results are cached in `data/query_*.json`.
