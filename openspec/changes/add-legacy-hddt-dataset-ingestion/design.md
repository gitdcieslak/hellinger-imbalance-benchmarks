# Design

## Data Layout

Use local directories under `data/`:

- `data/raw/legacy_hddt/` for source tarballs
- `data/extracted/legacy_hddt/` for unpacked files
- `data/processed/legacy_hddt/` for generated manifest

Raw/extracted/processed artifacts remain gitignored.

## Extraction

Extraction logic scans `.tar`, `.tar.gz`, and `.tgz` archives. When no archive is passed, exactly one archive must exist in `data/raw/legacy_hddt/`; otherwise extraction errors with a clear message.

Safety constraints:

- Reject path traversal members (for example `../evil.txt`).
- Reject symlink and hardlink members.
- Refuse overwrite unless `--force` is provided.

## Manifest

After extraction, generate `data/processed/legacy_hddt/manifest.json` with entries:

- `dataset_name`
- `relative_path`
- `file_size_bytes`
- `sha256`
- `detected_format` (`csv|data|arff|txt|unknown`)
- `n_rows` (`null`)
- `n_columns` (`null`)
- `notes` (empty string)

This provides provenance now and enables registry parsing later.

## Registry Scaffold

Add `LEGACY_HDDT_DATASET_REGISTRY = {}` in `src/hib/datasets/registry.py` as the future integration point for benchmark dataset loading.
