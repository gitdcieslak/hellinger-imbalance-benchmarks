# Data Directory

This repository keeps dataset artifacts local and out of git.

Expected layout:

```text
data/
├── README.md
├── .gitkeep
├── raw/
│   └── legacy_hddt/
├── extracted/
│   └── legacy_hddt/
└── processed/
    └── legacy_hddt/
```

Responsibilities:

- `data/raw/`: original, untouched source archives.
- `data/extracted/`: safely unpacked archive contents.
- `data/processed/`: generated metadata and processing outputs.

Legacy HDDT tarball placement:

```text
data/raw/legacy_hddt/<original_hddt_archive>.tar.gz
```

Workflow:

1. Place the original HDDT archive under `data/raw/legacy_hddt/`.
2. Run `python scripts/extract_legacy_hddt_datasets.py`.
3. Use `data/processed/legacy_hddt/manifest.json` as the reproducible provenance record.

Reproducibility expectations:

- Extraction and manifest generation are deterministic for a fixed archive.
- The manifest captures file path, size, hash, and detected format.
- Raw, extracted, and processed dataset artifacts remain gitignored.
