"""Extract and manifest legacy HDDT dataset archives."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.datasets.legacy import (  # noqa: E402
    DEFAULT_EXTRACTED_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    autodetect_single_archive,
    extract_archive,
    generate_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Archive path (.tar, .tar.gz, .tgz).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / DEFAULT_EXTRACTED_DIR,
        help="Directory to extract into.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing extracted files.",
    )
    args = parser.parse_args()

    archive_path = args.archive.resolve() if args.archive else autodetect_single_archive(ROOT / DEFAULT_RAW_DIR)
    output_dir = args.output_dir.resolve()
    manifest_path = (ROOT / DEFAULT_PROCESSED_DIR / "manifest.json").resolve()

    extracted = extract_archive(archive_path=archive_path, output_dir=output_dir, force=args.force)
    entries = generate_manifest(extracted_dir=output_dir, manifest_path=manifest_path)

    print(f"archive: {archive_path}")
    print(f"extracted_files: {len(extracted)}")
    print(f"manifest_entries: {len(entries)}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
