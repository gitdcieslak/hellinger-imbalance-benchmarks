"""Profile extracted legacy HDDT datasets from manifest entries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hib.datasets.legacy import DEFAULT_EXTRACTED_DIR, DEFAULT_PROCESSED_DIR  # noqa: E402
from hib.datasets.profile import (  # noqa: E402
    DEFAULT_PROFILE_CSV_PATH,
    DEFAULT_PROFILE_JSON_PATH,
    DEFAULT_PROFILE_MD_PATH,
    profile_legacy_hddt_datasets,
    write_profile_csv,
    write_profile_json,
    write_profile_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / DEFAULT_PROCESSED_DIR / "manifest.json",
        help="Manifest JSON path.",
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=ROOT / DEFAULT_EXTRACTED_DIR,
        help="Extracted data directory.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / DEFAULT_PROFILE_JSON_PATH,
        help="Profile JSON output path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / DEFAULT_PROFILE_CSV_PATH,
        help="Profile CSV output path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / DEFAULT_PROFILE_MD_PATH,
        help="Profile markdown output path.",
    )
    args = parser.parse_args()

    profile = profile_legacy_hddt_datasets(
        manifest_path=args.manifest.resolve(),
        extracted_dir=args.extracted_dir.resolve(),
    )
    json_path = write_profile_json(profile, args.output_json.resolve())
    csv_path = write_profile_csv(profile, args.output_csv.resolve())
    md_path = write_profile_markdown(profile, args.output_md.resolve())

    parsed_count = sum(1 for item in profile if item.get("parse_status") == "parsed")
    failed_count = len(profile) - parsed_count
    print(f"profile_entries: {len(profile)}")
    print(f"parsed_entries: {parsed_count}")
    print(f"failed_or_unsupported_entries: {failed_count}")
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")
    print(f"md: {md_path}")


if __name__ == "__main__":
    main()
