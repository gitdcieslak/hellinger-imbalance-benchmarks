import json
from pathlib import Path

from hib.datasets.profile import (
    profile_legacy_hddt_datasets,
    write_profile_csv,
    write_profile_markdown,
)


def _write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def test_profiles_csv_with_class_column(tmp_path):
    extracted = tmp_path / "extracted"
    file_path = extracted / "sample.csv"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("f1,f2,class\n1,2,A\n3,4,B\n5,6,B\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "dataset_name": "sample",
                "relative_path": "sample.csv",
                "detected_format": "csv",
                "file_size_bytes": file_path.stat().st_size,
                "sha256": "x",
            }
        ],
    )

    prof = profile_legacy_hddt_datasets(manifest, extracted)
    entry = prof[0]
    assert entry["parse_status"] == "parsed"
    assert entry["chosen_target_column"] == "class"
    assert entry["class_counts"] == {"B": 2, "A": 1}
    assert entry["imbalance_ratio"] == 2.0


def test_profiles_data_file_with_last_column_target(tmp_path):
    extracted = tmp_path / "extracted"
    file_path = extracted / "sample.data"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("1,2,yes\n3,4,no\n5,6,no\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "dataset_name": "sample",
                "relative_path": "sample.data",
                "detected_format": "data",
                "file_size_bytes": file_path.stat().st_size,
                "sha256": "x",
            }
        ],
    )

    prof = profile_legacy_hddt_datasets(manifest, extracted)
    assert prof[0]["chosen_target_column"] == "2"
    assert prof[0]["n_classes"] == 2


def test_delimiter_inference_tab_delimited(tmp_path):
    extracted = tmp_path / "extracted"
    file_path = extracted / "tab.txt"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("a\tb\ttarget\n1\t2\tx\n3\t4\ty\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "dataset_name": "tab",
                "relative_path": "tab.txt",
                "detected_format": "txt",
                "file_size_bytes": file_path.stat().st_size,
                "sha256": "x",
            }
        ],
    )

    prof = profile_legacy_hddt_datasets(manifest, extracted)
    assert prof[0]["parse_status"] == "parsed"
    assert prof[0]["chosen_target_column"] == "target"


def test_uncertain_target_left_null(tmp_path):
    extracted = tmp_path / "extracted"
    file_path = extracted / "continuous.csv"
    file_path.parent.mkdir(parents=True)
    rows = [f"{i},{i + 1},{i + 2}" for i in range(50)]
    file_path.write_text("a,b,c\n" + "\n".join(rows) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "dataset_name": "continuous",
                "relative_path": "continuous.csv",
                "detected_format": "csv",
                "file_size_bytes": file_path.stat().st_size,
                "sha256": "x",
            }
        ],
    )

    prof = profile_legacy_hddt_datasets(manifest, extracted)
    assert prof[0]["chosen_target_column"] is None
    assert "uncertain" in str(prof[0]["notes"]).lower()


def test_failed_parse_is_captured(tmp_path):
    extracted = tmp_path / "extracted"
    file_path = extracted / "broken.csv"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "dataset_name": "broken",
                "relative_path": "broken.csv",
                "detected_format": "csv",
                "file_size_bytes": file_path.stat().st_size,
                "sha256": "x",
            }
        ],
    )

    prof = profile_legacy_hddt_datasets(manifest, extracted)
    assert prof[0]["parse_status"] == "failed"
    assert prof[0]["parse_error"]


def test_markdown_and_csv_report_generation(tmp_path):
    profile = [
        {
            "dataset_name": "ok",
            "relative_path": "ok.csv",
            "detected_format": "csv",
            "file_size_bytes": 10,
            "sha256": "x",
            "parse_status": "parsed",
            "parse_error": None,
            "n_rows": 3,
            "n_columns": 2,
            "candidate_target_columns": ["class"],
            "chosen_target_column": "class",
            "class_counts": {"a": 2, "b": 1},
            "n_classes": 2,
            "minority_class": "b",
            "minority_count": 1,
            "majority_count": 2,
            "imbalance_ratio": 2.0,
            "positive_class_candidate": "b",
            "notes": "",
        },
        {
            "dataset_name": "bad",
            "relative_path": "bad.arff",
            "detected_format": "arff",
            "file_size_bytes": 10,
            "sha256": "y",
            "parse_status": "unsupported_arff",
            "parse_error": None,
            "n_rows": None,
            "n_columns": None,
            "candidate_target_columns": [],
            "chosen_target_column": None,
            "class_counts": None,
            "n_classes": None,
            "minority_class": None,
            "minority_count": None,
            "majority_count": None,
            "imbalance_ratio": None,
            "positive_class_candidate": None,
            "notes": "unsupported",
        },
    ]

    csv_path = write_profile_csv(profile, tmp_path / "profile.csv")
    md_path = write_profile_markdown(profile, tmp_path / "profile.md")

    csv_text = csv_path.read_text(encoding="utf-8")
    md_text = md_path.read_text(encoding="utf-8")
    assert "dataset_name" in csv_text
    assert "Legacy HDDT Dataset Profiling" in md_text
    assert "Failed or Unsupported Files" in md_text
