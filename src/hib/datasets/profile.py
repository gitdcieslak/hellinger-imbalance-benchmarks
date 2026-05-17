"""Profiling utilities for extracted legacy HDDT datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hib.datasets.legacy import (
    DEFAULT_EXTRACTED_DIR,
    DEFAULT_PROCESSED_DIR,
)


DEFAULT_MANIFEST_PATH = DEFAULT_PROCESSED_DIR / "manifest.json"
DEFAULT_PROFILE_JSON_PATH = DEFAULT_PROCESSED_DIR / "profile.json"
DEFAULT_PROFILE_CSV_PATH = Path("reports/legacy_hddt_dataset_profile.csv")
DEFAULT_PROFILE_MD_PATH = Path("reports/legacy_hddt_dataset_profile.md")

_NAME_CANDIDATES = ("class", "target", "label", "y")


def _load_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("manifest must contain a list of entries")
    return data


def _as_short_error(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text[:240]


def _pick_separator(text: str) -> str | None:
    sample_lines = [line for line in text.splitlines() if line.strip()][:20]
    if not sample_lines:
        return None
    candidate_seps = [",", "\t", ";", " "]
    best_sep: str | None = None
    best_score = -1
    for sep in candidate_seps:
        if sep == " ":
            counts = [len(line.split()) for line in sample_lines]
        else:
            counts = [line.count(sep) + 1 for line in sample_lines]
        if not counts:
            continue
        min_count = min(counts)
        max_count = max(counts)
        if min_count <= 1:
            continue
        consistency = min_count == max_count
        score = min_count + (1000 if consistency else 0)
        if score > best_score:
            best_score = score
            best_sep = sep
    return best_sep


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.dropna(axis=0, how="all")
    frame = frame.dropna(axis=1, how="all")
    return frame


def _parse_table_candidates(path: Path) -> tuple[list[pd.DataFrame], str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    sep = _pick_separator(text)
    if sep is None:
        raise ValueError("unable to infer delimiter")
    read_sep = r"\s+" if sep == " " else sep

    with_header = pd.read_csv(
        path,
        sep=read_sep,
        engine="python",
        header=0,
        na_values=["?", "NA", "N/A", "null", "None", ""],
    )
    without_header = pd.read_csv(
        path,
        sep=read_sep,
        engine="python",
        header=None,
        na_values=["?", "NA", "N/A", "null", "None", ""],
    )

    with_header = _clean_frame(with_header)
    without_header = _clean_frame(without_header)
    candidates = [with_header, without_header]
    if all(candidate.empty or candidate.shape[1] == 0 for candidate in candidates):
        raise ValueError("parsed table is empty")
    return candidates, sep


def _column_is_continuous(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return False
    unique_count = valid.nunique()
    ratio = unique_count / max(1, len(valid))
    return unique_count > 20 and ratio > 0.3


def _candidate_columns(frame: pd.DataFrame) -> list[object]:
    candidates: list[object] = []
    lower_names = {str(col).strip().lower(): col for col in frame.columns}
    for name in _NAME_CANDIDATES:
        if name in lower_names:
            candidates.append(lower_names[name])
    last_col = frame.columns[-1]
    first_col = frame.columns[0]
    if last_col not in candidates:
        candidates.append(last_col)
    if first_col not in candidates:
        candidates.append(first_col)
    return candidates


def _is_named_target_candidate(name: object) -> bool:
    return str(name).strip().lower() in _NAME_CANDIDATES


def _target_metrics(frame: pd.DataFrame, column: object) -> dict[str, object]:
    series = frame[column]
    missing_ratio = float(series.isna().mean())
    value_counts = series.value_counts(dropna=True)
    distinct = int(value_counts.shape[0])
    if distinct < 2 or distinct > 20:
        raise ValueError("target cardinality outside [2,20]")
    if missing_ratio > 0.5:
        raise ValueError("target missingness too high")
    if _column_is_continuous(series):
        raise ValueError("target appears continuous")

    counts_dict = {str(key): int(value) for key, value in value_counts.to_dict().items()}
    minority_class = min(counts_dict, key=lambda k: counts_dict[k])
    majority_class = max(counts_dict, key=lambda k: counts_dict[k])
    minority_count = counts_dict[minority_class]
    majority_count = counts_dict[majority_class]
    imbalance_ratio = float(majority_count / minority_count) if minority_count > 0 else None

    result: dict[str, object] = {
        "class_counts": counts_dict,
        "n_classes": distinct,
        "minority_class": minority_class,
        "minority_count": minority_count,
        "majority_count": majority_count,
        "imbalance_ratio": imbalance_ratio,
    }
    if distinct == 2:
        result["positive_class_candidate"] = minority_class
    return result


def _blank_profile_entry(entry: dict[str, object]) -> dict[str, object]:
    return {
        "dataset_name": entry.get("dataset_name"),
        "relative_path": entry.get("relative_path"),
        "detected_format": entry.get("detected_format", "unknown"),
        "file_size_bytes": entry.get("file_size_bytes"),
        "sha256": entry.get("sha256"),
        "parse_status": "failed",
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
        "notes": "",
    }


def profile_legacy_hddt_datasets(manifest_path: Path, extracted_dir: Path) -> list[dict[str, object]]:
    manifest_path = manifest_path.resolve()
    extracted_dir = extracted_dir.resolve()
    if not extracted_dir.exists():
        raise FileNotFoundError(f"extracted directory not found: {extracted_dir}")

    manifest_entries = _load_manifest(manifest_path)
    profiled: list[dict[str, object]] = []

    for entry in manifest_entries:
        result = _blank_profile_entry(entry)
        relative_path = entry.get("relative_path")
        if not isinstance(relative_path, str):
            result["parse_error"] = "manifest entry missing relative_path"
            profiled.append(result)
            continue

        file_path = extracted_dir / relative_path
        if not file_path.exists():
            result["parse_error"] = "file missing from extracted directory"
            profiled.append(result)
            continue

        detected_format = str(result["detected_format"])
        if detected_format == "arff":
            result["parse_status"] = "unsupported_arff"
            result["notes"] = "ARFF profiling unsupported without scipy.io.arff"
            profiled.append(result)
            continue
        if detected_format not in {"csv", "data", "txt"}:
            result["parse_status"] = "unsupported_format"
            result["notes"] = "Format not supported for profiling"
            profiled.append(result)
            continue

        try:
            frames, chosen_sep = _parse_table_candidates(file_path)
            chosen_target = None
            metrics: dict[str, object] | None = None
            best_frame: pd.DataFrame | None = None
            best_candidates: list[object] = []
            successful: list[tuple[int, int, pd.DataFrame, list[object], object, dict[str, object]]] = []

            for frame in frames:
                if frame.empty or frame.shape[1] == 0:
                    continue
                frame_candidates = _candidate_columns(frame)
                frame_target = None
                frame_metrics: dict[str, object] | None = None
                for candidate in frame_candidates:
                    try:
                        candidate_metrics = _target_metrics(frame, candidate)
                    except Exception:
                        continue
                    frame_target = candidate
                    frame_metrics = candidate_metrics
                    break
                if frame_target is not None and frame_metrics is not None:
                    score = 100 if _is_named_target_candidate(frame_target) else 0
                    successful.append(
                        (
                            score,
                            int(frame.shape[0]),
                            frame,
                            frame_candidates,
                            frame_target,
                            frame_metrics,
                        )
                    )

            if successful:
                (
                    _,
                    _,
                    best_frame,
                    best_candidates,
                    chosen_target,
                    metrics,
                ) = max(successful, key=lambda item: (item[0], item[1]))

            if best_frame is None:
                best_frame = max(frames, key=lambda frm: frm.shape[0])
                best_candidates = _candidate_columns(best_frame)

            result["n_rows"] = int(best_frame.shape[0])
            result["n_columns"] = int(best_frame.shape[1])
            result["candidate_target_columns"] = [str(col) for col in best_candidates]

            if chosen_target is None or metrics is None:
                result["parse_status"] = "parsed"
                result["notes"] = (
                    f"Parsed with delimiter '{chosen_sep}', target inference uncertain"
                )
            else:
                result["parse_status"] = "parsed"
                result["chosen_target_column"] = str(chosen_target)
                result.update(metrics)
                if result["n_classes"] and int(result["n_classes"]) > 2:
                    result["notes"] = "Multiclass target detected"
                else:
                    result["notes"] = ""
            result["parse_error"] = None
        except Exception as exc:
            result["parse_status"] = "failed"
            result["parse_error"] = _as_short_error(exc)

        profiled.append(result)

    return profiled


def write_profile_json(profile: list[dict[str, object]], output_path: Path) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)
        handle.write("\n")
    return output_path


def write_profile_csv(profile: list[dict[str, object]], output_path: Path) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for entry in profile:
        row = dict(entry)
        row["candidate_target_columns"] = json.dumps(
            row.get("candidate_target_columns", []), sort_keys=True
        )
        row["class_counts"] = json.dumps(row.get("class_counts"), sort_keys=True)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def write_profile_markdown(profile: list[dict[str, object]], output_path: Path) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _table_markdown(rows: list[dict[str, object]]) -> list[str]:
        if not rows:
            return []
        headers = list(rows[0].keys())
        table_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            values = [str(row.get(header, "")) for header in headers]
            table_lines.append("| " + " | ".join(values) + " |")
        return table_lines

    success_rows = [
        {
            "dataset_name": item["dataset_name"],
            "relative_path": item["relative_path"],
            "format": item["detected_format"],
            "n_rows": item["n_rows"],
            "n_columns": item["n_columns"],
            "target": item["chosen_target_column"],
            "n_classes": item["n_classes"],
            "imbalance_ratio": item["imbalance_ratio"],
            "notes": item["notes"],
        }
        for item in profile
        if item["parse_status"] == "parsed"
    ]
    failed_rows = [
        {
            "dataset_name": item["dataset_name"],
            "relative_path": item["relative_path"],
            "status": item["parse_status"],
            "parse_error": item["parse_error"],
            "notes": item["notes"],
        }
        for item in profile
        if item["parse_status"] != "parsed"
    ]

    lines = [
        "# Legacy HDDT Dataset Profiling",
        "",
        "## Parsed Datasets",
        "",
    ]
    if success_rows:
        lines.extend(_table_markdown(success_rows))
    else:
        lines.append("No datasets were successfully parsed.")

    lines.extend(["", "## Failed or Unsupported Files", ""])
    if failed_rows:
        lines.extend(_table_markdown(failed_rows))
    else:
        lines.append("No failed or unsupported files.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Target inference is conservative and may leave `chosen_target_column` empty when uncertain.",
            "- Candidate target columns are chosen from common names, then last and first columns.",
            "- Multiclass datasets are profiled as-is; benchmark integration may require binary conversion later.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
