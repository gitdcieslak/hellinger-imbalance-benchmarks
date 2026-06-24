"""Audit HDDT dataset coverage and compare strict vs relaxed validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_hddt_dataset_accessibility_validation import (  # noqa: E402
    DEFAULT_INVENTORY,
    DEFAULT_TASKS,
    build_tasks,
    discover_datasets,
    preprocess_frame,
    read_raw_data,
)
from report_hddt_dataset_accessibility_validation import (  # noqa: E402
    TARGET,
    SURVIVAL,
    _markdown_table,
    cv_r2,
    regime_variance,
    support_metrics,
)


RESULTS_DIR = ROOT / "results" / "real_validation"
REPORT_DIR = ROOT / "reports" / "real_validation"
DEFAULT_COVERAGE_CSV = RESULTS_DIR / "hddt_dataset_coverage_audit.csv"
DEFAULT_OVR_CSV = RESULTS_DIR / "hddt_ovr_task_audit.csv"
DEFAULT_COVERAGE_MD = REPORT_DIR / "hddt_dataset_coverage_audit.md"
DEFAULT_COMPARISON_MD = REPORT_DIR / "hddt_strict_vs_relaxed_comparison.md"
DEFAULT_OOS_MD = REPORT_DIR / "hddt_out_of_support_regions.md"
DEFAULT_STRICT = RESULTS_DIR / "hddt_accessibility_validation.csv"
DEFAULT_RELAXED = RESULTS_DIR / "hddt_accessibility_validation_relaxed.csv"


def classify_skip_reason(reason: str) -> str:
    text = str(reason or "").lower()
    if not text:
        return "included"
    if "n_rows" in text or "rows <" in text:
        return "too_few_rows"
    if "positive-count" in text or "positive" in text and "rules" in text:
        return "positive_fraction_too_high"
    if "fewer than 2" in text or "class" in text and "constant" in text:
        return "constant_or_invalid_target"
    if "preprocess" in text or "imput" in text or "onehot" in text:
        return "preprocessing_failure"
    if "format" in text or "parse" in text:
        return "unsupported_format"
    if "target" in text:
        return "target_column_unknown"
    return "loader_failure"


def class_skip_reason(n_rows: int, positive_count: int, positive_fraction: float, min_rows: int, min_positives: int, max_positive_fraction: float) -> str:
    if n_rows < min_rows:
        return "too_few_rows"
    if positive_count < min_positives:
        return "too_few_positives"
    if positive_fraction > max_positive_fraction:
        return "positive_fraction_too_high"
    return "included"


def task_lookup(tasks: pd.DataFrame) -> dict[str, dict[str, int]]:
    grouped = tasks.groupby("dataset_name")
    return {
        name: {
            "tasks_created": int(group["usable"].astype(bool).sum()),
            "tasks_skipped": int((~group["usable"].astype(bool)).sum()),
            "skip_reason": "; ".join(sorted(set(str(item) for item in group.loc[~group["usable"].astype(bool), "skip_reason"] if str(item))))
        }
        for name, group in grouped
    }


def coverage_audit(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory = discover_datasets(data_root, DEFAULT_INVENTORY)
    strict_tasks = build_tasks(inventory, 200, 30, 0.35, DEFAULT_TASKS)
    relaxed_tasks = build_tasks(inventory, 100, 20, 0.50, RESULTS_DIR / "hddt_task_inventory_relaxed.csv")
    strict_lookup = task_lookup(strict_tasks)
    relaxed_lookup = task_lookup(relaxed_tasks)
    rows = []
    ovr_rows = []
    for item in inventory.itertuples(index=False):
        loaded = bool(item.usable)
        load_reason = "" if loaded else str(item.skip_reason)
        strict = strict_lookup.get(item.dataset_name, {"tasks_created": 0, "tasks_skipped": 1, "skip_reason": load_reason})
        relaxed = relaxed_lookup.get(item.dataset_name, {"tasks_created": 0, "tasks_skipped": 1, "skip_reason": load_reason})
        rows.append(
            {
                "dataset_name": item.dataset_name,
                "data_path": item.data_path,
                "names_path": item.names_path,
                "file_size": item.file_size,
                "loaded_successfully": loaded,
                "load_failure_reason": load_reason,
                "n_rows": item.n_rows,
                "n_columns": item.n_columns,
                "n_classes": item.n_classes,
                "class_counts": item.class_counts,
                "minority_fraction": item.minority_fraction,
                "binary_or_multiclass": "binary" if int(item.n_classes) == 2 else "multiclass" if int(item.n_classes) > 2 else "invalid",
                "tasks_created": relaxed["tasks_created"],
                "tasks_skipped": relaxed["tasks_skipped"],
                "skip_reason": relaxed["skip_reason"] or strict["skip_reason"],
                "skip_reason_category": classify_skip_reason(relaxed["skip_reason"] or strict["skip_reason"] or load_reason),
                "strict_tasks_created": strict["tasks_created"],
                "relaxed_tasks_created": relaxed["tasks_created"],
            }
        )
        if not loaded:
            continue
        try:
            df = read_raw_data(Path(item.data_path))
            _, y = preprocess_frame(df)
            counts = y.value_counts()
            for label, count in counts.items():
                frac = float(count / len(y))
                strict_reason = class_skip_reason(len(y), int(count), frac, 200, 30, 0.35)
                relaxed_reason = class_skip_reason(len(y), int(count), frac, 100, 20, 0.50)
                ovr_rows.append(
                    {
                        "dataset_name": item.dataset_name,
                        "class_label": str(label),
                        "positive_count": int(count),
                        "positive_fraction": frac,
                        "included_under_strict_rules": strict_reason == "included",
                        "included_under_relaxed_rules": relaxed_reason == "included",
                        "skip_reason_if_excluded": "" if relaxed_reason == "included" else relaxed_reason,
                        "strict_skip_reason_if_excluded": "" if strict_reason == "included" else strict_reason,
                    }
                )
        except Exception as exc:
            ovr_rows.append({"dataset_name": item.dataset_name, "class_label": "", "positive_count": 0, "positive_fraction": np.nan, "included_under_strict_rules": False, "included_under_relaxed_rules": False, "skip_reason_if_excluded": f"loader_failure: {exc}", "strict_skip_reason_if_excluded": f"loader_failure: {exc}"})
    coverage = pd.DataFrame(rows)
    ovr = pd.DataFrame(ovr_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(DEFAULT_COVERAGE_CSV, index=False)
    ovr.to_csv(DEFAULT_OVR_CSV, index=False)
    return coverage, ovr, relaxed_tasks


def load_successful(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "fit_failed" in df.columns:
        df = df[~df["fit_failed"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    return df.dropna(subset=["breadth", "elevation", TARGET, SURVIVAL]).reset_index(drop=True)


def validation_metrics(path: Path, label: str) -> dict[str, object]:
    df = load_successful(path)
    if df.empty:
        return {"run": label, "datasets_evaluated": 0, "tasks_evaluated": 0, "rows_generated": 0, "fraction_inside_support": np.nan, "mean_nearest_support_distance": np.nan, "survival_morphology_cv_r2": np.nan, "cliffiness_morphology_cv_r2": np.nan, "between_within_regime_ratio": np.nan, "projected_regime_counts": "{}"}
    supported, support = support_metrics(df)
    var = regime_variance(df)
    return {
        "run": label,
        "datasets_evaluated": int(df["dataset_name"].nunique()),
        "tasks_evaluated": int(df["task_id"].nunique()),
        "rows_generated": int(len(df)),
        "fraction_inside_support": support["fraction_inside_support"],
        "mean_nearest_support_distance": support["mean_nearest_support_distance"],
        "survival_morphology_cv_r2": cv_r2(df, SURVIVAL, ["breadth", "elevation"]),
        "cliffiness_morphology_cv_r2": cv_r2(df, TARGET, ["breadth", "elevation"]),
        "between_within_regime_ratio": var["between_within_ratio"],
        "projected_regime_counts": json.dumps(df["regime_id"].value_counts(dropna=False).sort_index().astype(int).to_dict(), sort_keys=True),
    }


def region_label(row: pd.Series) -> str:
    b = float(row["breadth"])
    e = float(row["elevation"])
    if e >= 0.75 and b < 0.5:
        return "high elevation / low breadth"
    if e >= 0.75 and b < 1.2:
        return "high elevation / moderate breadth"
    if e < 0.2 and b >= 1.0:
        return "low elevation / high breadth"
    return "other"


def write_reports(coverage: pd.DataFrame, ovr: pd.DataFrame, strict_path: Path, relaxed_path: Path) -> tuple[Path, ...]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    coverage_summary = pd.DataFrame(
        [
            {"metric": "datasets_discovered", "value": int(len(coverage))},
            {"metric": "loaded_successfully", "value": int(coverage["loaded_successfully"].sum())},
            {"metric": "strict_tasks_created", "value": int(coverage["strict_tasks_created"].sum())},
            {"metric": "relaxed_tasks_created", "value": int(coverage["relaxed_tasks_created"].sum())},
        ]
    )
    category_summary = coverage.groupby("skip_reason_category", as_index=False).size().rename(columns={"size": "n_datasets"})
    DEFAULT_COVERAGE_MD.write_text("\n".join(["# HDDT Dataset Coverage Audit", "", "## Summary", _markdown_table(coverage_summary), "", "## Skip Categories", _markdown_table(category_summary), "", "## Dataset-Level Audit", _markdown_table(coverage)]) + "\n", encoding="utf-8")

    comparison = pd.DataFrame([validation_metrics(strict_path, "strict"), validation_metrics(relaxed_path, "relaxed")])
    DEFAULT_COMPARISON_MD.write_text("\n".join(["# HDDT Strict vs Relaxed Comparison", "", _markdown_table(comparison), "", "## Interpretation", comparison_interpretation(comparison)]) + "\n", encoding="utf-8")

    relaxed = load_successful(relaxed_path)
    supported, _ = support_metrics(relaxed)
    outside = supported[~supported["inside_axis_support"]].copy()
    if not outside.empty:
        outside["region"] = outside.apply(region_label, axis=1)
    region_summary = outside.groupby("region", as_index=False).agg(n_runs=("model_id", "size"), mean_breadth=("breadth", "mean"), mean_elevation=("elevation", "mean"), mean_cliffiness=(TARGET, "mean"), mean_survival_auc=(SURVIVAL, "mean"), mean_nearest_support_distance=("nearest_support_distance", "mean")) if not outside.empty else pd.DataFrame()
    answer = "Yes. Most out-of-support validation points occupy high-elevation regions underrepresented in the original atlas." if not region_summary.empty and region_summary["region"].astype(str).str.contains("high elevation").any() else "Not clearly; out-of-support points are diffuse or insufficient."
    DEFAULT_OOS_MD.write_text("\n".join(["# HDDT Out-of-Support Morphology Regions", "", "## Region Summary", _markdown_table(region_summary), "", "## Answer", answer, "", "## Top Outside-Support Examples", _markdown_table(outside.sort_values("nearest_support_distance", ascending=False).head(30)[["dataset_name", "task_id", "model_id", "breadth", "elevation", "nearest_support_distance", "region"]] if not outside.empty else pd.DataFrame())]) + "\n", encoding="utf-8")
    return DEFAULT_COVERAGE_MD, DEFAULT_COMPARISON_MD, DEFAULT_OOS_MD


def comparison_interpretation(comparison: pd.DataFrame) -> str:
    if comparison.empty or "relaxed" not in set(comparison["run"]):
        return "Relaxed validation results are unavailable."
    relaxed = comparison.set_index("run").loc["relaxed"]
    if float(relaxed["survival_morphology_cv_r2"]) > 0.2 and float(relaxed["fraction_inside_support"]) < 0.5:
        return "Broader partial support: survival morphology generalizes, but most validation points are outside original atlas support, so projected regime separation should be treated as out-of-distribution evidence."
    if float(relaxed["survival_morphology_cv_r2"]) > 0.2 and float(relaxed["between_within_regime_ratio"]) >= 1.0:
        return "Stronger support: relaxed inclusion preserves survival morphology and improves projected regime separation."
    return "No support: relaxed inclusion does not preserve survival morphology or regime structure."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--strict-input", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--relaxed-input", type=Path, default=DEFAULT_RELAXED)
    args = parser.parse_args()
    coverage, ovr, _ = coverage_audit(args.data_root)
    outputs = [DEFAULT_COVERAGE_CSV, DEFAULT_OVR_CSV, *write_reports(coverage, ovr, args.strict_input, args.relaxed_input)]
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
