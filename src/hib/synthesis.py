"""Consolidated paper-facing synthesis across occupancy, elasticity, regimes, and calibration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_csv(path: str | Path, required_columns: set[str], label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {', '.join(missing)}")
    return frame


def build_paper_synthesis_table(
    occupancy_summary: pd.DataFrame,
    elasticity_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    regime_persistence: pd.DataFrame,
) -> pd.DataFrame:
    """Build model-level synthesis table from component summaries."""

    occupancy_model = (
        occupancy_summary.groupby("model_id", as_index=False)
        .agg(
            mean_occupancy_entropy=("occupancy_entropy_mean", "mean"),
            mean_posterior_sparsity=("posterior_sparsity_index_mean", "mean"),
            mean_threshold_persistence=("threshold_occupancy_persistence_mean", "mean"),
            mean_quantization_score=("quantization_score_mean", "mean"),
        )
        .sort_values("model_id")
    )

    elasticity_model = (
        elasticity_summary.groupby("model_id", as_index=False)
        .agg(
            mean_recall_elasticity=("mean_abs_recall_elasticity", "mean"),
            mean_operational_smoothness=("operational_smoothness_index", "mean"),
            mean_max_recall_jump=("max_recall_jump", "mean"),
        )
        .sort_values("model_id")
    )

    regime_model = regime_summary[
        [
            "model_id",
            "inferred_regime",
            "mean_effective_support_size",
            "mean_histogram_entropy",
        ]
    ].copy()

    cal = calibration_summary.copy()
    best_idx = cal.groupby("model_id")["mean_ece"].idxmin()
    best = cal.loc[best_idx, ["model_id", "calibration_method", "mean_ece", "mean_brier", "mean_smoothness"]].rename(
        columns={
            "calibration_method": "best_calibration_method",
            "mean_ece": "best_mean_ece",
            "mean_brier": "best_mean_brier",
            "mean_smoothness": "best_mean_smoothness",
        }
    )
    raw = cal[cal["calibration_method"] == "raw"][["model_id", "mean_ece", "mean_brier"]].rename(
        columns={"mean_ece": "raw_mean_ece", "mean_brier": "raw_mean_brier"}
    )
    calibration_model = best.merge(raw, on="model_id", how="left")
    calibration_model["ece_delta_vs_raw"] = calibration_model["best_mean_ece"] - calibration_model["raw_mean_ece"]
    calibration_model["brier_delta_vs_raw"] = calibration_model["best_mean_brier"] - calibration_model["raw_mean_brier"]

    persistence_model = (
        regime_persistence.groupby("model_id", as_index=False)
        .agg(regime_variants=("inferred_regime", lambda s: int(s.nunique())))
        .sort_values("model_id")
    )

    merged = occupancy_model.merge(elasticity_model, on="model_id", how="outer")
    merged = merged.merge(regime_model, on="model_id", how="outer")
    merged = merged.merge(calibration_model, on="model_id", how="outer")
    merged = merged.merge(persistence_model, on="model_id", how="left")
    return merged.sort_values("model_id").reset_index(drop=True)


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if pd.isna(value):
        return ""
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def paper_synthesis_to_markdown(synthesis: pd.DataFrame) -> str:
    """Render consolidated markdown report."""

    columns = [
        "model_id",
        "inferred_regime",
        "best_calibration_method",
        "best_mean_ece",
        "ece_delta_vs_raw",
        "mean_recall_elasticity",
        "mean_operational_smoothness",
        "mean_threshold_persistence",
        "mean_quantization_score",
    ]

    lines = [
        "# Operational Synthesis Report",
        "",
        "This report consolidates occupancy, threshold elasticity, regime synthesis, and calibration interaction outputs.",
        "",
        "## Integrated Model Table",
        "",
        _markdown_table(synthesis, columns),
        "",
        "## Interpretation Notes",
        "",
    ]
    for row in synthesis.itertuples(index=False):
        lines.append(f"### {row.model_id}")
        lines.append(f"- regime: {row.inferred_regime}")
        lines.append(f"- best_calibration: {row.best_calibration_method} (delta_ece_vs_raw={row.ece_delta_vs_raw:.4f})")
        lines.append(
            f"- elasticity_vs_persistence: recall_elasticity={row.mean_recall_elasticity:.4f}, threshold_persistence={row.mean_threshold_persistence:.4f}"
        )
        lines.append(
            f"- occupancy_shape: entropy={row.mean_occupancy_entropy:.4f}, quantization_score={row.mean_quantization_score:.4f}, regime_variants={int(row.regime_variants) if pd.notna(row.regime_variants) else 0}"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_paper_synthesis_report(
    occupancy_summary_path: str | Path,
    elasticity_summary_path: str | Path,
    regime_summary_path: str | Path,
    calibration_summary_path: str | Path,
    regime_persistence_path: str | Path,
    output_markdown_path: str | Path,
) -> Path:
    """Load component summaries and write consolidated markdown report."""

    occupancy = _load_csv(
        occupancy_summary_path,
        {"model_id", "occupancy_entropy_mean", "posterior_sparsity_index_mean", "threshold_occupancy_persistence_mean", "quantization_score_mean"},
        "occupancy summary",
    )
    elasticity = _load_csv(
        elasticity_summary_path,
        {"model_id", "mean_abs_recall_elasticity", "operational_smoothness_index", "max_recall_jump"},
        "elasticity summary",
    )
    regimes = _load_csv(
        regime_summary_path,
        {"model_id", "inferred_regime", "mean_effective_support_size", "mean_histogram_entropy"},
        "regime summary",
    )
    calibration = _load_csv(
        calibration_summary_path,
        {"model_id", "calibration_method", "mean_ece", "mean_brier", "mean_smoothness"},
        "calibration summary",
    )
    persistence = _load_csv(
        regime_persistence_path,
        {"model_id", "calibration_method", "inferred_regime"},
        "regime persistence summary",
    )

    synthesis = build_paper_synthesis_table(occupancy, elasticity, regimes, calibration, persistence)
    markdown = paper_synthesis_to_markdown(synthesis)
    output_path = Path(output_markdown_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path
