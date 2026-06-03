"""Build a compact Paper 2 pilot summary from topology/allocation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


SURVIVAL_COLUMNS = [
    "minority_survival_auc",
    "persistence",
    "minority_survival_cliffiness",
]

ALLOCATION_COLUMNS = [
    "positive_histogram_entropy",
    "positive_effective_score_bins",
    "positive_max_bin_mass",
    "positive_top_bin_mass",
]

DROPOUT_CONTRAST_METRICS = [
    "minority_survival_auc",
    "minority_survival_cliffiness",
    "positive_histogram_entropy",
    "positive_effective_score_bins",
    "positive_max_bin_mass",
    "positive_top_bin_mass",
    "auroc",
    "average_precision",
]

DROPOUT_CONTRASTS = [
    ("bce_dropout_0_1", "bce", "mlp_bce_dropout_0_1 - mlp_bce"),
    ("bce_dropout_0_3", "bce", "mlp_bce_dropout_0_3 - mlp_bce"),
    (
        "weighted_bce_dropout_0_1",
        "weighted_bce",
        "mlp_weighted_bce_dropout_0_1 - mlp_weighted_bce",
    ),
    (
        "weighted_bce_dropout_0_3",
        "weighted_bce",
        "mlp_weighted_bce_dropout_0_3 - mlp_weighted_bce",
    ),
]


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(format(value, floatfmt) if pd.notna(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _extract_interpretation_bullets(markdown: str) -> list[str]:
    in_section = False
    bullets = []
    for line in markdown.splitlines():
        if line.strip() == "## Interpretation":
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and line.startswith("- "):
            bullets.append(line)
    return bullets


def _extract_mediation_bullets(markdown: str) -> list[str]:
    bullets = []
    for line in markdown.splitlines():
        if line.startswith("- Interpretation:"):
            bullets.append(line)
    return bullets


def dropout_contrast_table(run_level: pd.DataFrame) -> pd.DataFrame:
    available_metrics = [metric for metric in DROPOUT_CONTRAST_METRICS if metric in run_level]
    means = run_level.groupby("objective", as_index=True)[available_metrics].mean()
    rows = []
    for variant, baseline, label in DROPOUT_CONTRASTS:
        if variant not in means.index or baseline not in means.index:
            continue
        row = {"contrast": label}
        for metric in available_metrics:
            row[f"delta_{metric}"] = float(means.loc[variant, metric] - means.loc[baseline, metric])
        rows.append(row)
    return pd.DataFrame(rows)


def build_paper2_pilot_report(
    topology_csv: Path,
    topology_vs_allocation_md: Path,
    mediation_md: Path,
) -> str:
    df = pd.read_csv(topology_csv)
    successful = df[df["fit_failed"].astype(str).str.lower() != "true"].copy()
    run_level = successful.drop_duplicates(["objective", "seed", "model_name"])

    objectives = sorted(run_level["objective"].dropna().astype(str).unique())
    seeds = sorted(run_level["seed"].dropna().astype(int).unique())
    topology_modes = sorted(successful["topology_mode"].dropna().astype(str).unique())
    dataset_ids = sorted(successful["dataset_id"].dropna().astype(str).unique())
    n_positive_values = sorted(successful["n_positive"].dropna().astype(int).unique())
    minority_count = int(n_positive_values[0] * 2) if len(n_positive_values) == 1 else "unknown"

    objective_means = (
        run_level.groupby("objective", as_index=False)[SURVIVAL_COLUMNS + ALLOCATION_COLUMNS]
        .mean()
        .sort_values("objective")
    )
    dropout_contrasts = dropout_contrast_table(run_level)

    topology_md = topology_vs_allocation_md.read_text(encoding="utf-8")
    mediation_text = mediation_md.read_text(encoding="utf-8")
    topology_interpretation = _extract_interpretation_bullets(topology_md)
    mediation_interpretation = _extract_mediation_bullets(mediation_text)

    lines = [
        "# Paper 2 Pilot Summary: MLP Objective Perturbations, Allocation, And Accessibility",
        "",
        "## Research Question",
        "Do MLP objective perturbations alter minority accessibility survival geometry through score-allocation behavior?",
        "",
        "## Experimental Setup",
        f"- Dataset: {', '.join(dataset_ids) if dataset_ids else 'unknown'}",
        f"- Objectives: {', '.join(objectives)}",
        f"- Seeds: {min(seeds)}-{max(seeds)} ({len(seeds)} seeds)" if seeds else "- Seeds: unknown",
        "- Skew ratio: 100-to-1",
        f"- Minority count: {minority_count}",
        f"- Run-level observations: {len(run_level)}",
        f"- Topology modes: {', '.join(topology_modes)}",
        "- Metrics: survival AUC, persistence, cliffiness, positive score allocation shape, kNN topology, radius topology, and mediation-style ridge regression.",
        "",
        "## Key Results",
        "### Objective-Level Survival And Allocation Means",
        _markdown_table(objective_means),
        "",
        "### Allocation-vs-Topology Explanatory Comparison",
    ]
    lines.extend(topology_interpretation or ["- No interpretation bullets found in topology-vs-allocation summary."])
    lines.extend(
        [
            "",
            "### Mediation-Style Regression Summary",
        ]
    )
    lines.extend(mediation_interpretation or ["- See allocation mediation report for target-level regression details."])
    lines.extend(
        [
            "",
            "### Dropout As Feature-Subspace Sampling",
            "Dropout variants approximate feature-subspace sampling by training the sklearn MLP on Bernoulli-masked feature augmentations. The table reports mean differences relative to the matching non-dropout objective baseline.",
            _markdown_table(dropout_contrasts),
        ]
    )
    lines.extend(
        [
            "",
            "## Current Interpretation",
            "- Allocation is the stronger first-order explanation for the observed minority survival geometry in this pilot.",
            "- Topology is currently weak or non-incremental once score-allocation behavior is included.",
            "- Topology may require harder datasets, better representation spaces, or temporal/deployment settings to become informative.",
            "",
            "## What We Believe Now",
            "- MLP objective perturbations appear to change minority accessibility primarily by changing how positive-class scores are allocated across the score range.",
            "- Positive score concentration and bin occupancy explain survival AUC and cliffiness better than latent positive-support topology in this synthetic pilot.",
            "- Weighted BCE produces higher survival levels, while oversampling changes score allocation and survival geometry in a distinct way.",
            "",
            "## What Remains Unresolved",
            "- Whether latent topology becomes important on harder, multi-modal, or real enterprise datasets remains unresolved.",
            "- Whether representation topology is obscured by sklearn MLP embeddings rather than genuinely uninformative remains unresolved.",
            "- Whether calibration, threshold policy, or deployment-time drift mediates the same effects remains unresolved.",
            "",
            "## Threats To Validity",
            "- Small synthetic-only pilot.",
            f"- Only {len(run_level)} run-level observations.",
            "- sklearn MLP representation may be crude for representation-topology inference.",
            "- Persistence/survival metrics are threshold-grid dependent.",
            "- Correlations and mediation-style regressions are not causal evidence.",
            "",
            "## Next Experiments",
            "- Larger synthetic sweep across skew, separation, minority count, noise, and cluster structure.",
            "- Model-family sweep including boosted and tree models.",
            "- Temporal/enterprise dataset validation.",
            "- Score-allocation ablation or calibration experiment to directly perturb score concentration.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(
    topology_csv: Path,
    topology_vs_allocation_md: Path,
    mediation_md: Path,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_paper2_pilot_report(topology_csv, topology_vs_allocation_md, mediation_md),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology-csv",
        type=Path,
        default=ROOT / "results" / "topology" / "mlp_objectives_topology.csv",
    )
    parser.add_argument(
        "--topology-vs-allocation",
        type=Path,
        default=ROOT / "reports" / "topology" / "topology_vs_allocation_summary.md",
    )
    parser.add_argument(
        "--mediation",
        type=Path,
        default=ROOT / "reports" / "topology" / "allocation_mediation_summary.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "topology" / "paper2_pilot_summary.md",
    )
    args = parser.parse_args()
    output = write_report(
        args.topology_csv,
        args.topology_vs_allocation,
        args.mediation,
        args.output,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
