"""Test whether cliff-driving positives are rank effects or vulnerable cohorts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
DEFAULT_POSITIVE_SCORES = ROOT / "results" / "real_validation" / "hddt_positive_scores.csv"
OUT = ROOT / "reports" / "topology"

from report_hddt_dataset_accessibility_validation import _markdown_table  # noqa: E402
from report_minority_reachability_cohorts import largest_drop_transitions, load_positive_scores  # noqa: E402


RANK_BINS = np.linspace(0.0, 1.0, 11)
RANK_LABELS = [f"{int(100 * RANK_BINS[i])}-{int(100 * RANK_BINS[i + 1])}%" for i in range(len(RANK_BINS) - 1)]


def construct_positive_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Add ranks and largest-drop membership to per-positive run rows."""
    df = scores.copy()
    df["posterior_score"] = df["score"].astype(float)
    run_sizes = df.groupby("run_id")["posterior_score"].transform("size")
    df["posterior_rank"] = df.groupby("run_id")["posterior_score"].rank(method="average", ascending=True)
    df["posterior_rank_percentile"] = np.where(run_sizes > 1, (df["posterior_rank"] - 1.0) / (run_sizes - 1.0), 0.0)
    transitions, members = largest_drop_transitions(df)
    df = df.merge(transitions[["run_id", "drop_size"]], on="run_id", how="left")
    driver_keys = set(zip(members["run_id"], members["positive_key"])) if not members.empty else set()
    df["cliff_driver"] = [(run_id, key) in driver_keys for run_id, key in zip(df["run_id"], df["positive_key"])]
    df["drop_fraction"] = df["drop_size"].fillna(0.0).astype(float)
    df["cliffiness"] = df["minority_survival_cliffiness"].astype(float)
    df["rank_decile"] = pd.cut(df["posterior_rank_percentile"], bins=RANK_BINS, labels=RANK_LABELS, include_lowest=True)
    return df


def rank_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in [("all_positives", df), ("cliff_drivers", df[df["cliff_driver"]])]:
        rows.append(
            {
                "group": label,
                "n_rows": int(len(group)),
                "mean_score": float(group["posterior_score"].mean()),
                "median_score": float(group["posterior_score"].median()),
                "mean_rank_percentile": float(group["posterior_rank_percentile"].mean()),
                "median_rank_percentile": float(group["posterior_rank_percentile"].median()),
                "bottom_decile_fraction": float((group["posterior_rank_percentile"] <= 0.1).mean()),
                "top_decile_fraction": float((group["posterior_rank_percentile"] >= 0.9).mean()),
            }
        )
    return pd.DataFrame(rows)


def driver_frequency(df: pd.DataFrame) -> pd.DataFrame:
    freq = df.groupby(["dataset_id", "task_id", "positive_key"], as_index=False).agg(
        positive_id=("example_id", "first"),
        n_seen=("run_id", "nunique"),
        n_driver=("cliff_driver", "sum"),
        mean_score=("posterior_score", "mean"),
        median_score=("posterior_score", "median"),
        mean_rank_percentile=("posterior_rank_percentile", "mean"),
        median_rank_percentile=("posterior_rank_percentile", "median"),
        mean_breadth=("breadth", "mean"),
        mean_elevation=("elevation", "mean"),
        mean_cliffiness=("cliffiness", "mean"),
        n_models=("model_id", "nunique"),
        n_seeds=("seed", "nunique"),
    )
    freq["driver_frequency"] = freq["n_driver"] / freq["n_seen"].clip(lower=1)
    return freq.sort_values("driver_frequency", ascending=False).reset_index(drop=True)


def driver_frequency_summary(freq: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_positive_examples": int(len(freq)),
                "never_driver_fraction": float((freq["driver_frequency"] == 0.0).mean()),
                "occasional_driver_fraction": float(((freq["driver_frequency"] > 0.0) & (freq["driver_frequency"] < 0.8)).mean()),
                "persistent_driver_fraction": float((freq["driver_frequency"] > 0.8).mean()),
                "persistent_non_driver_fraction": float((freq["driver_frequency"] < 0.05).mean()),
                "mean_driver_frequency": float(freq["driver_frequency"].mean()),
                "median_driver_frequency": float(freq["driver_frequency"].median()),
            }
        ]
    )


def rank_controlled_rates(df: pd.DataFrame) -> pd.DataFrame:
    rates = df.groupby("rank_decile", observed=True).agg(
        n_rows=("cliff_driver", "size"),
        driver_rate=("cliff_driver", "mean"),
        mean_score=("posterior_score", "mean"),
        mean_breadth=("breadth", "mean"),
        mean_elevation=("elevation", "mean"),
        mean_cliffiness=("cliffiness", "mean"),
    )
    return rates.reset_index()


def bin_level_vulnerability(df: pd.DataFrame) -> pd.DataFrame:
    per_positive_bin = df.groupby(["rank_decile", "positive_key"], observed=True, as_index=False).agg(driver_frequency=("cliff_driver", "mean"), n_seen=("run_id", "nunique"))
    return per_positive_bin.groupby("rank_decile", observed=True, as_index=False).agg(
        n_positive_bin_entries=("positive_key", "size"),
        mean_positive_driver_frequency=("driver_frequency", "mean"),
        std_positive_driver_frequency=("driver_frequency", "std"),
        high_frequency_positive_fraction=("driver_frequency", lambda s: float((s > 0.8).mean())),
    )


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=True)
    total = len(y_true)
    value = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        value += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return value if total else np.nan


def predictive_models(df: pd.DataFrame, max_rows: int | None = 300_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_df = df.dropna(subset=["posterior_rank_percentile", "breadth", "elevation", "cliffiness", "cliff_driver"]).copy()
    if max_rows and len(model_df) > max_rows:
        model_df = model_df.sample(max_rows, random_state=7)
    y = model_df["cliff_driver"].astype(int).to_numpy()
    groups = model_df["task_id"].astype(str).to_numpy()
    specs = {
        "rank_only": ["posterior_rank_percentile"],
        "rank_plus_morphology": ["posterior_rank_percentile", "breadth", "elevation", "cliffiness"],
    }
    if len(np.unique(y)) < 2:
        return pd.DataFrame(), pd.DataFrame()
    n_groups = len(np.unique(groups))
    cv = GroupKFold(n_splits=min(5, n_groups)) if n_groups >= 2 else StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
    rows = []
    calibration_rows = []
    for name, features in specs.items():
        X = model_df[features].to_numpy(dtype=float)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
        kwargs = {"groups": groups} if isinstance(cv, GroupKFold) else {}
        prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba", **kwargs)[:, 1]
        rows.append(
            {
                "model": name,
                "n_rows": int(len(model_df)),
                "positive_rate": float(y.mean()),
                "auroc": float(roc_auc_score(y, prob)),
                "log_loss": float(log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6))),
                "ece_10bin": _ece(y, prob),
            }
        )
        true_frac, pred_frac = calibration_curve(y, prob, n_bins=10, strategy="quantile")
        for i, (truth, pred) in enumerate(zip(true_frac, pred_frac)):
            calibration_rows.append({"model": name, "bin": i, "mean_predicted_probability": float(pred), "observed_driver_rate": float(truth)})
    scores = pd.DataFrame(rows)
    if set(scores["model"]) == {"rank_only", "rank_plus_morphology"}:
        rank = scores[scores["model"] == "rank_only"].iloc[0]
        morph = scores[scores["model"] == "rank_plus_morphology"].iloc[0]
        scores["delta_auroc_vs_rank"] = scores["auroc"] - float(rank["auroc"])
        scores["delta_log_loss_vs_rank"] = scores["log_loss"] - float(rank["log_loss"])
        scores.loc[scores["model"] == "rank_plus_morphology", "morphology_adds_signal"] = bool((float(morph["auroc"]) - float(rank["auroc"])) >= 0.01 or (float(rank["log_loss"]) - float(morph["log_loss"])) >= 0.005)
    return scores, pd.DataFrame(calibration_rows)


def persistent_group_profiles(df: pd.DataFrame, freq: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = freq.assign(
        vulnerability_group=np.select(
            [freq["driver_frequency"] > 0.8, freq["driver_frequency"] < 0.05],
            ["persistent_driver", "persistent_non_driver"],
            default="intermediate",
        )
    )[["positive_key", "vulnerability_group", "driver_frequency"]]
    joined = df.merge(groups, on="positive_key", how="left")
    profiles = joined.groupby("vulnerability_group", as_index=False).agg(
        n_rows=("positive_key", "size"),
        n_positive_examples=("positive_key", "nunique"),
        mean_score=("posterior_score", "mean"),
        median_score=("posterior_score", "median"),
        mean_rank_percentile=("posterior_rank_percentile", "mean"),
        median_rank_percentile=("posterior_rank_percentile", "median"),
        mean_breadth=("breadth", "mean"),
        mean_elevation=("elevation", "mean"),
        mean_cliffiness=("cliffiness", "mean"),
        mean_driver_frequency=("driver_frequency", "mean"),
    )
    top_persistent = freq[freq["driver_frequency"] > 0.8].head(100)
    return profiles, top_persistent


def evidence_assessment(model_scores: pd.DataFrame, freq_summary: pd.DataFrame) -> str:
    if model_scores.empty:
        return "Predictability could not be evaluated because only one driver class was present."
    rank = model_scores[model_scores["model"] == "rank_only"].iloc[0]
    morph = model_scores[model_scores["model"] == "rank_plus_morphology"].iloc[0]
    persistent = float(freq_summary["persistent_driver_fraction"].iloc[0])
    auroc_gain = float(morph["auroc"] - rank["auroc"])
    logloss_gain = float(rank["log_loss"] - morph["log_loss"])
    if persistent >= 0.05 and (auroc_gain >= 0.01 or logloss_gain >= 0.005):
        return "Evidence favors H1: persistent vulnerable positives exist and morphology adds signal beyond posterior rank."
    if persistent < 0.01 and (auroc_gain >= 0.01 or logloss_gain >= 0.005):
        return "Evidence is mixed: rank alone is insufficient and morphology adds signal, but a stable >0.8-frequency vulnerable-positive cohort is not observed."
    if persistent < 0.01:
        return "Evidence favors H0 against a stable cohort: persistent drivers are absent and morphology adds little incremental signal beyond rank."
    return "Evidence is mixed: some persistence exists, but incremental morphology signal is limited after rank control."


def plot_rank_histogram(df: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["posterior_rank_percentile"], bins=30, density=True, alpha=0.45, label="all positives")
    ax.hist(df[df["cliff_driver"]]["posterior_rank_percentile"], bins=30, density=True, alpha=0.65, label="cliff drivers")
    ax.set_xlabel("posterior rank percentile (0=lowest score, 1=highest score)")
    ax.set_ylabel("density")
    ax.set_title("Rank Percentile Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_rank_ecdf(df: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, group in [("all positives", df), ("cliff drivers", df[df["cliff_driver"]])]:
        values = np.sort(group["posterior_rank_percentile"].to_numpy(dtype=float))
        y = np.arange(1, len(values) + 1) / max(1, len(values))
        ax.plot(values, y, label=label)
    ax.set_xlabel("posterior rank percentile")
    ax.set_ylabel("ECDF")
    ax.set_title("Rank Percentile ECDF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_driver_frequency(freq: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(freq["driver_frequency"], bins=np.linspace(0, 1, 31), color="slateblue", alpha=0.85)
    ax.set_xlabel("driver frequency")
    ax.set_ylabel("positive examples")
    ax.set_title("Persistent Accessibility-Vulnerable Positives")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_rank_controlled_rates(rates: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rates["rank_decile"].astype(str), rates["driver_rate"], marker="o")
    ax.set_xlabel("posterior rank percentile bin")
    ax.set_ylabel("P(cliff driver)")
    ax.set_title("Rank-Controlled Driver Rate")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_model_calibration(calibration: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    for model, group in calibration.groupby("model"):
        ax.plot(group["mean_predicted_probability"], group["observed_driver_rate"], marker="o", label=model)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed driver rate")
    ax.set_title("Driver Model Calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_persistent_profiles(profiles: pd.DataFrame, output: Path) -> Path:
    metrics = ["mean_score", "mean_rank_percentile", "mean_breadth", "mean_elevation", "mean_cliffiness"]
    available = profiles.set_index("vulnerability_group")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    width = 0.25
    for i, group in enumerate(["persistent_non_driver", "intermediate", "persistent_driver"]):
        if group not in available.index:
            continue
        ax.bar(x + (i - 1) * width, [available.loc[group, m] for m in metrics], width=width, label=group)
    ax.set_xticks(x, metrics, rotation=35, ha="right")
    ax.set_ylabel("mean value")
    ax.set_title("Persistent Driver vs Non-Driver Profiles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def write_reports(output_dir: Path = OUT, positive_scores: Path = DEFAULT_POSITIVE_SCORES, max_model_rows: int | None = 300_000) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = load_positive_scores(positive_scores)
    table = construct_positive_table(scores)
    rank_summary = rank_distribution_summary(table)
    freq = driver_frequency(table)
    freq_summary = driver_frequency_summary(freq)
    rates = rank_controlled_rates(table)
    bin_variability = bin_level_vulnerability(table)
    model_scores, calibration = predictive_models(table, max_rows=max_model_rows)
    profiles, top_persistent = persistent_group_profiles(table, freq)
    assessment = evidence_assessment(model_scores, freq_summary)

    table_path = output_dir / "accessibility_vulnerable_positive_table.csv"
    freq_path = output_dir / "accessibility_vulnerable_driver_frequency.csv"
    rates_path = output_dir / "accessibility_vulnerable_rank_controlled_rates.csv"
    model_path = output_dir / "accessibility_vulnerable_model_scores.csv"
    persistent_path = output_dir / "accessibility_vulnerable_persistent_examples.csv"
    summary_path = output_dir / "accessibility_vulnerable_positive_analysis.md"
    table.to_csv(table_path, index=False)
    freq.to_csv(freq_path, index=False)
    rates.to_csv(rates_path, index=False)
    model_scores.to_csv(model_path, index=False)
    top_persistent.to_csv(persistent_path, index=False)

    summary = pd.concat(
        [
            pd.DataFrame([{"metric": "n_positive_run_rows", "value": len(table)}]),
            pd.DataFrame([{"metric": "n_runs", "value": table["run_id"].nunique()}]),
            pd.DataFrame([{"metric": "driver_row_rate", "value": table["cliff_driver"].mean()}]),
            pd.DataFrame([{"metric": "n_unique_positives", "value": freq["positive_key"].nunique()}]),
        ],
        ignore_index=True,
    )
    sections = [
        "# Accessibility-Vulnerable Positive Analysis",
        "",
        "## Executive Answer",
        assessment,
        "",
        "Rank percentile is defined within each run as `0 = lowest posterior score` and `1 = highest posterior score`.",
        "",
        "## Aggregate Summary",
        _markdown_table(summary),
        "",
        "## Analysis 1: Rank Distribution",
        _markdown_table(rank_summary),
        "",
        "## Analysis 2: Driver Frequency",
        _markdown_table(freq_summary),
        "",
        "## Analysis 3: Rank-Controlled Driver Rate",
        _markdown_table(rates),
        "",
        "## Within-Bin Positive Heterogeneity",
        _markdown_table(bin_variability),
        "",
        "## Analysis 4: Predictability",
        _markdown_table(model_scores),
        "",
        "## Analysis 5: Persistent Driver Profiles",
        _markdown_table(profiles),
        "",
        "## Top Persistent Drivers",
        _markdown_table(top_persistent.head(30)),
        "",
        "## Caveat",
        "This analysis uses captured posterior scores and run-level morphology only. It does not yet include raw feature vectors or local topology component membership for each positive example.",
    ]
    summary_path.write_text("\n".join(sections) + "\n", encoding="utf-8")

    outputs: list[Path] = [summary_path, table_path, freq_path, rates_path, model_path, persistent_path]
    outputs.extend(
        [
            plot_rank_histogram(table, output_dir / "accessibility_vulnerable_rank_histogram.png"),
            plot_rank_ecdf(table, output_dir / "accessibility_vulnerable_rank_ecdf.png"),
            plot_driver_frequency(freq, output_dir / "accessibility_vulnerable_driver_frequency.png"),
            plot_rank_controlled_rates(rates, output_dir / "accessibility_vulnerable_rank_controlled_rates.png"),
            plot_persistent_profiles(profiles, output_dir / "accessibility_vulnerable_persistent_profiles.png"),
        ]
    )
    if not calibration.empty:
        calibration_path = output_dir / "accessibility_vulnerable_model_calibration.csv"
        calibration.to_csv(calibration_path, index=False)
        outputs.append(calibration_path)
        outputs.append(plot_model_calibration(calibration, output_dir / "accessibility_vulnerable_model_calibration.png"))
    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-scores", type=Path, default=DEFAULT_POSITIVE_SCORES)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--max-model-rows", type=int, default=300_000, help="Subsample rows for predictive modeling; use 0 for all rows.")
    args = parser.parse_args()
    max_rows = None if args.max_model_rows == 0 else args.max_model_rows
    for output in write_reports(args.output_dir, args.positive_scores, max_model_rows=max_rows):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
