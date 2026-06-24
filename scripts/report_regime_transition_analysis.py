"""Analyze transitions through accessibility morphology regimes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_hdbscan_umap_robustness import consolidate_hdbscan, hdbscan_labels, native_embedding  # noqa: E402
from report_morphology_atlas import DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, TARGET, SURVIVAL, _markdown_table, load_classical_atlas, prepare_feature_space  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "topology"
INPUTS = {
    "allocation_trajectory": RESULTS / "allocation_trajectory.csv",
    "weighted_dropout_sweep_dense": RESULTS / "weighted_dropout_sweep_dense.csv",
    "density_threshold_sweep": RESULTS / "density_threshold_sweep.csv",
    "density_separability_factorial_dense": RESULTS / "density_separability_factorial_dense.csv",
    "fragmentation_sweep": RESULTS / "fragmentation_sweep.csv",
    "mlp_objectives_topology": RESULTS / "mlp_objectives_topology.csv",
    "classical_allocator_morphology": RESULTS / "classical_allocator_morphology.csv",
}


def safe_numeric(df: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def normalize_base(df: pd.DataFrame, family: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["source_file"] = family
    out["family"] = family
    out["model_id"] = df.get("model_id", df.get("objective", df.get("model_name", "unknown"))).astype(str) if any(c in df.columns for c in ["model_id", "objective", "model_name"]) else "unknown"
    out["seed"] = safe_numeric(df, "seed", 0).fillna(0).astype(int)
    out["skew_ratio"] = safe_numeric(df, "skew_ratio")
    out["breadth"] = safe_numeric(df, "breadth")
    out["elevation"] = safe_numeric(df, "elevation")
    out[TARGET] = safe_numeric(df, TARGET)
    out[SURVIVAL] = safe_numeric(df, SURVIVAL)
    out["auroc"] = safe_numeric(df, "auroc")
    out["average_precision"] = safe_numeric(df, "average_precision")
    return out


def add_transition_columns(df: pd.DataFrame, intervention: str, trajectory_id: pd.Series, step_value: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["intervention"] = intervention
    out["trajectory_id"] = pd.Series(trajectory_id).astype(str).fillna("missing_trajectory").to_numpy()
    out["step_value"] = pd.to_numeric(pd.Series(step_value), errors="coerce").to_numpy()
    out = out.dropna(subset=["breadth", "elevation", TARGET, SURVIVAL, "step_value"]).copy()
    out["trajectory_id"] = out["trajectory_id"].fillna("missing_trajectory").astype(str)
    out = out.sort_values(["trajectory_id", "step_value"]).reset_index(drop=True)
    out["step_index"] = out.groupby("trajectory_id").cumcount().astype(int)
    return out


def load_transition_rows(paths: dict[str, Path] = INPUTS) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    notes = []
    for family, path in paths.items():
        if not path.exists():
            notes.append(f"missing input: {path}")
            continue
        raw = pd.read_csv(path)
        if "fit_failed" in raw.columns:
            raw = raw[~raw["fit_failed"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
        if family == "mlp_objectives_topology":
            raw = raw.rename(columns={"positive_histogram_entropy": "breadth", "positive_top_bin_mass": "elevation"})
            notes.append("mlp_objectives_topology used positive_histogram_entropy as breadth and positive_top_bin_mass as elevation, matching prior morphology scripts.")
        base = normalize_base(raw, family)
        if family == "allocation_trajectory":
            tid = raw["objective"].astype(str) + "|seed=" + raw["seed"].astype(str)
            frames.append(add_transition_columns(base, "training epoch", tid, raw["epoch"]))
        elif family == "weighted_dropout_sweep_dense":
            tid = "skew=" + raw["skew_ratio"].astype(str) + "|seed=" + raw["seed"].astype(str)
            frames.append(add_transition_columns(base, "dropout increase", tid, raw["dropout_rate"]))
        elif family in {"density_threshold_sweep", "density_separability_factorial_dense"}:
            tid_density = family + "|model=" + raw["model_id"].astype(str) + "|seed=" + raw["seed"].astype(str) + "|dist=" + raw["centroid_distance"].astype(str)
            frames.append(add_transition_columns(base, "density increase", tid_density, raw["minority_cov"]))
            tid_sep = family + "|model=" + raw["model_id"].astype(str) + "|seed=" + raw["seed"].astype(str) + "|cov=" + raw["minority_cov"].astype(str)
            frames.append(add_transition_columns(base, "separability increase", tid_sep, raw["centroid_distance"]))
        elif family == "fragmentation_sweep":
            tid = raw["model_id"].astype(str) + "|seed=" + raw["seed"].astype(str) + "|skew=" + raw["skew_ratio"].astype(str)
            frames.append(add_transition_columns(base, "fragmentation increase", tid, raw["n_islands"]))
        elif family == "mlp_objectives_topology":
            objective_order = {"bce": 0, "bce_dropout_0_1": 1, "bce_dropout_0_3": 2, "weighted_bce": 3, "weighted_bce_dropout_0_1": 4, "weighted_bce_dropout_0_3": 5, "oversampled_bce": 6}
            raw_step = raw["objective"].map(objective_order)
            tid = raw["dataset_id"].astype(str) + "|seed=" + raw["seed"].astype(str) + "|split=" + raw["split_id"].astype(str) + "|mode=" + raw["topology_mode"].astype(str) + "|k=" + raw["k"].fillna("na").astype(str)
            frames.append(add_transition_columns(base, "objective intervention", tid, raw_step))
        elif family == "classical_allocator_morphology":
            model_order = {"cart": 0, "hddt": 1, "bagged_hddt": 2, "random_forest": 3, "xgboost": 4, "lightgbm": 5}
            raw_step = raw["model_id"].map(model_order)
            tid = "seed=" + raw["seed"].astype(str) + "|skew=" + raw["skew_ratio"].astype(str)
            frames.append(add_transition_columns(base, "allocator family change", tid, raw_step))
    pooled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    pooled["row_id"] = np.arange(len(pooled))
    return pooled, notes


def train_native_regime_classifier(input_csv: Path = DEFAULT_INPUT):
    df = load_classical_atlas(input_csv)
    enriched, _, _, X_pca, _, _ = prepare_feature_space(df)
    labels, _ = hdbscan_labels(X_pca, min_cluster_size=15, min_samples=5)
    atlas = enriched.copy().reset_index(drop=True)
    atlas["row_id"] = np.arange(len(atlas))
    atlas["hdbscan_cluster"] = labels
    result = consolidate_hdbscan(atlas)
    mapped = result["mapped"].copy()
    characterization = result["characterization"].copy()
    label_map = {int(row.regime_id): f"{row.interpretation} ({int(row.regime_id)})" for row in characterization.itertuples()}
    train = mapped.dropna(subset=["regime_id", "breadth", "elevation"]).copy()
    model = RandomForestClassifier(n_estimators=250, min_samples_leaf=4, random_state=42)
    model.fit(train[["breadth", "elevation"]], train["regime_id"].astype(int))
    return model, label_map, mapped, result


def assign_regimes(pooled: pd.DataFrame, model, label_map: dict[int, str], native_mapped: pd.DataFrame) -> pd.DataFrame:
    out = pooled.copy()
    out["regime_id"] = model.predict(out[["breadth", "elevation"]]).astype(int)
    out["assigned_by"] = "classifier_projection"
    out["regime_label"] = out["regime_id"].map(label_map).fillna(out["regime_id"].astype(str))
    native = native_mapped[["allocation_family", "seed", "skew_ratio", "breadth", "elevation", "regime_id"]].copy()
    native = native.rename(columns={"allocation_family": "model_id", "regime_id": "native_regime_id"})
    keys = ["model_id", "seed", "skew_ratio", "breadth", "elevation"]
    merged = out.merge(native, on=keys, how="left")
    mask = merged["native_regime_id"].notna() & merged["source_file"].eq("classical_allocator_morphology")
    merged.loc[mask, "regime_id"] = merged.loc[mask, "native_regime_id"].astype(int)
    merged.loc[mask, "assigned_by"] = "native_cluster"
    merged["regime_label"] = merged["regime_id"].astype(int).map(label_map).fillna(merged["regime_id"].astype(str))
    return merged.drop(columns=["native_regime_id"])


def build_step_transitions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trajectory_id, group in df.sort_values(["trajectory_id", "step_index", "step_value"]).groupby("trajectory_id"):
        group = group.drop_duplicates(subset=["step_value"], keep="first").sort_values("step_value")
        if len(group) < 2:
            continue
        records = list(group.itertuples(index=False))
        for idx, (a, b) in enumerate(zip(records[:-1], records[1:])):
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "family": a.family,
                    "intervention": a.intervention,
                    "model_id": a.model_id,
                    "seed": int(a.seed),
                    "skew_ratio": float(a.skew_ratio) if pd.notna(a.skew_ratio) else np.nan,
                    "from_regime": int(a.regime_id),
                    "to_regime": int(b.regime_id),
                    "from_label": a.regime_label,
                    "to_label": b.regime_label,
                    "step_from": float(a.step_value),
                    "step_to": float(b.step_value),
                    "step_delta": float(b.step_value - a.step_value),
                    "regime_changed": int(a.regime_id != b.regime_id),
                    "morphology_distance": float(np.hypot(b.breadth - a.breadth, b.elevation - a.elevation)),
                    "delta_breadth": float(b.breadth - a.breadth),
                    "delta_elevation": float(b.elevation - a.elevation),
                    "delta_cliffiness": float(getattr(b, TARGET) - getattr(a, TARGET)),
                    "delta_survival_auc": float(getattr(b, SURVIVAL) - getattr(a, SURVIVAL)),
                    "edge_index": idx,
                }
            )
    return pd.DataFrame(rows)


def build_path_summary(df: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trajectory_id, group in df.sort_values(["trajectory_id", "step_value"]).groupby("trajectory_id"):
        group = group.drop_duplicates(subset=["step_value"], keep="first").sort_values("step_value")
        if len(group) < 2:
            continue
        path_steps = steps[steps["trajectory_id"].eq(trajectory_id)]
        first = group.iloc[0]
        last = group.iloc[-1]
        changed = path_steps[path_steps["regime_changed"].eq(1)]
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "family": first.family,
                "intervention": first.intervention,
                "model_id": first.model_id,
                "seed": int(first.seed),
                "skew_ratio": float(first.skew_ratio) if pd.notna(first.skew_ratio) else np.nan,
                "start_regime": int(first.regime_id),
                "end_regime": int(last.regime_id),
                "transition_edges": ";".join(f"{int(row.from_regime)}->{int(row.to_regime)}" for row in changed.itertuples()),
                "number_of_regime_changes": int(path_steps["regime_changed"].sum()),
                "first_transition_step": float(changed.iloc[0].step_to) if not changed.empty else np.nan,
                "total_path_length_morphology": float(path_steps["morphology_distance"].sum()),
                "delta_breadth": float(last.breadth - first.breadth),
                "delta_elevation": float(last.elevation - first.elevation),
                "delta_cliffiness": float(last[TARGET] - first[TARGET]),
                "delta_survival_auc": float(last[SURVIVAL] - first[SURVIVAL]),
            }
        )
    return pd.DataFrame(rows)


def transition_edges(steps: pd.DataFrame) -> pd.DataFrame:
    changed = steps[steps["regime_changed"].eq(1)].copy()
    if changed.empty:
        return pd.DataFrame(columns=["from_regime", "to_regime", "count"])
    rows = []
    for (src, dst), group in changed.groupby(["from_regime", "to_regime"]):
        dominant = group["intervention"].value_counts().idxmax()
        rows.append(
            {
                "from_regime": int(src),
                "to_regime": int(dst),
                "count": int(len(group)),
                "mean_delta_cliffiness": float(group["delta_cliffiness"].mean()),
                "mean_delta_survival_auc": float(group["delta_survival_auc"].mean()),
                "mean_delta_breadth": float(group["delta_breadth"].mean()),
                "mean_delta_elevation": float(group["delta_elevation"].mean()),
                "dominant_experiment_family": dominant,
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def intervention_summary(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for intervention, group in paths.groupby("intervention"):
        rows.append(
            {
                "intervention": intervention,
                "most_common_start_regime": int(group["start_regime"].mode().iloc[0]),
                "most_common_end_regime": int(group["end_regime"].mode().iloc[0]),
                "transition_probability": float((group["number_of_regime_changes"] > 0).mean()),
                "mean_delta_cliffiness": float(group["delta_cliffiness"].mean()),
                "mean_delta_survival_auc": float(group["delta_survival_auc"].mean()),
                "n_paths": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("transition_probability", ascending=False)


def stability_summary(steps: pd.DataFrame) -> pd.DataFrame:
    if steps.empty:
        return pd.DataFrame()
    out = steps.copy()
    out["step_size"] = "small"
    for intervention, idx in out.groupby("intervention").groups.items():
        vals = out.loc[idx, "step_delta"].abs()
        q1, q2 = vals.quantile([0.33, 0.66])
        out.loc[idx, "step_size"] = np.where(vals <= q1, "small", np.where(vals <= q2, "medium", "large"))
    rows = []
    for (intervention, step_size), group in out.groupby(["intervention", "step_size"]):
        rows.append(
            {
                "intervention": intervention,
                "step_size": step_size,
                "p_same_regime": float((group["regime_changed"] == 0).mean()),
                "p_regime_change": float((group["regime_changed"] == 1).mean()),
                "mean_morphology_distance": float(group["morphology_distance"].mean()),
                "mean_cliffiness_change": float(group["delta_cliffiness"].abs().mean()),
                "n_steps": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def transition_events(steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if steps.empty:
        return pd.DataFrame(), pd.DataFrame()
    morph_thr = float(steps["morphology_distance"].quantile(0.90))
    cliff_thr = float(steps["delta_cliffiness"].abs().quantile(0.90))
    events = steps[(steps["regime_changed"].eq(1)) | (steps["morphology_distance"] >= morph_thr) | (steps["delta_cliffiness"].abs() >= cliff_thr)].copy()
    summary = events.groupby("intervention", as_index=False).agg(
        transition_event_count=("trajectory_id", "size"),
        median_step_value=("step_to", "median"),
        mean_morphology_distance=("morphology_distance", "mean"),
        mean_abs_delta_cliffiness=("delta_cliffiness", lambda s: float(np.abs(s).mean())),
    )
    return events, summary


def transition_r2(paths: pd.DataFrame) -> pd.DataFrame:
    data = paths.dropna(subset=["delta_cliffiness", "delta_breadth", "delta_elevation", "start_regime", "end_regime"]).copy()
    if len(data) < 10:
        return pd.DataFrame()
    y = data["delta_cliffiness"]
    morph = data[["delta_breadth", "delta_elevation"]]
    regime = pd.get_dummies(data[["start_regime", "end_regime"]].astype(str))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, X in [("morphology_delta", morph), ("regime_transition", regime), ("morphology_delta_plus_regime", pd.concat([morph, regime], axis=1))]:
        pred = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), X, y, cv=cv)
        rows.append({"feature_set": name, "cv_r2_delta_cliffiness": float(r2_score(y, pred))})
    return pd.DataFrame(rows)


def plot_transition_graph(edges: pd.DataFrame, label_map: dict[int, str], output: Path) -> Path:
    regimes = sorted(set(edges["from_regime"]).union(set(edges["to_regime"]))) if not edges.empty else sorted(label_map)
    angles = np.linspace(0, 2 * np.pi, len(regimes), endpoint=False) if regimes else []
    pos = {r: (np.cos(a), np.sin(a)) for r, a in zip(regimes, angles)}
    fig, ax = plt.subplots(figsize=(7, 7))
    for r, (x, y) in pos.items():
        ax.scatter([x], [y], s=900, color="lightsteelblue", edgecolor="black", zorder=3)
        ax.text(x, y, str(r), ha="center", va="center", fontsize=12, weight="bold")
    if not edges.empty:
        max_count = max(edges["count"].max(), 1)
        for row in edges.itertuples(index=False):
            x1, y1 = pos[int(row.from_regime)]
            x2, y2 = pos[int(row.to_regime)]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1 + 4 * row.count / max_count, "alpha": 0.65, "color": "firebrick"})
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(int(row.count)), fontsize=8)
    ax.set_title("Directed Regime Transition Graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_paths_morphology(df: pd.DataFrame, steps: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["breadth"], df["elevation"], c=df["regime_id"], cmap="tab10", s=12, alpha=0.25)
    sample = steps[steps["regime_changed"].eq(1)].head(300)
    for row in sample.itertuples(index=False):
        a = df[(df["trajectory_id"].eq(row.trajectory_id)) & (df["step_value"].eq(row.step_from))].head(1)
        b = df[(df["trajectory_id"].eq(row.trajectory_id)) & (df["step_value"].eq(row.step_to))].head(1)
        if not a.empty and not b.empty:
            ax.annotate("", xy=(float(b.iloc[0].breadth), float(b.iloc[0].elevation)), xytext=(float(a.iloc[0].breadth), float(a.iloc[0].elevation)), arrowprops={"arrowstyle": "->", "alpha": 0.25, "lw": 0.8})
    ax.set_xlabel("breadth")
    ax.set_ylabel("elevation")
    ax.set_title("Regime Transition Paths in Morphology Space")
    fig.colorbar(sc, ax=ax, label="regime_id")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_paths_umap(df: pd.DataFrame, steps: pd.DataFrame, output: Path) -> Path:
    coords = native_embedding(StandardScaler().fit_transform(df[["breadth", "elevation"]].to_numpy(dtype=float))) if len(df) > 20 else df[["breadth", "elevation"]].to_numpy(dtype=float)
    plot_df = df.copy()
    plot_df["umap_x"] = coords[:, 0]
    plot_df["umap_y"] = coords[:, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(plot_df["umap_x"], plot_df["umap_y"], c=plot_df["regime_id"], cmap="tab10", s=12, alpha=0.35)
    for row in steps[steps["regime_changed"].eq(1)].head(300).itertuples(index=False):
        a = plot_df[(plot_df["trajectory_id"].eq(row.trajectory_id)) & (plot_df["step_value"].eq(row.step_from))].head(1)
        b = plot_df[(plot_df["trajectory_id"].eq(row.trajectory_id)) & (plot_df["step_value"].eq(row.step_to))].head(1)
        if not a.empty and not b.empty:
            ax.annotate("", xy=(float(b.iloc[0].umap_x), float(b.iloc[0].umap_y)), xytext=(float(a.iloc[0].umap_x), float(a.iloc[0].umap_y)), arrowprops={"arrowstyle": "->", "alpha": 0.22, "lw": 0.8})
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Regime Transition Paths in UMAP Space")
    fig.colorbar(sc, ax=ax, label="regime_id")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_delta_cliffiness(paths: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = [(name, group["delta_cliffiness"].dropna().to_numpy()) for name, group in paths.groupby("intervention")]
    ax.boxplot([values for _, values in groups], labels=[name for name, _ in groups], vert=True)
    ax.axhline(0, color="black", lw=1, alpha=0.5)
    ax.set_ylabel("delta cliffiness")
    ax.set_title("Path-Level Cliffiness Change by Intervention")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_stability(stability: pd.DataFrame, output: Path) -> Path:
    pivot = stability.pivot_table(index="intervention", columns="step_size", values="p_regime_change", aggfunc="mean").fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="magma", vmin=0, vmax=max(1e-6, float(pivot.max().max())))
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Regime Change Probability by Intervention and Step Size")
    fig.colorbar(im, ax=ax, label="P(regime change)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def classify_success(interventions: pd.DataFrame, r2: pd.DataFrame) -> str:
    moving = interventions[interventions["transition_probability"] >= 0.25] if not interventions.empty else pd.DataFrame()
    if not r2.empty:
        lookup = r2.set_index("feature_set")["cv_r2_delta_cliffiness"]
        if lookup.get("morphology_delta_plus_regime", -np.inf) > lookup.get("morphology_delta", np.inf) + 0.05:
            return "Exceptional Success"
    if len(moving) >= 2:
        return "Strong Success"
    if len(moving) >= 1:
        return "Weak Success"
    return "Unsuccessful"


def write_report(output_dir: Path, paths: dict[str, Path] = INPUTS) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pooled, notes = load_transition_rows(paths)
    model, label_map, native_mapped, native_result = train_native_regime_classifier(DEFAULT_INPUT)
    assigned = assign_regimes(pooled, model, label_map, native_mapped)
    steps = build_step_transitions(assigned)
    path_summary = build_path_summary(assigned, steps)
    edges = transition_edges(steps)
    interventions = intervention_summary(path_summary)
    stability = stability_summary(steps)
    events, event_summary = transition_events(steps)
    r2 = transition_r2(path_summary)
    success = classify_success(interventions, r2)

    outputs = []
    assigned_csv = output_dir / "regime_transition_assigned_runs.csv"
    assigned.to_csv(assigned_csv, index=False)
    outputs.append(assigned_csv)
    path_csv = output_dir / "regime_transition_paths.csv"
    path_summary.to_csv(path_csv, index=False)
    outputs.append(path_csv)
    edge_csv = output_dir / "regime_transition_edges.csv"
    edges.to_csv(edge_csv, index=False)
    outputs.append(edge_csv)
    outputs.append(plot_transition_graph(edges, label_map, output_dir / "regime_transition_graph.png"))
    outputs.append(plot_paths_morphology(assigned, steps, output_dir / "regime_transition_paths_morphology_space.png"))
    outputs.append(plot_paths_umap(assigned, steps, output_dir / "regime_transition_paths_umap_space.png"))
    outputs.append(plot_delta_cliffiness(path_summary, output_dir / "regime_transition_delta_cliffiness.png"))
    outputs.append(plot_stability(stability, output_dir / "regime_stability_by_intervention.png"))

    axes = pd.DataFrame(
        [
            {"input": "allocation_trajectory", "axis": "epoch", "intervention": "training epoch"},
            {"input": "weighted_dropout_sweep_dense", "axis": "dropout_rate", "intervention": "dropout increase"},
            {"input": "density_threshold_sweep", "axis": "minority_cov and centroid_distance", "intervention": "density increase; separability increase"},
            {"input": "density_separability_factorial_dense", "axis": "minority_cov and centroid_distance", "intervention": "density increase; separability increase"},
            {"input": "fragmentation_sweep", "axis": "n_islands", "intervention": "fragmentation increase"},
            {"input": "mlp_objectives_topology", "axis": "objective order proxy", "intervention": "objective intervention"},
            {"input": "classical_allocator_morphology", "axis": "model family order proxy", "intervention": "allocator family change"},
        ]
    )
    report = output_dir / "regime_transition_analysis.md"
    text = "\n".join(
        [
            "# Regime Transition Analysis",
            "",
            f"Overall classification: **{success}**",
            "",
            "## 1. Inputs Used",
            _markdown_table(pd.DataFrame([{"input": key, "path": str(path), "exists": path.exists()} for key, path in paths.items()])),
            "",
            "## 2. Regime Assignment Method",
            "Native HDBSCAN labels were reconstructed on the classical allocator atlas and consolidated to the current best 4-regime vocabulary. A RandomForestClassifier trained on native classical breadth/elevation labels projected all pooled experiment rows into that vocabulary. Exact classical rows matched to native labels are marked `native_cluster`; all other rows are marked `classifier_projection`.",
            "",
            "Regime labels:",
            _markdown_table(pd.DataFrame([{"regime_id": key, "regime_label": value} for key, value in sorted(label_map.items())])),
            "",
            "Notes:",
            "\n".join(f"- {note}" for note in notes) if notes else "- No input caveats.",
            "",
            "## 3. Transition Axes",
            _markdown_table(axes),
            "",
            "## 4. Transition Graph Summary",
            _markdown_table(edges.head(20)),
            "",
            "## 5. Intervention-Level Transition Effects",
            _markdown_table(interventions),
            "",
            "## 6. Regime Stability",
            _markdown_table(stability),
            "",
            "## 7. Transition Event Findings",
            _markdown_table(event_summary),
            "",
            "## Delta Cliffiness Explanation Check",
            _markdown_table(r2),
            "",
            "## 8. Manuscript Implications",
            manuscript_implication(success),
            "",
            "## 9. Limitations",
            "These are observational paths through controlled sweeps, not randomized causal interventions. Objective and allocator-family axes use pragmatic order proxies and should be treated as exploratory. Regime labels outside the classical atlas are classifier projections from breadth/elevation, not native HDBSCAN assignments.",
        ]
    ) + "\n"
    report.write_text(text, encoding="utf-8")
    outputs.append(report)
    return tuple(outputs)


def manuscript_implication(success: str) -> str:
    if success == "Exceptional Success":
        return "Training and data perturbations induce structured movement through accessibility regime space, and regime transitions add predictive value for cliffiness changes beyond continuous morphology deltas. This may justify an appendix or exploratory main-text subsection."
    if success == "Strong Success":
        return "Training and data perturbations induce structured movement through accessibility regime space. This is best framed as exploratory appendix evidence unless further causal controls are added."
    if success == "Weak Success":
        return "At least one intervention family shows non-random regime movement, but transition dynamics remain intervention-specific. Best placement is Future Work or exploratory appendix."
    return "Regimes summarize static accessibility behavior, but current transition data are insufficient to infer dynamics. Keep this as Future Work."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for path in write_report(args.output_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
