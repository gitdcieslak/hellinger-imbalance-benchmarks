"""Run accessibility-regime validation on HDDT/UCI-style benchmark datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
for module_name in list(sys.modules):
    if module_name == "hib" or module_name.startswith("hib."):
        del sys.modules[module_name]

from hib.allocation_shape import positive_score_allocation_shape_metrics  # noqa: E402
from hib.models import OptionalDependencyUnavailable, make_model  # noqa: E402
from hib.occupancy import compute_occupancy_metrics, empirical_reachability_curve  # noqa: E402


RESULTS_DIR = ROOT / "results" / "real_validation"
DEFAULT_INVENTORY = RESULTS_DIR / "hddt_dataset_inventory.csv"
DEFAULT_TASKS = RESULTS_DIR / "hddt_task_inventory.csv"
DEFAULT_OUTPUT = RESULTS_DIR / "hddt_accessibility_validation.csv"
THRESHOLDS = np.linspace(0.0, 1.0, 101).tolist()
DEFAULT_MODELS = "cart,random_forest,xgboost,lightgbm,logistic_regression"


def _json_counter(values: pd.Series) -> str:
    counts = Counter(values.astype(str))
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def data_files(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.rglob("*.data") if not p.name.startswith("._") and p.stat().st_size > 0)


def matching_names(path: Path) -> Path | None:
    names = path.with_suffix(".names")
    return names if names.exists() and not names.name.startswith("._") else None


def detect_format(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("|")]
    if not lines:
        return "empty"
    sample = lines[0]
    if ":" in sample and "," not in sample:
        return "libsvm"
    if "," in sample:
        return "csv"
    return "whitespace"


def read_raw_data(path: Path) -> pd.DataFrame:
    fmt = detect_format(path)
    if fmt == "empty":
        raise ValueError("empty file")
    if fmt == "libsvm":
        from sklearn.datasets import load_svmlight_file

        X, y = load_svmlight_file(str(path))
        df = pd.DataFrame.sparse.from_spmatrix(X)
        df["target"] = y
        return df
    sep = "," if fmt == "csv" else r"\s+"
    return pd.read_csv(path, header=None, sep=sep, engine="python", na_values=["?", "NA", "nan", ""], comment="|")


def profile_dataset(path: Path) -> dict[str, Any]:
    names = matching_names(path)
    row = {
        "dataset_name": path.stem,
        "data_path": str(path),
        "names_path": str(names) if names else "",
        "file_size": int(path.stat().st_size),
        "detected_format": detect_format(path),
        "target_column": "last",
        "usable": False,
        "skip_reason": "",
    }
    try:
        df = read_raw_data(path)
        if df.shape[1] < 2:
            raise ValueError("fewer than 2 columns")
        y = df.iloc[:, -1]
        counts = Counter(y.dropna().astype(str))
        row.update(
            {
                "n_rows": int(len(df)),
                "n_columns": int(df.shape[1]),
                "n_classes": int(len(counts)),
                "class_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
                "minority_fraction": float(min(counts.values()) / max(1, sum(counts.values()))) if counts else np.nan,
                "usable": bool(len(df) > 0 and len(counts) >= 2),
                "skip_reason": "" if len(counts) >= 2 else "fewer than 2 classes",
            }
        )
    except Exception as exc:
        row.update({"n_rows": 0, "n_columns": 0, "n_classes": 0, "class_counts": "{}", "minority_fraction": np.nan, "skip_reason": str(exc)})
    return row


def discover_datasets(data_root: Path, output_csv: Path = DEFAULT_INVENTORY) -> pd.DataFrame:
    rows = [profile_dataset(path) for path in data_files(data_root)]
    inventory = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_csv, index=False)
    return inventory


def preprocess_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].astype(str)
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    return X, y


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = []
    categorical_cols = []
    for col in X.columns:
        numeric = pd.to_numeric(X[col], errors="coerce")
        if numeric.notna().mean() >= 0.85:
            X[col] = numeric
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols))
    return ColumnTransformer(transformers, remainder="drop")


def build_tasks(inventory: pd.DataFrame, min_rows: int, min_positives: int, max_positive_fraction: float, output_csv: Path = DEFAULT_TASKS) -> pd.DataFrame:
    rows = []
    for item in inventory.itertuples(index=False):
        if not bool(item.usable):
            rows.append({"task_id": f"{item.dataset_name}:skip", "dataset_name": item.dataset_name, "positive_label": "", "n_rows": getattr(item, "n_rows", 0), "n_features_raw": 0, "n_features_processed": 0, "positive_count": 0, "positive_fraction": np.nan, "task_type": "skip", "usable": False, "skip_reason": item.skip_reason})
            continue
        try:
            df = read_raw_data(Path(item.data_path))
            X, y = preprocess_frame(df)
            counts = y.value_counts()
            if len(df) < min_rows:
                rows.append({"task_id": f"{item.dataset_name}:skip", "dataset_name": item.dataset_name, "positive_label": "", "n_rows": len(df), "n_features_raw": X.shape[1], "n_features_processed": 0, "positive_count": 0, "positive_fraction": np.nan, "task_type": "skip", "usable": False, "skip_reason": f"n_rows < {min_rows}"})
                continue
            candidates = counts[counts >= min_positives]
            candidates = candidates[(candidates / len(y)) <= max_positive_fraction]
            if candidates.empty:
                rows.append({"task_id": f"{item.dataset_name}:skip", "dataset_name": item.dataset_name, "positive_label": "", "n_rows": len(df), "n_features_raw": X.shape[1], "n_features_processed": 0, "positive_count": 0, "positive_fraction": float(counts.min() / len(y)), "task_type": "skip", "usable": False, "skip_reason": "no class satisfies positive-count/fraction rules"})
                continue
            task_type = "binary_minority" if len(counts) == 2 else "one_vs_rest"
            labels = [counts.idxmin()] if len(counts) == 2 else list(candidates.sort_values().index)
            pre = make_preprocessor(X.copy())
            try:
                n_processed = int(pre.fit_transform(X).shape[1])
            except Exception:
                n_processed = 0
            for label in labels:
                positive_count = int((y == label).sum())
                rows.append({"task_id": f"{item.dataset_name}:{label}", "dataset_name": item.dataset_name, "positive_label": str(label), "n_rows": len(df), "n_features_raw": X.shape[1], "n_features_processed": n_processed, "positive_count": positive_count, "positive_fraction": float(positive_count / len(df)), "task_type": task_type, "usable": True, "skip_reason": ""})
        except Exception as exc:
            rows.append({"task_id": f"{item.dataset_name}:skip", "dataset_name": item.dataset_name, "positive_label": "", "n_rows": 0, "n_features_raw": 0, "n_features_processed": 0, "positive_count": 0, "positive_fraction": np.nan, "task_type": "skip", "usable": False, "skip_reason": str(exc)})
    tasks = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(output_csv, index=False)
    return tasks


def model_factory(model_id: str, seed: int):
    if model_id == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed)
    if model_id == "mlp_weighted_bce":
        return make_model("mlp_weighted", seed)
    return make_model(model_id, seed)


def predict_scores(model, X_test) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X_test))[:, 1]
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X_test), dtype=float)
        return (raw - raw.min()) / max(1e-9, raw.max() - raw.min())
    pred = np.asarray(model.predict(X_test), dtype=float)
    return pred


def fit_regime_projector():
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from report_hdbscan_umap_robustness import consolidate_hdbscan, hdbscan_labels
        from report_morphology_atlas import DEFAULT_INPUT, load_classical_atlas, prepare_feature_space
        from sklearn.ensemble import RandomForestClassifier

        df = load_classical_atlas(DEFAULT_INPUT)
        enriched, _, _, X_pca, _, _ = prepare_feature_space(df)
        labels, _ = hdbscan_labels(X_pca, min_cluster_size=15, min_samples=5)
        atlas = enriched.copy().reset_index(drop=True)
        atlas["row_id"] = np.arange(len(atlas))
        atlas["hdbscan_cluster"] = labels
        result = consolidate_hdbscan(atlas)
        mapped = result["mapped"].dropna(subset=["regime_id"])
        clf = RandomForestClassifier(n_estimators=250, min_samples_leaf=4, random_state=42).fit(mapped[["breadth", "elevation"]], mapped["regime_id"].astype(int))
        labels_map = {int(row.regime_id): str(row.interpretation) for row in result["characterization"].itertuples()}
        return clf, labels_map
    except Exception:
        return None, {}


REACHABILITY_COLUMNS = [
    "run_id",
    "dataset_id",
    "task_id",
    "model_id",
    "seed",
    "threshold",
    "minority_reachability",
    "minority_survival_auc",
    "minority_survival_cliffiness",
    "breadth",
    "elevation",
    "regime_id",
]
POSITIVE_SCORE_COLUMNS = [
    "run_id",
    "dataset_id",
    "task_id",
    "model_id",
    "seed",
    "split_id",
    "example_id",
    "positive_label",
    "score",
    "minority_survival_auc",
    "minority_survival_cliffiness",
    "breadth",
    "elevation",
    "regime_id",
]


def reachability_rows(base: dict[str, Any], metrics: dict[str, Any], y_true: np.ndarray, y_score: np.ndarray, thresholds: list[float]) -> list[dict[str, Any]]:
    run_id = f"{base['task_id']}|{base['model_id']}|seed={base['seed']}|split={base['split_id']}"
    rows = []
    for point in empirical_reachability_curve(y_true, y_score, thresholds):
        rows.append(
            {
                "run_id": run_id,
                "dataset_id": base["dataset_name"],
                "task_id": base["task_id"],
                "model_id": base["model_id"],
                "seed": int(base["seed"]),
                "threshold": point["threshold"],
                "minority_reachability": point["minority_reachability"],
                "minority_survival_auc": metrics.get("minority_survival_auc", np.nan),
                "minority_survival_cliffiness": metrics.get("minority_survival_cliffiness", np.nan),
                "breadth": metrics.get("breadth", np.nan),
                "elevation": metrics.get("elevation", np.nan),
                "regime_id": metrics.get("regime_id", np.nan),
            }
        )
    return rows


def write_reachability_rows(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    pd.DataFrame(rows, columns=REACHABILITY_COLUMNS).to_csv(output, mode="a", header=not exists, index=False)


def positive_score_rows(base: dict[str, Any], metrics: dict[str, Any], test_idx: np.ndarray, y_true: np.ndarray, y_score: np.ndarray) -> list[dict[str, Any]]:
    run_id = f"{base['task_id']}|{base['model_id']}|seed={base['seed']}|split={base['split_id']}"
    rows = []
    for original_idx, score in zip(np.asarray(test_idx)[np.asarray(y_true) == 1], np.asarray(y_score)[np.asarray(y_true) == 1], strict=False):
        rows.append(
            {
                "run_id": run_id,
                "dataset_id": base["dataset_name"],
                "task_id": base["task_id"],
                "model_id": base["model_id"],
                "seed": int(base["seed"]),
                "split_id": int(base["split_id"]),
                "example_id": int(original_idx),
                "positive_label": base["positive_label"],
                "score": float(score),
                "minority_survival_auc": metrics.get("minority_survival_auc", np.nan),
                "minority_survival_cliffiness": metrics.get("minority_survival_cliffiness", np.nan),
                "breadth": metrics.get("breadth", np.nan),
                "elevation": metrics.get("elevation", np.nan),
                "regime_id": metrics.get("regime_id", np.nan),
            }
        )
    return rows


def write_positive_score_rows(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    pd.DataFrame(rows, columns=POSITIVE_SCORE_COLUMNS).to_csv(output, mode="a", header=not exists, index=False)


def evaluate_task(data_path: Path, task: pd.Series, model_ids: list[str], seeds: list[int], regime_model, regime_labels, reachability_output: Path | None = None, positive_scores_output: Path | None = None) -> list[dict[str, Any]]:
    df = read_raw_data(data_path)
    X_raw, y_raw = preprocess_frame(df)
    y = (y_raw.astype(str) == str(task.positive_label)).astype(int).to_numpy()
    rows = []
    splitter = StratifiedShuffleSplit(n_splits=len(seeds), test_size=0.5, random_state=0)
    for split_idx, (seed, (train_idx, test_idx)) in enumerate(zip(seeds, splitter.split(X_raw, y), strict=False)):
        X_train_raw = X_raw.iloc[train_idx].copy()
        X_test_raw = X_raw.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]
        pre = make_preprocessor(X_train_raw)
        X_train = pre.fit_transform(X_train_raw)
        X_test = pre.transform(X_test_raw)
        for model_id in model_ids:
            base = {"task_id": task.task_id, "dataset_name": task.dataset_name, "positive_label": task.positive_label, "model_id": model_id, "seed": int(seed), "split_id": int(split_idx), "n_rows": int(len(y)), "n_train": int(len(y_train)), "n_test": int(len(y_test)), "positive_count": int(y.sum()), "positive_fraction": float(y.mean()), "n_features_raw": int(X_raw.shape[1]), "n_features_processed": int(X_train.shape[1]), "fit_failed": False, "failure_reason": ""}
            try:
                model = model_factory(model_id, seed)
                model.fit(X_train, y_train)
                scores = np.clip(predict_scores(model, X_test), 0.0, 1.0)
                pos_scores = scores[y_test == 1]
                occ = compute_occupancy_metrics(y_test, scores, THRESHOLDS)
                shape = positive_score_allocation_shape_metrics(pos_scores)
                breadth = float(shape["positive_histogram_entropy"])
                elevation = float(shape["positive_top_bin_mass"])
                regime_id = np.nan
                regime_label = ""
                method = "unavailable"
                if regime_model is not None:
                    regime_id = int(regime_model.predict(pd.DataFrame({"breadth": [breadth], "elevation": [elevation]}))[0])
                    regime_label = regime_labels.get(regime_id, str(regime_id))
                    method = "projected_real_validation"
                row = {
                    **base,
                    "auroc": float(roc_auc_score(y_test, scores)) if len(np.unique(y_test)) == 2 else np.nan,
                    "average_precision": float(average_precision_score(y_test, scores)),
                    "brier_score": float(brier_score_loss(y_test, scores)),
                    "breadth": breadth,
                    "elevation": elevation,
                    "regime_id": regime_id,
                    "regime_label": regime_label,
                    "regime_assignment_method": method,
                }
                for key in ["minority_survival_auc", "minority_survival_cliffiness", "minority_survival_max_drop", "minority_survival_effective_drop_count"]:
                    row[key] = occ[key]
                row.update(shape)
                rows.append(row)
                if reachability_output is not None:
                    write_reachability_rows(reachability_rows(base, row, y_test, scores, THRESHOLDS), reachability_output)
                if positive_scores_output is not None:
                    write_positive_score_rows(positive_score_rows(base, row, test_idx, y_test, scores), positive_scores_output)
            except OptionalDependencyUnavailable as exc:
                rows.append({**base, "fit_failed": True, "failure_reason": str(exc)})
            except Exception as exc:
                rows.append({**base, "fit_failed": True, "failure_reason": str(exc)})
    return rows


def run_validation(data_root: Path, datasets: str, models: str, seeds: str, output: Path, min_rows: int, min_positives: int, max_positive_fraction: float, save_reachability_curves: Path | None = None, save_positive_scores: Path | None = None) -> pd.DataFrame:
    inventory = discover_datasets(data_root)
    tasks = build_tasks(inventory, min_rows, min_positives, max_positive_fraction)
    if datasets != "auto":
        wanted = {item.strip() for item in datasets.split(",") if item.strip()}
        tasks = tasks[tasks["dataset_name"].isin(wanted)]
    usable_tasks = tasks[tasks["usable"].astype(bool)].copy()
    data_lookup = inventory.set_index("dataset_name")["data_path"].to_dict()
    model_ids = [item.strip() for item in models.split(",") if item.strip()]
    seed_values = [int(item.strip()) for item in seeds.split(",") if item.strip()]
    regime_model, regime_labels = fit_regime_projector()
    if save_reachability_curves is not None and save_reachability_curves.exists():
        save_reachability_curves.unlink()
    if save_positive_scores is not None and save_positive_scores.exists():
        save_positive_scores.unlink()
    rows = []
    for task in usable_tasks.itertuples(index=False):
        rows.extend(evaluate_task(Path(data_lookup[task.dataset_name]), pd.Series(task._asdict()), model_ids, seed_values, regime_model, regime_labels, save_reachability_curves, save_positive_scores))
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", default="auto")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--min-positives", type=int, default=30)
    parser.add_argument("--max-positive-fraction", type=float, default=0.35)
    parser.add_argument("--save-reachability-curves", type=Path, default=None)
    parser.add_argument("--save-positive-scores", type=Path, default=None)
    args = parser.parse_args()
    result = run_validation(args.data_root, args.datasets, args.models, args.seeds, args.output, args.min_rows, args.min_positives, args.max_positive_fraction, args.save_reachability_curves, args.save_positive_scores)
    print(f"wrote {args.output} rows={len(result)}")
    print(f"wrote {DEFAULT_INVENTORY}")
    print(f"wrote {DEFAULT_TASKS}")
    if args.save_reachability_curves is not None:
        print(f"wrote {args.save_reachability_curves}")
    if args.save_positive_scores is not None:
        print(f"wrote {args.save_positive_scores}")


if __name__ == "__main__":
    main()
