"""Check whether morphology regimes add signal beyond allocator family and skew."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from report_morphology_regime_consolidation import (  # noqa: E402
    CLUSTER_COL,
    DEFAULT_ATLAS,
    DEFAULT_OUTPUT_DIR,
    TARGET,
    _markdown_table,
    choose_k,
    cluster_centroids,
    evaluate_k_range,
    load_atlas,
)


def bucket(series: pd.Series, n: int = 3) -> pd.Series:
    return pd.qcut(series.rank(method="first"), q=n, labels=False).astype(int)


def _feature_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    family_skew = pd.get_dummies(df[["allocation_family", "skew_ratio"]].astype(str), prefix=["family", "skew"])
    regime = pd.get_dummies(df[["regime_id"]].astype(str), prefix=["regime"])
    coordinates = df[["breadth", "elevation"]].reset_index(drop=True)
    return {
        "Family + skew": family_skew,
        "Regime ID": regime,
        "Family + skew + regime": pd.concat([family_skew, regime], axis=1),
        "Family + skew + coordinates": pd.concat([family_skew, coordinates], axis=1),
        "Family + skew + regime + coordinates": pd.concat([family_skew, regime, coordinates], axis=1),
    }


def _classification_accuracy(X: pd.DataFrame, y: pd.Series) -> float:
    min_count = int(y.value_counts().min())
    cv = StratifiedKFold(n_splits=max(2, min(5, min_count)), shuffle=True, random_state=7)
    pred = cross_val_predict(LogisticRegression(max_iter=3000), X, y, cv=cv)
    return float(accuracy_score(y, pred))


def _regression_r2(X: pd.DataFrame, y: pd.Series) -> float:
    cv = KFold(n_splits=5, shuffle=True, random_state=7)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pred = cross_val_predict(model, X, y, cv=cv)
    return float(r2_score(y, pred))


def assign_regimes(atlas_csv: Path = DEFAULT_ATLAS) -> tuple[pd.DataFrame, int]:
    df = load_atlas(atlas_csv)
    centroids = cluster_centroids(df)
    evaluation, assignments = evaluate_k_range(df, centroids)
    k = choose_k(evaluation, compression=_compression_for_choice(df, assignments))
    assignment = assignments[k]
    mapped = df.merge(assignment[["cluster_id", "regime_id"]], left_on=CLUSTER_COL, right_on="cluster_id", how="left")
    return mapped, k


def _compression_for_choice(df: pd.DataFrame, assignments: dict[int, pd.DataFrame]) -> pd.DataFrame:
    from report_morphology_regime_consolidation import compression_curve

    return compression_curve(df, assignments)


def evaluate_confounding(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    feature_sets = _feature_sets(df)
    y_bucket = bucket(df[TARGET])
    y = df[TARGET]
    rows = []
    for name, X in feature_sets.items():
        rows.append(
            {
                "feature_set": name,
                "cliffiness_bucket_cv_accuracy": _classification_accuracy(X, y_bucket),
                "cliffiness_cv_r2": _regression_r2(X, y),
            }
        )
    scores = pd.DataFrame(rows)
    lookup = scores.set_index("feature_set")
    deltas = {
        "regime_accuracy_gain_over_family_skew": float(
            lookup.loc["Family + skew + regime", "cliffiness_bucket_cv_accuracy"] - lookup.loc["Family + skew", "cliffiness_bucket_cv_accuracy"]
        ),
        "regime_r2_gain_over_family_skew": float(lookup.loc["Family + skew + regime", "cliffiness_cv_r2"] - lookup.loc["Family + skew", "cliffiness_cv_r2"]),
        "regime_accuracy_gain_over_family_skew_coordinates": float(
            lookup.loc["Family + skew + regime + coordinates", "cliffiness_bucket_cv_accuracy"]
            - lookup.loc["Family + skew + coordinates", "cliffiness_bucket_cv_accuracy"]
        ),
        "regime_r2_gain_over_family_skew_coordinates": float(
            lookup.loc["Family + skew + regime + coordinates", "cliffiness_cv_r2"] - lookup.loc["Family + skew + coordinates", "cliffiness_cv_r2"]
        ),
    }
    return scores, deltas


def conclusion(deltas: dict[str, float]) -> str:
    if deltas["regime_accuracy_gain_over_family_skew"] > 0.02 and deltas["regime_r2_gain_over_family_skew"] > 0.02:
        return "Regime ID adds signal beyond allocator family and skew."
    if deltas["regime_accuracy_gain_over_family_skew"] > 0.0 or deltas["regime_r2_gain_over_family_skew"] > 0.0:
        return "Regime ID adds weak or mixed signal beyond allocator family and skew."
    return "Regime ID does not add held-out signal beyond allocator family and skew in this check."


def write_report(output_dir: Path, atlas_csv: Path = DEFAULT_ATLAS) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df, k = assign_regimes(atlas_csv)
    scores, deltas = evaluate_confounding(df)
    report = output_dir / "regime_confounding_report.md"
    text = "\n".join(
        [
            "# Regime Confounding Check",
            "",
            f"Macro-regime count: {k}",
            "",
            "## Cross-Validated Scores",
            _markdown_table(scores),
            "",
            "## Incremental Regime Signal",
            _markdown_table(pd.DataFrame([deltas])),
            "",
            "## Conclusion",
            conclusion(deltas),
        ]
    ) + "\n"
    report.write_text(text, encoding="utf-8")
    scores_json = output_dir / "regime_confounding_scores.json"
    scores_json.write_text(json.dumps({"scores": scores.to_dict(orient="records"), "deltas": deltas}, indent=2), encoding="utf-8")
    return report, scores_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-csv", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for path in write_report(args.output_dir, atlas_csv=args.atlas_csv):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
