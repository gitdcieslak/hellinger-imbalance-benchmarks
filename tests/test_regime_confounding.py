import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_regime_confounding.py"
    spec = importlib.util.spec_from_file_location("report_regime_confounding", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_regime_confounding module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows():
    rows = []
    specs = [
        (0, "cart", 25, 0.25, 0.05, 0.0),
        (1, "cart", 100, 0.65, 0.25, 0.08),
        (2, "rf", 25, 1.10, 0.10, 0.45),
        (3, "rf", 100, 1.55, 0.30, 0.55),
    ]
    for cluster, family, skew, breadth, elevation, cliffiness in specs:
        for i in range(12):
            rows.append(
                {
                    "hdbscan_cluster": cluster,
                    "allocation_family": family,
                    "skew_ratio": skew,
                    "breadth": breadth + 0.01 * (i % 3),
                    "elevation": elevation + 0.01 * (i // 3),
                    "minority_survival_cliffiness": cliffiness + 0.01 * (i % 4),
                    "minority_survival_auc": elevation + 0.1,
                    "vertical_slack": 0.3 - elevation,
                    "normalized_position_between_envelopes": min(1.0, elevation + 0.2),
                }
            )
    return rows


def test_evaluate_confounding_outputs_expected_rows():
    module = _load_module()
    df = pd.DataFrame(_rows())
    assignment = pd.DataFrame({"cluster_id": [0, 1, 2, 3], "regime_id": [0, 0, 1, 1]})
    mapped = df.merge(assignment, left_on="hdbscan_cluster", right_on="cluster_id", how="left")

    scores, deltas = module.evaluate_confounding(mapped)

    assert len(scores) == 5
    assert "regime_accuracy_gain_over_family_skew" in deltas
    assert scores["cliffiness_cv_r2"].notna().all()


def test_report_generation(tmp_path):
    module = _load_module()
    atlas = tmp_path / "morphology_atlas_clusters.csv"
    pd.DataFrame(_rows()).to_csv(atlas, index=False)

    outputs = module.write_report(tmp_path, atlas_csv=atlas)

    names = {path.name for path in outputs}
    assert "regime_confounding_report.md" in names
    assert "regime_confounding_scores.json" in names
    for path in outputs:
        assert path.exists()
