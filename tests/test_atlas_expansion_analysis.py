import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_atlas_expansion_analysis.py"
    spec = importlib.util.spec_from_file_location("report_atlas_expansion_analysis", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_region_label_categories():
    module = _load_module()

    assert module.region_label(pd.Series({"breadth": 0.2, "elevation": 0.9})) == "high elevation / low breadth"
    assert module.region_label(pd.Series({"breadth": 0.9, "elevation": 0.9})) == "high elevation / moderate breadth"
    assert module.region_label(pd.Series({"breadth": 1.2, "elevation": 0.1})) == "low elevation / high breadth"


def test_attach_support_distance_marks_axis_support():
    module = _load_module()
    atlas = pd.DataFrame({"breadth": [0.0, 1.0], "elevation": [0.0, 1.0]})
    hddt = pd.DataFrame({"breadth": [0.5, 2.0], "elevation": [0.5, 0.5]})

    out = module.attach_support_distance(atlas, hddt)

    assert bool(out.loc[0, "inside_axis_support"])
    assert not bool(out.loc[1, "inside_axis_support"])
    assert "nearest_support_distance" in out.columns


def test_grouped_cv_scores_shape():
    module = _load_module()
    df = pd.DataFrame(
        {
            "dataset_name": ["a"] * 5 + ["b"] * 5,
            "breadth": [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4],
            "elevation": [0.9, 0.8, 0.75, 0.7, 0.65, 0.2, 0.25, 0.3, 0.35, 0.4],
            "minority_survival_auc": [0.9, 0.8, 0.75, 0.7, 0.65, 0.3, 0.35, 0.4, 0.45, 0.5],
            "minority_survival_cliffiness": [0.1, 0.2, 0.25, 0.3, 0.35, 0.8, 0.7, 0.65, 0.6, 0.55],
            "regime": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )

    scores = module.grouped_cv_scores(df, "regime")

    assert set(scores["target"]) == {"minority_survival_auc", "minority_survival_cliffiness"}
    assert len(scores) == 4
