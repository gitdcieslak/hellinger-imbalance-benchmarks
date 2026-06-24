import importlib.util
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_regime_transition_analysis.py"
    spec = importlib.util.spec_from_file_location("report_regime_transition_analysis", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_regime_transition_analysis module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assigned_rows():
    rows = []
    for trajectory_id, values in {
        "a": [(0, 0.1, 0.05, 0, 0.2, 0.1), (1, 1.1, 0.02, 1, 0.5, 0.2), (2, 1.3, 0.2, 1, 0.4, 0.4)],
        "b": [(0, 0.2, 0.04, 0, 0.1, 0.1), (1, 0.25, 0.05, 0, 0.1, 0.2)],
    }.items():
        for step, breadth, elevation, regime, cliff, survival in values:
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "step_index": step,
                    "step_value": float(step),
                    "family": "test_family",
                    "intervention": "density increase",
                    "model_id": "m",
                    "seed": 0,
                    "skew_ratio": 100,
                    "breadth": breadth,
                    "elevation": elevation,
                    "regime_id": regime,
                    "regime_label": str(regime),
                    "minority_survival_cliffiness": cliff,
                    "minority_survival_auc": survival,
                }
            )
    return pd.DataFrame(rows)


def test_build_transition_paths_and_edges():
    module = _load_module()
    assigned = _assigned_rows()

    steps = module.build_step_transitions(assigned)
    paths = module.build_path_summary(assigned, steps)
    edges = module.transition_edges(steps)

    assert len(steps) == 3
    assert paths.loc[paths["trajectory_id"].eq("a"), "number_of_regime_changes"].iloc[0] == 1
    assert edges.iloc[0]["from_regime"] == 0
    assert edges.iloc[0]["to_regime"] == 1


def test_stability_and_events():
    module = _load_module()
    steps = module.build_step_transitions(_assigned_rows())

    stability = module.stability_summary(steps)
    events, event_summary = module.transition_events(steps)

    assert {"p_same_regime", "p_regime_change"}.issubset(stability.columns)
    assert not events.empty
    assert "transition_event_count" in event_summary.columns


def test_assign_regimes_marks_native_matches():
    module = _load_module()
    pooled = pd.DataFrame(
        {
            "source_file": ["classical_allocator_morphology", "other"],
            "model_id": ["cart", "x"],
            "seed": [0, 0],
            "skew_ratio": [25, 25],
            "breadth": [0.2, 1.0],
            "elevation": [0.05, 0.2],
        }
    )
    train_x = pd.DataFrame({"breadth": [0.2, 1.0], "elevation": [0.05, 0.2]})
    train_y = [2, 0]
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(train_x, train_y)
    native = pd.DataFrame({"allocation_family": ["cart"], "seed": [0], "skew_ratio": [25], "breadth": [0.2], "elevation": [0.05], "regime_id": [2]})

    assigned = module.assign_regimes(pooled, model, {0: "broad", 2: "floor"}, native)

    assert assigned.loc[0, "assigned_by"] == "native_cluster"
    assert assigned.loc[1, "assigned_by"] == "classifier_projection"
