import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_minority_reachability_cohorts.py"
    spec = importlib.util.spec_from_file_location("report_minority_reachability_cohorts", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _scores():
    rows = []
    for seed in [0, 1]:
        for model in ["cart", "rf"]:
            for example_id, score in [(1, 0.2), (2, 0.2), (3, 0.8), (4, 0.9)]:
                rows.append(
                    {
                        "run_id": f"d:1|{model}|seed={seed}|split={seed}",
                        "dataset_id": "d",
                        "task_id": "d:1",
                        "model_id": model,
                        "seed": seed,
                        "split_id": seed,
                        "example_id": example_id,
                        "positive_label": 1,
                        "score": score,
                        "minority_survival_auc": 0.5,
                        "minority_survival_cliffiness": 0.5,
                        "breadth": 0.3,
                        "elevation": 0.8,
                        "regime_id": 1,
                    }
                )
    return pd.DataFrame(rows)


def test_largest_drop_transitions_identifies_cohort():
    module = _load_module()
    scores = module.load_positive_scores_from_frame(_scores()) if hasattr(module, "load_positive_scores_from_frame") else _scores().assign(positive_key=lambda d: d["dataset_id"] + "|" + d["task_id"] + "|" + d["example_id"].astype(str), score_round=lambda d: d["score"].round(6))

    transitions, members = module.largest_drop_transitions(scores)

    assert len(transitions) == 4
    assert set(transitions["n_positives_removed"]) == {2}
    assert set(members["example_id"]) == {1, 2}


def test_largest_drop_transitions_ignores_terminal_one_scores():
    module = _load_module()
    scores = pd.DataFrame(
        [
            {
                "run_id": "d:1|cart|seed=0|split=0",
                "dataset_id": "d",
                "task_id": "d:1",
                "model_id": "cart",
                "seed": 0,
                "split_id": 0,
                "example_id": example_id,
                "positive_label": 1,
                "score": score,
                "minority_survival_auc": 0.5,
                "minority_survival_cliffiness": 0.5,
                "breadth": 0.3,
                "elevation": 0.8,
                "regime_id": 1,
                "positive_key": f"d|d:1|{example_id}",
                "score_round": round(score, 6),
            }
            for example_id, score in [(1, 1.0), (2, 1.0), (3, 1.0), (4, 0.7), (5, 0.7)]
        ]
    )

    transitions, members = module.largest_drop_transitions(scores)

    assert transitions.loc[0, "largest_drop_threshold"] == 0.7
    assert transitions.loc[0, "n_positives_removed"] == 2
    assert set(members["example_id"]) == {4, 5}


def test_overlap_and_persistence_nonempty():
    module = _load_module()
    scores = _scores().assign(positive_key=lambda d: d["dataset_id"] + "|" + d["task_id"] + "|" + d["example_id"].astype(str), score_round=lambda d: d["score"].round(6))
    _, members = module.largest_drop_transitions(scores)

    overlap, _ = module.overlap_analysis(members)
    persistence = module.cohort_persistence(scores, members)

    assert not overlap.empty
    assert persistence["cohort_persistence"].max() == 1.0


def test_positive_reachability_matrix_written(tmp_path):
    module = _load_module()
    scores = _scores().assign(positive_key=lambda d: d["dataset_id"] + "|" + d["task_id"] + "|" + d["example_id"].astype(str), score_round=lambda d: d["score"].round(6))
    output = tmp_path / "matrix.csv"

    matrix = module.positive_reachability_matrix(scores, output, max_runs=1)

    assert output.exists()
    assert {"run_id", "positive_key", "threshold", "reachable"}.issubset(matrix.columns)
    assert len(matrix) == 4 * 101


def test_profile_table_has_driver_groups():
    module = _load_module()
    scores = _scores().assign(positive_key=lambda d: d["dataset_id"] + "|" + d["task_id"] + "|" + d["example_id"].astype(str), score_round=lambda d: d["score"].round(6))
    _, members = module.largest_drop_transitions(scores)

    profiles = module.profile_table(scores, members)

    assert set(profiles["is_cliff_driver"]) == {False, True}
