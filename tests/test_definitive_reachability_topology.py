import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_report_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_definitive_reachability_topology.py"
    spec = importlib.util.spec_from_file_location("report_definitive_reachability_topology", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_hddt_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_hddt_dataset_accessibility_validation.py"
    spec = importlib.util.spec_from_file_location("run_hddt_dataset_accessibility_validation", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _curve_rows(n_runs=24):
    thresholds = np.linspace(0, 1, 11)
    rows = []
    for i in range(n_runs):
        dataset = f"d{i % 4}"
        model = "cart" if i % 2 == 0 else "random_forest"
        survival = 0.2 + 0.03 * i
        cliff = 0.8 if model == "cart" else 0.2
        breadth = 0.1 + 0.02 * i
        elevation = 1 - breadth / 2
        curve = np.clip(1 - thresholds * (0.5 + cliff / 2), 0, 1)
        for t, r in zip(thresholds, curve):
            rows.append(
                {
                    "run_id": f"run{i}",
                    "dataset_id": dataset,
                    "task_id": f"{dataset}:1",
                    "model_id": model,
                    "seed": i,
                    "threshold": t,
                    "minority_reachability": r,
                    "minority_survival_auc": survival,
                    "minority_survival_cliffiness": cliff,
                    "breadth": breadth,
                    "elevation": elevation,
                    "regime_id": i % 3,
                }
            )
    return pd.DataFrame(rows)


def test_empirical_reachability_curve_monotone():
    root = Path(__file__).resolve().parents[1]
    script = root / "src" / "hib" / "occupancy.py"
    spec = importlib.util.spec_from_file_location("local_occupancy", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    y = np.array([1, 1, 1, 0])
    scores = np.array([0.2, 0.5, 0.9, 0.1])
    rows = module.empirical_reachability_curve(y, scores, np.linspace(0, 1, 11))
    values = np.array([row["minority_reachability"] for row in rows])

    assert len(rows) == 11
    assert np.all(np.diff(values) <= 1e-12)


def test_curve_writer_one_row_per_threshold(tmp_path):
    module = _load_hddt_module()
    base = {"task_id": "d:1", "dataset_name": "d", "model_id": "cart", "seed": 0, "split_id": 0}
    metrics = {"minority_survival_auc": 0.5, "minority_survival_cliffiness": 0.8, "breadth": 0.2, "elevation": 0.9, "regime_id": 1}
    rows = module.reachability_rows(base, metrics, np.array([1, 1, 0]), np.array([0.2, 0.8, 0.1]), [0.0, 0.5, 1.0])
    output = tmp_path / "curves.csv"

    module.write_reachability_rows(rows, output)
    written = pd.read_csv(output)

    assert len(written) == 3
    assert set(["run_id", "threshold", "minority_reachability"]).issubset(written.columns)


def test_report_handles_missing_optional_regime(tmp_path):
    module = _load_report_module()
    curves = _curve_rows().drop(columns=["regime_id"])
    path = tmp_path / "curves.csv"
    curves.to_csv(path, index=False)

    outputs = module.write_report([path], tmp_path / "reports")
    summary = pd.read_json(tmp_path / "reports" / "definitive_reachability_summary.json", typ="series")

    assert any(str(output).endswith("definitive_reachability_topology.md") for output in outputs)
    assert "survival_r2" in summary.index
    assert pd.isna(summary["regime_accuracy"])


def test_summary_json_contains_required_metrics(tmp_path):
    module = _load_report_module()
    path = tmp_path / "curves.csv"
    _curve_rows().to_csv(path, index=False)

    module.write_report([path], tmp_path / "reports")
    summary = pd.read_json(tmp_path / "reports" / "definitive_reachability_summary.json", typ="series")

    for key in ["n_runs", "topology_clusters", "survival_r2", "cliffiness_r2", "breadth_r2", "elevation_r2", "regime_accuracy", "model_family_accuracy", "outcome"]:
        assert key in summary.index
