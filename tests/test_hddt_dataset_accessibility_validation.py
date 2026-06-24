import importlib.util
from pathlib import Path

import pandas as pd


def _load_runner():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_hddt_dataset_accessibility_validation.py"
    spec = importlib.util.spec_from_file_location("run_hddt_dataset_accessibility_validation", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_reporter():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "report_hddt_dataset_accessibility_validation.py"
    spec = importlib.util.spec_from_file_location("report_hddt_dataset_accessibility_validation", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_audit():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "audit_hddt_dataset_coverage.py"
    spec = importlib.util.spec_from_file_location("audit_hddt_dataset_coverage", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_discovery_skips_macos_metadata(tmp_path):
    runner = _load_runner()
    data_root = tmp_path / "hddt-data"
    data_root.mkdir()
    (data_root / "toy.data").write_text("1,a,no\n2,b,no\n3,a,yes\n4,b,no\n", encoding="utf-8")
    (data_root / "toy.names").write_text("toy names", encoding="utf-8")
    (data_root / "._toy.data").write_text("metadata", encoding="utf-8")

    inventory = runner.discover_datasets(data_root, tmp_path / "inventory.csv")

    assert list(inventory["dataset_name"]) == ["toy"]
    assert inventory.iloc[0]["usable"] is True or bool(inventory.iloc[0]["usable"])


def test_task_construction_binary_minority(tmp_path):
    runner = _load_runner()
    data_root = tmp_path / "hddt-data"
    data_root.mkdir()
    rows = [f"{i},x,no" for i in range(40)] + [f"{i},y,yes" for i in range(10)]
    (data_root / "toy.data").write_text("\n".join(rows), encoding="utf-8")
    inventory = runner.discover_datasets(data_root, tmp_path / "inventory.csv")

    tasks = runner.build_tasks(inventory, min_rows=20, min_positives=5, max_positive_fraction=0.35, output_csv=tmp_path / "tasks.csv")

    usable = tasks[tasks["usable"].astype(bool)]
    assert len(usable) == 1
    assert usable.iloc[0]["positive_label"] == "yes"


def test_report_generation(tmp_path):
    reporter = _load_reporter()
    result = tmp_path / "validation.csv"
    pd.DataFrame(
        [
            {
                "dataset_name": "toy",
                "task_id": "toy:yes",
                "positive_label": "yes",
                "model_id": "cart",
                "seed": 0,
                "fit_failed": False,
                "positive_fraction": 0.2,
                "auroc": 0.8,
                "average_precision": 0.4,
                "brier_score": 0.1,
                "minority_survival_auc": 0.5,
                "minority_survival_cliffiness": 0.2,
                "breadth": 0.6,
                "elevation": 0.1,
                "regime_id": 1,
            },
            {
                "dataset_name": "toy",
                "task_id": "toy:yes",
                "positive_label": "yes",
                "model_id": "rf",
                "seed": 1,
                "fit_failed": False,
                "positive_fraction": 0.2,
                "auroc": 0.9,
                "average_precision": 0.5,
                "brier_score": 0.08,
                "minority_survival_auc": 0.7,
                "minority_survival_cliffiness": 0.4,
                "breadth": 1.2,
                "elevation": 0.2,
                "regime_id": 3,
            },
        ]
    ).to_csv(result, index=False)

    outputs = reporter.write_report(result, tmp_path / "summary.md")

    names = {path.name for path in outputs}
    assert "summary.md" in names
    assert "hddt_morphology_space.png" in names


def test_audit_skip_reason_classification():
    audit = _load_audit()

    assert audit.class_skip_reason(99, 50, 0.2, 100, 20, 0.5) == "too_few_rows"
    assert audit.class_skip_reason(200, 19, 0.1, 100, 20, 0.5) == "too_few_positives"
    assert audit.class_skip_reason(200, 120, 0.6, 100, 20, 0.5) == "positive_fraction_too_high"
    assert audit.class_skip_reason(200, 50, 0.25, 100, 20, 0.5) == "included"


def test_coverage_audit_outputs_ovr_rows(tmp_path):
    audit = _load_audit()
    data_root = tmp_path / "hddt-data"
    data_root.mkdir()
    rows = [f"{i},a" for i in range(30)] + [f"{i},b" for i in range(25)] + [f"{i},c" for i in range(10)]
    (data_root / "multi.data").write_text("\n".join(rows), encoding="utf-8")

    coverage, ovr, _ = audit.coverage_audit(data_root)

    assert "multi" in set(coverage["dataset_name"])
    assert set(ovr["class_label"]) == {"a", "b", "c"}
    assert "included_under_relaxed_rules" in ovr.columns
