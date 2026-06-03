import importlib.util
from pathlib import Path

import pandas as pd


def _load_report_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_paper2_pilot.py"
    spec = importlib.util.spec_from_file_location("report_paper2_pilot", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_paper2_pilot module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper2_pilot_report_contains_required_sections(tmp_path):
    module = _load_report_module()
    csv_path = tmp_path / "topology.csv"
    topology_md = tmp_path / "topology.md"
    mediation_md = tmp_path / "mediation.md"

    pd.DataFrame(
        [
            {
                "dataset_id": "synthetic-gaussian-skew-100-to-1",
                "objective": "bce",
                "seed": 0,
                "model_name": "mlp_bce",
                "fit_failed": False,
                "topology_mode": "knn",
                "n_positive": 50,
                "minority_survival_auc": 0.4,
                "persistence": 0.5,
                "minority_survival_cliffiness": 0.6,
                "positive_histogram_entropy": 0.9,
                "positive_effective_score_bins": 2.5,
                "positive_max_bin_mass": 0.6,
                "positive_top_bin_mass": 0.1,
            }
        ]
    ).to_csv(csv_path, index=False)
    topology_md.write_text("## Interpretation\n- allocation is stronger.\n", encoding="utf-8")
    mediation_md.write_text("## Interpretation\n- allocation mediates survival.\n", encoding="utf-8")

    report = module.build_paper2_pilot_report(csv_path, topology_md, mediation_md)

    assert "## Research Question" in report
    assert "## Experimental Setup" in report
    assert "## Key Results" in report
    assert "## What We Believe Now" in report
    assert "### Dropout As Feature-Subspace Sampling" in report
    assert "## What Remains Unresolved" in report
    assert "## Threats To Validity" in report
    assert "## Next Experiments" in report
    assert "Allocation is the stronger first-order explanation" in report
