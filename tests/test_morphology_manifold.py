import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_morphology_manifold.py"
    spec = importlib.util.spec_from_file_location("report_morphology_manifold", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_morphology_manifold module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(offset: float = 0.0, n: int = 8):
    rows = []
    for seed in range(n):
        breadth = 0.15 + 0.08 * seed + offset
        elevation = 0.95 - 0.07 * seed + 0.5 * offset
        rows.append(
            {
                "model_id": "mlp_weighted_bce" if seed % 2 == 0 else "mlp_weighted_bce_dropout_0_1",
                "seed": seed,
                "breadth": breadth,
                "elevation": elevation,
                "minority_survival_auc": 0.25 + 0.55 * elevation - 0.08 * breadth,
                "minority_survival_cliffiness": 0.1 + (breadth - 0.45) ** 2 + 0.25 * (1.0 - elevation),
                "auroc": 0.7 + 0.01 * seed,
                "average_precision": 0.2 + 0.01 * seed,
            }
        )
    return rows


def test_prepare_dataset_normalizes_inputs_and_handles_missing(tmp_path):
    module = _load_module()
    used = tmp_path / "weighted_dropout_sweep_dense.csv"
    df = pd.DataFrame(_rows()).rename(columns={"breadth": "positive_histogram_entropy", "elevation": "positive_top_bin_mass"})
    df = df.drop(columns=["model_id"]).assign(dropout_rate=0.1)
    df.to_csv(used, index=False)
    missing = tmp_path / "missing.csv"

    prepared, messages, diagnostics = module.prepare_dataset([used, missing])

    assert not prepared.empty
    assert any("used:" in message for message in messages)
    assert any("missing:" in message for message in messages)
    assert {"breadth", "elevation", "pc1", "pc2", "family_arc_position"}.issubset(prepared.columns)
    assert "pc1_explained_variance_ratio" in set(diagnostics["diagnostic"])


def test_pca_coordinate_construction():
    module = _load_module()
    df = pd.DataFrame(_rows()).assign(experiment_family="a", model="m")

    out, diagnostics = module.add_pca_coordinates(df)

    assert {"pc1", "pc2", "global_pc1_position"}.issubset(out.columns)
    assert np.isclose(out["pc1"].mean(), 0.0, atol=1e-10)
    assert out["global_pc1_position"].between(0.0, 1.0).all()
    assert diagnostics.loc[diagnostics["diagnostic"] == "pc1_explained_variance_ratio", "value"].iloc[0] > 0.9


def test_arc_length_and_position_normalization():
    module = _load_module()
    df = pd.DataFrame(
        {
            "experiment_family": ["a", "a", "a", "b", "b"],
            "model": ["m"] * 5,
            "pc1": [0.0, 1.0, 2.0, 0.0, 2.0],
            "pc2": [0.0, 0.0, 0.0, 1.0, 1.0],
            "breadth": [0.0, 1.0, 2.0, 0.0, 2.0],
            "elevation": [1.0, 0.5, 0.0, 1.0, 0.0],
            "minority_survival_auc": [0.1, 0.2, 0.3, 0.2, 0.4],
            "minority_survival_cliffiness": [0.3, 0.2, 0.1, 0.4, 0.2],
        }
    )

    out = module.add_arc_coordinates(df)

    assert out.groupby("experiment_family")["family_arc_position"].min().eq(0.0).all()
    assert out.groupby("experiment_family")["family_arc_position"].max().eq(1.0).all()
    assert out["global_arc_position"].between(0.0, 1.0).all()
    assert {"accessibility_coordinate_pc1", "accessibility_coordinate_arc", "accessibility_coordinate_kernel"}.issubset(out.columns)


def test_report_generation_creates_required_sections_and_plots(tmp_path):
    module = _load_module()
    paths = []
    for filename, offset in [
        ("density_threshold_sweep.csv", 0.00),
        ("fragmentation_sweep.csv", 0.05),
        ("allocation_trajectory.csv", 0.10),
    ]:
        path = tmp_path / filename
        df = pd.DataFrame(_rows(offset))
        if filename == "allocation_trajectory.csv":
            df = df.rename(columns={"model_id": "objective"})
        df.to_csv(path, index=False)
        paths.append(path)
    output_md = tmp_path / "morphology_manifold_summary.md"

    outputs = module.write_report(paths, output_md)
    report = output_md.read_text(encoding="utf-8")

    assert "# Morphology Manifold Experiments" in report
    assert "## PCA / Manifold Diagnostics" in report
    assert "## Experiment 1: Manifold Coordinates" in report
    assert "## Experiment 2: Arc Length" in report
    assert "## Experiment 3: Universal Accessibility Coordinate" in report
    assert "Is morphology space approximately one-dimensional?" in report
    assert "Is there evidence for a universal scalar accessibility coordinate?" in report
    assert len(outputs) == 6
    for path in outputs:
        assert path.exists()
