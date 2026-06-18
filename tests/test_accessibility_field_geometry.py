import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "report_accessibility_ridges.py"
    spec = importlib.util.spec_from_file_location("report_accessibility_ridges", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load report_accessibility_ridges module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flat_field_has_zero_gradients_and_zero_edges():
    module = _load_module()
    z = np.ones((7, 7))

    geom = module.compute_surface_geometry(z)

    assert np.allclose(geom["gradient_magnitude"], 0.0)
    assert int(geom["edge_pixel_count"]) == 0
    assert int(geom["ridge_pixel_count"]) == 0


def test_linear_ramp_has_constant_gradient_and_no_ridges():
    module = _load_module()
    x = np.linspace(0.0, 3.0, 7)
    y = np.linspace(0.0, 2.0, 5)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    z = 2.0 * xx + 0.5 * yy

    geom = module.compute_surface_geometry(z, x, y)
    interior = geom["gradient_magnitude"][1:-1, 1:-1]

    assert np.allclose(interior, np.sqrt(2.0**2 + 0.5**2))
    assert int(geom["ridge_pixel_count"]) == 0


def test_step_function_has_clear_edge():
    module = _load_module()
    z = np.zeros((9, 9))
    z[:, 5:] = 1.0

    geom = module.compute_surface_geometry(z)

    assert int(geom["edge_pixel_count"]) > 0
    assert np.max(geom["sobel_strength"][:, 4:6]) > 0.0


def test_gaussian_bump_has_curvature_signal():
    module = _load_module()
    x = np.linspace(-2.0, 2.0, 21)
    y = np.linspace(-2.0, 2.0, 21)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    z = np.exp(-(xx**2 + yy**2))

    geom = module.compute_surface_geometry(z, x, y)

    lap = geom["laplacian"]

    assert np.nanmin(lap) < 0.0
    assert np.nanmax(lap) > 0.0
    assert np.max(geom["max_abs_hessian_eigenvalue"]) > 0.0
