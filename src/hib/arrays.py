"""Array normalization helpers for model execution paths."""

from __future__ import annotations

from typing import Any

import numpy as np


def ensure_numpy_array(X: Any) -> np.ndarray:
    """Normalize feature containers to dense numpy arrays."""

    if hasattr(X, "toarray"):
        raise TypeError("sparse feature matrices are not supported in benchmark runners")
    if isinstance(X, np.ndarray):
        out = X
    else:
        out = np.asarray(X)
    if out.ndim != 2:
        raise ValueError(f"expected 2D feature array, got shape {out.shape}")
    if out.dtype == object:
        raise TypeError("object-dtype feature arrays are not supported")
    return out


def ensure_numpy_vector(y: Any) -> np.ndarray:
    """Normalize label containers to 1D numpy vectors."""

    if hasattr(y, "toarray"):
        raise TypeError("sparse label vectors are not supported in benchmark runners")
    if isinstance(y, np.ndarray):
        out = y
    else:
        out = np.asarray(y)
    out = np.ravel(out)
    if out.dtype == object:
        raise TypeError("object-dtype label vectors are not supported")
    return out
