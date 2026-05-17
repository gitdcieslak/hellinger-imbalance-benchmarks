import numpy as np
import pandas as pd
import pytest

from hib.arrays import ensure_numpy_array, ensure_numpy_vector


def test_dataframe_input_normalization():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    arr = ensure_numpy_array(frame)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)


def test_ndarray_passthrough():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = ensure_numpy_array(arr)
    assert out is arr


def test_vector_normalization():
    y = ensure_numpy_vector([[0], [1], [0]])
    assert y.shape == (3,)


def test_object_dtype_rejected():
    with pytest.raises(TypeError, match="object-dtype"):
        ensure_numpy_array(np.array([["a"], ["b"]], dtype=object))
