import cupy as cp
import numpy as np
import pyarrow.cuda as pa


def test_numpy_arrow_roundtrip():
    num_array = np.asarray([1, 2, 3, 4])
    pa_arr = pa.array(num_array)
    assert len(pa_arr) == 4
    assert all(pa_arr.to_numpy() == num_array)


def test_cupy_arrow_roundtrip():
    num_array = cp.asarray([1, 2, 3, 4])
    # to numba
    pa_arr = pa.CudaBuffer.from_buffer(num_array)
    assert len(pa_arr) == 4
    assert all(pa_arr.to_numpy() == num_array)
