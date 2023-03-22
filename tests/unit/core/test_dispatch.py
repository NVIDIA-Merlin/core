#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import pandas as pd
import pytest

from merlin.core.compat import HAS_GPU
from merlin.core.compat import cupy as cp
from merlin.core.dispatch import concat_columns, is_list_dtype, list_val_dtype, make_df

if HAS_GPU:
    _DEVICES = ["cpu", "gpu"]
else:
    _DEVICES = ["cpu"]


@pytest.mark.parametrize("device", _DEVICES)
def test_list_dtypes(tmpdir, device):
    df = make_df(device=device)
    df["vals"] = [
        [[0, 1, 2], [3, 4], [5]],
    ]
    # Check that the index can be arbitrary
    df.set_index(np.array([2]), drop=True, inplace=True)

    assert is_list_dtype(df["vals"])
    assert list_val_dtype(df["vals"]) == np.dtype(np.int64)


@pytest.mark.parametrize("device", _DEVICES)
def test_concat_columns(device):
    df1 = make_df({"a": [1, 2], "b": [[3], [4, 5]]}, device=device)
    df2 = make_df({"c": [3, 4, 5]}, device=device)
    data_frames = [df1, df2]
    res = concat_columns(data_frames)
    assert res.columns.to_list() == ["a", "b", "c"]


@pytest.mark.skipif(not (cp and HAS_GPU), reason="Cupy not available")
def test_pandas_cupy_combo():
    rand_cp_nd_arr = cp.random.uniform(0.0, 1.0, size=100)
    with pytest.raises(TypeError) as exc_info:
        pd.DataFrame(rand_cp_nd_arr)

    assert "Implicit conversion to a NumPy array is not allowed" in str(exc_info)
    pd_df = pd.DataFrame(rand_cp_nd_arr.get())[0]
    mk_df = make_df(rand_cp_nd_arr)[0]
    assert all(pd_df.to_numpy() == mk_df.to_numpy())
