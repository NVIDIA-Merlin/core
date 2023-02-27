#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
import pytest

import merlin.dtypes as md
from merlin.core.compat import numpy as np
from merlin.core.protocols import SeriesLike
from merlin.table import NumpyColumn


@pytest.mark.parametrize("protocol", [SeriesLike])
def test_tensor_column_matches_protocols(protocol):
    obj = NumpyColumn(np.array([]))

    assert isinstance(obj, protocol)


def test_getitem():
    np_col = NumpyColumn(values=np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    assert np_col[0] == 1
    assert np_col[-1] == 8

    np_col = NumpyColumn(
        values=np.array([1, 2, 3, 4, 5, 6, 7, 8]), offsets=np.array([0, 2, 4, 6, 8])
    )
    assert all(np_col[0] == [1, 2])
    assert all(np_col[-1] == [7, 8])


def test_values():
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    np_col = NumpyColumn(values=values)
    assert all(np_col.values == values)

    np_col = NumpyColumn(values=values, offsets=np.array([0, 2, 4, 6, 8]))
    assert all(np_col.values == values)


def test_dtype():
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    np_col = NumpyColumn(values=values)
    assert np_col.dtype == md.dtype(values.dtype)

    np_col = NumpyColumn(values=values, offsets=np.array([0, 2, 4, 6, 8]))
    assert np_col.dtype == md.dtype(values.dtype)


def test_equality():
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    np_col = NumpyColumn(values=values)
    np_col_2 = NumpyColumn(values=values)
    assert np_col == np_col_2

    np_col_offs = NumpyColumn(values=values, offsets=np.array([0, 2, 4, 6, 8]))
    assert np_col != np_col_offs

    np_col_3 = NumpyColumn(values=np.array([1, 2, 3, 4]))
    assert np_col != np_col_3
