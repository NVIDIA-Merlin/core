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
import cupy as cp
import numpy as np
import pytest
import tensorflow as tf
import torch as th

import merlin.dtypes as md
from merlin.dag.table import CupyColumn, NumpyColumn, TensorflowColumn, TorchColumn


@pytest.mark.parametrize(
    "column_class,constructor,dtype",
    [
        (NumpyColumn, np.array, np.int32),
        (CupyColumn, cp.array, cp.int32),
        (TensorflowColumn, tf.constant, tf.int32),
        (TorchColumn, th.tensor, th.int32),
    ],
)
def test_array_types(column_class, constructor, dtype):
    values = constructor([1, 2, 3, 4, 5], dtype=dtype)
    offsets = constructor([0, 3, 5], dtype=dtype)

    column = column_class(values)
    assert all(column.values == values)

    column = column_class(values, offsets)
    assert all(column.values == values)
    assert all(column.offsets == offsets)

    assert column.dtype == md.int32


@pytest.mark.parametrize(
    "source_class,constructor,dtype",
    [
        (NumpyColumn, np.array, np.int32),
        (CupyColumn, cp.array, cp.int32),
        (TensorflowColumn, tf.constant, tf.int32),
        (TorchColumn, th.tensor, th.int32),
    ],
)
@pytest.mark.parametrize(
    "dest_class,dest_type",
    [
        (NumpyColumn, np.ndarray),
        (CupyColumn, cp._core.core.ndarray),
        (TensorflowColumn, tf.Tensor),
        (TorchColumn, th.Tensor),
    ],
)
def test_column_casting(source_class, constructor, dtype, dest_class, dest_type):
    values = constructor([1, 2, 3, 4, 5], dtype=dtype)
    offsets = constructor([0, 3, 5], dtype=dtype)

    source_column = source_class(values, offsets)
    dest_column = dest_class.cast(source_column)

    assert dest_column is not None
    assert isinstance(dest_column, dest_class)
    assert isinstance(dest_column.values, dest_type)
    assert isinstance(dest_column.offsets, dest_type)
