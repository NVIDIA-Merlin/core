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
from merlin.table import CupyColumn, NumpyColumn, TensorflowColumn, TorchColumn
from merlin.table.tensor_column import Device


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
    "source_class,constructor,dtype,source_device",
    [
        (NumpyColumn, np.array, np.int32, Device.CPU),
        (CupyColumn, cp.array, cp.int32, Device.GPU),
        (TensorflowColumn, tf.constant, tf.int32, Device.GPU),
        (TorchColumn, th.tensor, th.int32, Device.CPU),
        # TODO: Create Torch GPU tensor
        # TODO: Create Tensorflow CPU tensor
    ],
)
@pytest.mark.parametrize(
    "dest_class,dest_type,dest_device",
    [
        (NumpyColumn, np.ndarray, Device.CPU),
        (CupyColumn, cp._core.core.ndarray, Device.GPU),
        (TensorflowColumn, tf.Tensor, None),
        (TorchColumn, th.Tensor, None),
    ],
)
def test_column_casting(
    source_class, constructor, dtype, source_device, dest_class, dest_type, dest_device
):
    dest_device = dest_device or source_device

    values = constructor([1, 2, 3, 4, 5], dtype=dtype)
    offsets = constructor([0, 3, 5], dtype=dtype)

    source_column = source_class(values, offsets)

    assert source_column.device == source_device

    dest_column = dest_class.cast(source_column)

    assert dest_column is not None
    assert isinstance(dest_column, dest_class)
    assert isinstance(dest_column.values, dest_type)
    assert isinstance(dest_column.offsets, dest_type)
    assert dest_column.device == dest_device
