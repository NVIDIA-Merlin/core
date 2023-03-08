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
from typing import Any, List, Tuple, Type

import pytest

import merlin.dtypes as md
from merlin.core.compat import HAS_GPU
from merlin.core.compat import cupy as cp
from merlin.core.compat import numpy as np
from merlin.core.compat import tensorflow as tf
from merlin.core.compat import torch as th
from merlin.core.protocols import SeriesLike
from merlin.dtypes.shape import Shape
from merlin.table import CupyColumn, Device, NumpyColumn, TensorflowColumn, TorchColumn

col_types_and_constructors: List[Tuple[Type, Any]] = []

if np:
    col_types_and_constructors.append((NumpyColumn, np.array))

if cp:
    col_types_and_constructors.append((CupyColumn, cp.array))

if tf:
    col_types_and_constructors.append((TensorflowColumn, tf.constant))

if th:
    col_types_and_constructors.append((TorchColumn, th.tensor))


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


@pytest.mark.skipif(not (cp and HAS_GPU), reason="requires GPU")
def test_cupy_cpu_transfer():
    values = cp.array([1, 2, 3])
    offsets = cp.array([0, 1, 3])

    gpu_col = CupyColumn(values, offsets)
    cpu_col = gpu_col.cpu()

    assert cpu_col.device == Device.CPU
    assert isinstance(cpu_col, NumpyColumn)

    cpu_col_again = cpu_col.cpu()

    assert cpu_col_again.device == Device.CPU
    assert isinstance(cpu_col_again, NumpyColumn)


@pytest.mark.skipif(not (cp and HAS_GPU), reason="requires GPU")
def test_numpy_gpu_transfer():
    values = np.array([1, 2, 3])
    offsets = np.array([0, 1, 3])

    cpu_col = NumpyColumn(values, offsets)
    gpu_col = cpu_col.gpu()

    assert gpu_col.device == Device.GPU
    assert isinstance(gpu_col, CupyColumn)

    gpu_col_again = gpu_col.gpu()

    assert gpu_col_again.device == Device.GPU
    assert isinstance(gpu_col_again, CupyColumn)


@pytest.mark.skipif(not (HAS_GPU and th), reason="requires Torch and GPU")
def test_torch_data_transfer():
    values = th.tensor([1, 2, 3])
    offsets = th.tensor([0, 1, 3])

    cpu_col = TorchColumn(values, offsets)
    gpu_col = cpu_col.gpu()
    cpu_col_again = gpu_col.cpu()

    assert gpu_col.device == Device.GPU
    assert cpu_col_again.device == Device.CPU


@pytest.mark.skipif(not (tf and HAS_GPU), reason="requires TensorFlow and GPU")
def test_tf_data_transfer():
    values = tf.constant([1, 2, 3])
    offsets = tf.constant([0, 1, 3])

    cpu_col = TensorflowColumn(values, offsets)
    gpu_col = cpu_col.gpu()
    cpu_col_again = gpu_col.cpu()

    assert gpu_col.device == Device.GPU
    assert cpu_col_again.device == Device.CPU


@pytest.mark.parametrize("col_type, constructor", col_types_and_constructors)
def test_shape(col_type, constructor):
    values = constructor([1, 2, 3, 4, 5, 6, 7, 8])
    col = col_type(values=values)
    assert col.shape == Shape((8,))
    assert len(col) == 8
    assert col.is_list is False
    assert col.is_ragged is False

    values = constructor([[1, 2, 3, 4, 5, 6, 7, 8]])
    col = col_type(values=values)
    assert col.shape == Shape((1, 8))
    assert len(col) == 1
    assert col.is_list is True
    assert col.is_ragged is False

    values = constructor([1, 2, 3, 4, 5, 6, 7, 8])
    offsets = constructor([0, 2, 4, 6, 8])
    col = col_type(values=values, offsets=offsets)
    assert col.shape == Shape((4, 2))
    assert len(col) == 4
    assert col.is_list is True
    assert col.is_ragged is False

    values = constructor([1, 2, 3, 4, 5, 6, 7, 8])
    offsets = constructor([0, 1, 3, 5, 8])
    col = col_type(values=values, offsets=offsets)
    assert col.shape == Shape((4, None))
    assert len(col) == 4
    assert col.is_list is True
    assert col.is_ragged is True
