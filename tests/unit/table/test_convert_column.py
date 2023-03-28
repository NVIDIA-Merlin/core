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
from typing import List, Type

import pytest

from merlin.core.compat import HAS_GPU
from merlin.core.compat import cupy as cp
from merlin.core.compat import numpy as np
from merlin.core.compat import tensorflow as tf
from merlin.core.compat import torch as th
from merlin.table import CupyColumn, NumpyColumn, TensorColumn, TensorflowColumn, TorchColumn
from merlin.table.conversions import convert_col

source_cols: List[TensorColumn] = []
output_col_types: List[Type] = []

if cp and HAS_GPU:
    cp_array = cp.asarray([1, 2, 3, 4])

    source_cols.append(CupyColumn(values=cp_array, offsets=cp_array))
    output_col_types.append(CupyColumn)

if np:
    np_array = np.array([1, 2, 3, 4])

    source_cols.append(NumpyColumn(values=np_array, offsets=np_array))
    output_col_types.append(NumpyColumn)

if tf and HAS_GPU:
    with tf.device("/CPU"):
        tf_tensor = tf.convert_to_tensor(np.array([1, 2, 3, 4]))
        offsets_tensor = tf.convert_to_tensor(np.array([0, 1, 2, 3, 4]))
        cpu_tf_column = TensorflowColumn(values=tf_tensor, offsets=tf_tensor)
    with tf.device("/GPU:0"):
        tf_tensor = tf.convert_to_tensor(np.array([1, 2, 3, 4]))
        offsets_tensor = tf.convert_to_tensor(np.array([0, 1, 2, 3, 4]))
        gpu_tf_column = TensorflowColumn(values=tf_tensor, offsets=offsets_tensor)

    source_cols.extend([cpu_tf_column, gpu_tf_column])
    output_col_types.append(TensorflowColumn)

if th and HAS_GPU:
    th_tensor = th.tensor([1, 2, 3, 4])
    cpu_th_column = TorchColumn(values=th_tensor, offsets=th_tensor)

    th_tensor = th.tensor([1, 2, 3, 4]).cuda()
    gpu_th_column = TorchColumn(values=th_tensor, offsets=th_tensor)

    source_cols.extend([cpu_th_column, gpu_th_column])
    output_col_types.append(TorchColumn)


@pytest.mark.parametrize("source_cols", source_cols)
@pytest.mark.parametrize("output_col", output_col_types)
def test_convert_col(source_cols, output_col):
    if source_cols.device not in output_col.supported_devices():
        with pytest.raises(NotImplementedError) as exc:
            converted_col = convert_col(source_cols, output_col)
        assert "Could not convert from type" in str(exc.value)
    else:
        converted_col = convert_col(source_cols, output_col)
        assert isinstance(converted_col, output_col)


@pytest.mark.parametrize("output_col", output_col_types)
def test_3d_convert_np(output_col):
    arr = []
    row_lengths = []
    batch_size = 3
    embedding_size = 20
    row_length_indexes = [1, 2, 3]
    for idx, x in enumerate(range(batch_size)):
        # simulate raggedness
        row_length = row_length_indexes[idx]
        arr.append(np.random.rand(row_length, embedding_size).tolist())
        row_lengths.append(row_length)
    num_embeddings = sum(row_lengths)
    column = NumpyColumn(arr)

    assert isinstance(column, NumpyColumn)
    assert column.shape.as_tuple == (batch_size, (0, None), embedding_size)
    assert column.values.shape[0] == num_embeddings
    assert column.values.shape[1] == embedding_size
    if column.device not in output_col.supported_devices():
        with pytest.raises(NotImplementedError) as exc:
            converted_col = convert_col(column, output_col)
        assert "Could not convert from type" in str(exc.value)
    else:
        converted_col = convert_col(column, output_col)
        assert isinstance(converted_col, output_col)
        assert converted_col.shape.as_tuple == (batch_size, (0, None), embedding_size)
        assert converted_col.values.shape[0] == num_embeddings
        assert converted_col.values.shape[1] == embedding_size


@pytest.mark.skipif(not cp, reason="cupy not available")
@pytest.mark.parametrize("output_col", output_col_types)
def test_3d_convert_cp(output_col):
    arr = []
    row_lengths = []
    batch_size = 3
    embedding_size = 20
    row_length_indexes = [1, 2, 3]
    for idx, x in enumerate(range(batch_size)):
        # simulate raggedness
        row_length = row_length_indexes[idx]
        arr.append(np.random.rand(row_length, embedding_size).tolist())
        row_lengths.append(row_length)
    num_embeddings = sum(row_lengths)
    column = CupyColumn(arr)

    assert isinstance(column, CupyColumn)
    assert column.shape.as_tuple == (batch_size, (0, None), embedding_size)
    assert column.values.shape[0] == num_embeddings
    assert column.values.shape[1] == embedding_size
    if column.device not in output_col.supported_devices():
        with pytest.raises(NotImplementedError) as exc:
            converted_col = convert_col(column, output_col)
        assert "Could not convert from type" in str(exc.value)
    else:
        converted_col = convert_col(column, output_col)
        assert isinstance(converted_col, output_col)
        assert converted_col.shape.as_tuple == (batch_size, (0, None), embedding_size)
        assert converted_col.values.shape[0] == num_embeddings
        assert converted_col.values.shape[1] == embedding_size


@pytest.mark.skipif(not cp, reason="cupy not available")
@pytest.mark.parametrize("output_col", output_col_types)
def test_3d_convert_cp_nd(output_col):
    batch_size = 1

    embedding_size = 1
    data = cp.asarray([[1]])
    column = CupyColumn(data)

    assert isinstance(column, CupyColumn)
    assert column.shape.as_tuple == (batch_size, embedding_size)
    assert column.values.shape[1] == embedding_size
    if column.device not in output_col.supported_devices():
        with pytest.raises(NotImplementedError) as exc:
            converted_col = convert_col(column, output_col)
        assert "Could not convert from type" in str(exc.value)
    else:
        converted_col = convert_col(column, output_col)
        assert isinstance(converted_col, output_col)
        assert converted_col.shape.as_tuple == (batch_size, embedding_size)
        assert converted_col.values.shape[1] == embedding_size
