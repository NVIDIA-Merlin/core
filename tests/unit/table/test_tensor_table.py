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
from typing import List, Tuple

import pytest

from merlin.core.compat import cupy as cp
from merlin.core.compat import numpy as np
from merlin.core.compat import tensorflow as tf
from merlin.core.compat import torch as th
from merlin.core.dispatch import (
    HAS_GPU,
    df_from_dict,
    df_from_tensor_table,
    dict_from_df,
    make_df,
    tensor_table_from_df,
)
from merlin.core.protocols import DictLike, Transformable
from merlin.dag import BaseOperator, ColumnSelector
from merlin.table import CupyColumn, Device, NumpyColumn, TensorflowColumn, TensorTable, TorchColumn
from merlin.table.conversions import convert_col
from tests.conftest import assert_eq

array_constructors: List[Tuple] = []
cpu_target_packages: List[Tuple] = []
gpu_target_packages: List[Tuple] = []
gpu_source_col: List[Tuple] = []
cpu_source_col: List[Tuple] = []
if np:
    tensor_dict = {
        "a__values": np.array([1, 2, 3]),
        "a__offsets": np.array([0, 1, 3]),
    }
    array_constructors.append((np.array, NumpyColumn))
    cpu_target_packages.append((NumpyColumn, tensor_dict))
    cpu_source_col.append((NumpyColumn, np.array, np))

if cp:
    tensor_dict = {
        "a__values": cp.asarray([1, 2, 3]),
        "a__offsets": cp.asarray([0, 1, 3]),
    }
    array_constructors.append((cp.asarray, CupyColumn))
    gpu_target_packages.append((CupyColumn, tensor_dict))
    gpu_source_col.append((CupyColumn, cp.asarray, cp))

if tf:
    with tf.device("/CPU"):
        tensor_dict_cpu = {
            "a__values": tf.convert_to_tensor(np.array([1, 2, 3])),
            "a__offsets": tf.convert_to_tensor(np.array([0, 1, 3])),
        }
    with tf.device("/GPU:0"):
        tensor_dict_gpu = {
            "a__values": tf.convert_to_tensor(np.array([1, 2, 3])),
            "a__offsets": tf.convert_to_tensor(np.array([0, 1, 3])),
        }
    cpu_target_packages.append((TensorflowColumn, tensor_dict_cpu))
    gpu_target_packages.append((TensorflowColumn, tensor_dict_gpu))
    array_constructors.append((tf.constant, TensorflowColumn))
if th:
    tensor_dict_cpu = {
        "a__values": th.tensor([1, 2, 3], dtype=th.int32),
        "a__offsets": th.tensor([0, 1, 3], dtype=th.int32),
    }
    tensor_dict_gpu = {
        "a__values": th.tensor([1, 2, 3], dtype=th.int32).cuda(),
        "a__offsets": th.tensor([0, 1, 3], dtype=th.int32).cuda(),
    }
    cpu_target_packages.append((TorchColumn, tensor_dict_cpu))
    gpu_target_packages.append((TorchColumn, tensor_dict_gpu))
    array_constructors.append((th.tensor, TorchColumn))


@pytest.mark.parametrize("protocol", [DictLike, Transformable])
def test_tensortable_match_protocol(protocol):
    obj = TensorTable()

    assert isinstance(obj, protocol)


@pytest.mark.parametrize("array_constructor", array_constructors)
def test_tensortable_from_framework_arrays(array_constructor):
    constructor, column_type = array_constructor

    tensor_dict = {
        "a": constructor([1, 2, 3]),
        "b": constructor([3, 4, 5, 6]),
        "c": constructor([5, 6, 7]),
    }

    table = TensorTable(tensor_dict)
    assert isinstance(table, TensorTable)
    for column in table.columns:
        assert isinstance(table[column], column_type)


def test_tensortable_with_ragged_columns():
    tensor_dict = {
        "a__values": np.array([1, 2, 3]),
        "a__offsets": np.array([0, 1, 3]),
    }

    table = TensorTable(tensor_dict)
    assert table.columns == ["a"]
    assert all(table["a"].offsets == tensor_dict["a__offsets"])


class PaddingOperator(BaseOperator):
    def __init__(self, length=None, array_lib=np):
        self.length = length
        self.array_lib = array_lib

    def transform(self, col_selector, transformable):
        for col_name, col_data in transformable[col_selector.names].items():
            # dtype = col_data.dtype.to("numpy")

            dtype = self.array_lib.int32
            num_rows = len(col_data.offsets) - 1
            result = self.array_lib.zeros((num_rows, self.length), dtype=dtype)

            for i in range(num_rows):
                row_length = len(col_data[i])
                padding = self.array_lib.array([0] * (self.length - row_length), dtype=dtype)
                padded_row = self.array_lib.append(col_data[i], padding)
                result[i] = padded_row.astype(dtype)
            transformable[col_name] = type(col_data)(result)

        return transformable

    # TODO: Define what this op supports (and doesn't)


# target input, target column
# source input, source column
@pytest.mark.parametrize("source_column", cpu_source_col)
@pytest.mark.parametrize("target_column", cpu_target_packages)
def test_tensor_cpu_table_operator(source_column, target_column):
    source_column_type, source_col_constructor, array_lib = source_column
    target_column_type, target_input = target_column
    op = PaddingOperator(3, array_lib=array_lib)
    expected_output = source_col_constructor([[1, 0, 0], [2, 3, 0]])

    tensor_table = TensorTable(target_input)

    # Column conversions would happen in the executor
    for col_name, column in tensor_table.items():
        tensor_table[col_name] = convert_col(column, source_column_type)

    # Executor runs the ops
    result = op.transform(ColumnSelector(["a"]), tensor_table)

    # Column conversions would happen in the executor
    for col_name, column in result.items():
        result[col_name] = convert_col(column, target_column_type)

    # Check the results
    assert isinstance(result, TensorTable)
    for column in result.values():
        assert isinstance(column, target_column_type)

    assert result["a"].values.shape == expected_output.shape
    results = result["a"].values
    results = results.numpy() if hasattr(results, "numpy") else results
    assert np.array_equal(results, expected_output)


@pytest.mark.skipif(not cp, reason="cupy not available")
@pytest.mark.parametrize("source_column", gpu_source_col)
@pytest.mark.parametrize("target_column", gpu_target_packages)
def test_tensor_gpu_table_operator(source_column, target_column):
    source_column_type, source_col_constructor, array_lib = source_column
    target_column_type, target_input = target_column
    op = PaddingOperator(3, array_lib=array_lib)
    expected_output = source_col_constructor([[1, 0, 0], [2, 3, 0]])

    tensor_table = TensorTable(target_input)

    # Column conversions would happen in the executor
    for col_name, column in tensor_table.items():
        tensor_table[col_name] = convert_col(column, source_column_type)

    # Executor runs the ops
    result = op.transform(ColumnSelector(["a"]), tensor_table)

    # Column conversions would happen in the executor
    for col_name, column in result.items():
        result[col_name] = convert_col(column, target_column_type)

    # Check the results
    assert isinstance(result, TensorTable)
    for column in result.values():
        assert isinstance(column, target_column_type)

    assert result["a"].values.shape == expected_output.shape
    results = result["a"].values
    results = results.cpu() if hasattr(results, "cpu") else results
    results = results.numpy() if hasattr(results, "numpy") else cp.asnumpy(results)
    assert np.array_equal(results, cp.asnumpy(expected_output.get()))


def test_as_dict():
    tensor_dict = {
        "a__values": np.array([1, 2, 3]),
        "a__offsets": np.array([0, 1, 3]),
    }

    table = TensorTable(tensor_dict)

    assert table.as_dict == tensor_dict


@pytest.mark.parametrize("device", [None, "cpu"] if HAS_GPU else ["cpu"])
def test_df_to_tensor_table(device):
    df = make_df({"a": [[1, 2, 3], [4, 5, 6, 7]], "b": [1, 2]}, device=device)

    table = tensor_table_from_df(df)
    roundtrip_df = df_from_tensor_table(table)

    assert isinstance(table, TensorTable)
    expected_device = Device.CPU if device else Device.GPU
    assert table.device == expected_device

    assert_eq(df, roundtrip_df)


@pytest.mark.parametrize("device", [None, "cpu"] if HAS_GPU else ["cpu"])
def test_df_to_dict(device):
    df = make_df({"a": [[1, 2, 3], [4, 5, 6, 7]], "b": [1, 2]}, device=device)

    df_dict = dict_from_df(df)
    roundtrip_df = df_from_dict(df_dict)

    assert isinstance(df_dict, dict)
    assert_eq(df, roundtrip_df)
