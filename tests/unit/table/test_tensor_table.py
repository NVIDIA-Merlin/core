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
import numpy as np
import pytest

from merlin.core.compat import cupy as cp
from merlin.core.compat import numpy as np
from merlin.core.compat import tensorflow as tf
from merlin.core.compat import torch as th
from merlin.core.protocols import DictLike, Transformable
from merlin.dag import BaseOperator, ColumnSelector
from merlin.table import CupyColumn, NumpyColumn, TensorflowColumn, TorchColumn
from merlin.table.conversions import convert_col
from merlin.table.tensor_column import TensorColumn
from merlin.table.tensor_table import TensorTable

array_constructors = []
if np:
    array_constructors.append((np.array, NumpyColumn))

if cp:
    array_constructors.append((cp.asarray, CupyColumn))

if tf:
    array_constructors.append((tf.constant, TensorflowColumn))

if th:
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
    for column in table.columns():
        assert isinstance(table[column], column_type)


def test_tensortable_with_ragged_columns():
    tensor_dict = {
        "a__values": np.array([1, 2, 3]),
        "a__offsets": np.array([0, 1, 3]),
    }

    table = TensorTable(tensor_dict)
    assert table.columns() == ["a"]
    assert all(table["a"].offsets == tensor_dict["a__offsets"])


class PaddingOperator(BaseOperator):
    def __init__(self, length=None):
        self.length = length

    def transform(self, col_selector, transformable):
        for col_name, col_data in transformable[col_selector.names].items():
            # dtype = col_data.dtype.to("numpy")
            dtype = np.int32
            num_rows = len(col_data.offsets) - 1
            result = np.zeros((num_rows, self.length), dtype=dtype)

            for i in range(num_rows):
                row_length = len(col_data[i])
                padding = np.array([0] * (self.length - row_length), dtype=dtype)
                padded_row = np.append(col_data[i], padding)
                result[i] = padded_row.astype(dtype)
            transformable[col_name] = type(col_data)(result)

        return transformable

    # TODO: Define what this op supports (and doesn't)


def test_tensor_table_operator():
    op = PaddingOperator(3)
    expected_output = np.array([[1, 0, 0], [2, 3, 0]])

    tensor_dict = {
        "a__values": np.array([1, 2, 3]),
        "a__offsets": np.array([0, 1, 3]),
    }
    tensor_table = TensorTable(tensor_dict)

    result = op.transform(ColumnSelector(["a"]), tensor_table)

    assert isinstance(result, TensorTable)
    for column in result.values():
        assert isinstance(column, NumpyColumn)

    assert result["a"].values.shape == expected_output.shape
    assert all((result["a"].values == expected_output).reshape(-1))

    with tf.device("/CPU"):
        tensor_dict = {
            "a__values": tf.constant([1, 2, 3]),
            "a__offsets": tf.constant([0, 1, 3]),
        }
        tensor_table = TensorTable(tensor_dict)

    # Column conversions would happen in the executor
    for col_name, column in tensor_table.items():
        tensor_table[col_name] = convert_col(column, NumpyColumn)

    # Executor runs the ops
    result = op.transform(ColumnSelector(["a"]), tensor_table)

    # Column conversions would happen in the executor
    for col_name, column in result.items():
        result[col_name] = convert_col(column, TensorflowColumn)

    # Check the results
    assert isinstance(result, TensorTable)
    for column in result.values():
        assert isinstance(column, TensorflowColumn)

    assert result["a"].values.shape == expected_output.shape
    assert all(tf.reshape(result["a"].values.numpy() == expected_output, -1))


# TODO: Try the test above with Numpy and CPU Torch
# TODO: Try the test above with CuPy and GPU Tensorflow
# TODO: Try the test above with CuPy and GPU Torch

# TODO: Convert from tensor tables to dictionaries
# TODO: Convert dictionaries to tensor tables (mostly done)

# TODO: Convert from dataframes to tensor tables
# TODO: Convert from tensor tables to dataframes
