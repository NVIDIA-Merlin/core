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
from functools import singledispatch
from typing import Any, Dict, Tuple

from merlin.dag.utils import group_values_offsets
from merlin.dispatch.lazy import lazysingledispatch
from merlin.table.cupy_column import CupyColumn
from merlin.table.numpy_column import NumpyColumn
from merlin.table.tensor_column import TensorColumn
from merlin.table.tensorflow_column import TensorflowColumn
from merlin.table.torch_column import TorchColumn

TensorDict = Dict[str, Any]
ColumnDict = Dict[str, TensorColumn]

# TODO: Figure out if/how to put the column names in TensorColumn


class TensorTable:
    def __init__(self, columns: TensorDict = None):
        grouped_columns = group_values_offsets(columns or {})

        dict_cols = {}
        for name, column in grouped_columns.items():
            if isinstance(column, TensorColumn):
                dict_cols[name] = column
            elif isinstance(column, Tuple):
                dict_cols[name] = create_tensor_column(column[0], column[1])
            else:
                dict_cols[name] = create_tensor_column(column)

        self._columns = dict_cols

    def __iter__(self):
        return iter(self._columns)

    def __len__(self):
        return len(self._columns)

    def __getitem__(self, key):
        if isinstance(key, list):
            return TensorTable({k: self._columns[k] for k in key})
        else:
            return self._columns[key]

    def __setitem__(self, key, value):
        self._columns[key] = value

    def __delitem__(self, key):
        del self._columns[key]

    def keys(self):
        return self._columns.keys()

    def items(self):
        return self._columns.items()

    def values(self):
        return self._columns.values()

    def update(self, other):
        self._columns.update(other._columns)

    def copy(self):
        return TensorTable(self._columns.copy())

    def columns(self):
        return list(self.keys())

    def dtypes(self):
        return [column.dtype for column in self.values()]


@lazysingledispatch
def create_tensor_column(values, offsets=None):
    raise NotImplementedError()


@create_tensor_column.register_lazy("tensorflow")
def register_create_tf_column():
    import tensorflow as tf

    eager_tensor_type = type(tf.random.uniform((1,)))

    @create_tensor_column.register(tf.Tensor)
    @create_tensor_column.register(eager_tensor_type)
    def _create_tensor_column_tf(values, offsets=None):
        return TensorflowColumn(values, offsets)


@create_tensor_column.register_lazy("torch")
def register_create_tf_column():
    import torch as th

    @create_tensor_column.register(th.Tensor)
    def _create_tensor_column_torch(values, offsets=None):
        return TorchColumn(values, offsets)


@create_tensor_column.register_lazy("numpy")
def register_create_tf_column():
    import numpy as np

    @create_tensor_column.register(np.ndarray)
    def _create_tensor_column_numpy(values, offsets=None):
        return NumpyColumn(values, offsets)


@create_tensor_column.register_lazy("cupy")
def register_create_tf_column():
    import cupy as cp

    @create_tensor_column.register(cp.ndarray)
    def _create_tensor_column_cupy(values, offsets=None):
        return CupyColumn(values, offsets)
