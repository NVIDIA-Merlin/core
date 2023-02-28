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
from typing import Any, Dict

from merlin.dag.utils import group_values_offsets
from merlin.dispatch.lazy import lazy_singledispatch
from merlin.table.cupy_column import CupyColumn
from merlin.table.numpy_column import NumpyColumn
from merlin.table.tensor_column import TensorColumn
from merlin.table.tensorflow_column import TensorflowColumn
from merlin.table.torch_column import TorchColumn

TensorDict = Dict[str, Any]
ColumnDict = Dict[str, TensorColumn]


class TensorTable:
    """
    A DataFrameLike wrapper around a dictionary of arrays or tensors
    """

    def __init__(self, columns: TensorDict = None):
        grouped_columns = group_values_offsets(columns or {})
        dict_cols = {}
        for name, column in grouped_columns.items():
            if isinstance(column, TensorColumn):
                dict_cols[name] = column
            elif isinstance(column, tuple):
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

    @property
    def device(self):
        return list(self.values())[0].device

    def keys(self):
        """
        Return the keys (i.e. column names) of the column dictionary

        Exists for compatibility with the DictionaryLike protocol
        """
        return self._columns.keys()

    def items(self):
        """
        Return the keys and values (i.e. column names and column data)
        of the column dictionary

        Exists for compatibility with the DictionaryLike protocol
        """
        return self._columns.items()

    def values(self):
        """
        Return the values (i.e. columns) of the column dictionary

        Exists for compatibility with the DictionaryLike protocol
        """
        return self._columns.values()

    def update(self, other):
        """
        Update the column dictionary using the columns from another TensorTable

        Exists for compatibility with the DictionaryLike protocol
        """
        self._columns.update(other._columns)

    def copy(self):
        """
        Create a copy of the Tensor Table, the column dictionary

        Exists for compatibility with the DictionaryLike protocol
        """
        return TensorTable(self._columns.copy())

    @property
    def columns(self):
        """
        Return the names of the columns

        Exists for compatibility with the DataFrameLike protocol
        """
        return list(self.keys())

    def dtypes(self):
        """
        Returns a list of the dtypes of all columns in the Tensor Table column
        dictionary

        Exists for compatibility with the DataFrameLike protocol
        """
        return [column.dtype for column in self.values()]

    @property
    def as_dict(self):
        """
        Convert to a flat dictionary of arrays or tensors

        Ragged columns will be represented as values and offsets
        """
        result = {}
        for col_name, tensor_col in self._columns.items():
            if tensor_col.offsets is not None:
                result[f"{col_name}__values"] = tensor_col.values
                result[f"{col_name}__offsets"] = tensor_col.offsets
            else:
                result[col_name] = tensor_col.values
        return result


@lazy_singledispatch
def create_tensor_column(values, offsets=None):
    """
    Create the appropriate TensorColumn subclass from the type of the supplied values and offsets
    """
    raise NotImplementedError()


@create_tensor_column.register_lazy("tensorflow")
def _register_create_tf_column():
    import tensorflow as tf

    eager_tensor_type = type(tf.random.uniform((1,)))

    @create_tensor_column.register(tf.Tensor)
    @create_tensor_column.register(eager_tensor_type)
    def _create_tensor_column_tf(values, offsets=None):
        return TensorflowColumn(values, offsets)


@create_tensor_column.register_lazy("torch")
def _register_create_torch_column():
    import torch as th

    @create_tensor_column.register(th.Tensor)
    def _create_tensor_column_torch(values, offsets=None):
        return TorchColumn(values, offsets)


@create_tensor_column.register_lazy("numpy")
def _register_create_numpy_column():
    import numpy as np

    @create_tensor_column.register(np.ndarray)
    def _create_tensor_column_numpy(values, offsets=None):
        return NumpyColumn(values, offsets)


@create_tensor_column.register_lazy("cupy")
def _register_create_cupy_column():
    import cupy as cp

    @create_tensor_column.register(cp.ndarray)
    def _create_tensor_column_cupy(values, offsets=None):
        return CupyColumn(values, offsets)
