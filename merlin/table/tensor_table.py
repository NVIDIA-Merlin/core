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
from typing import Any, Dict, List, Type, Union

from merlin.dag.utils import group_values_offsets
from merlin.table.conversions import convert_col, df_from_tensor_table, tensor_table_from_df
from merlin.table.cupy_column import CupyColumn
from merlin.table.numpy_column import NumpyColumn
from merlin.table.tensor_column import TensorColumn, create_tensor_column
from merlin.table.tensorflow_column import TensorflowColumn
from merlin.table.torch_column import TorchColumn

TensorDict = Dict[str, Any]
ColumnDict = Dict[str, TensorColumn]


class TensorTable:
    """
    A DataFrameLike wrapper around a dictionary of arrays or tensors
    """

    @classmethod
    def from_df(cls, df, schema=None):
        return tensor_table_from_df(df, schema=schema)

    def __init__(self, columns: TensorDict = None, _unsafe=False):
        cols_dict = self._convert_arrays_to_columns(columns, _unsafe=_unsafe)

        if not _unsafe:
            self._validate_columns(cols_dict)

        self._columns = cols_dict

    def as_tensor_type(self, tensor_type: Union[Type, TensorColumn]) -> "TensorTable":
        """Returns new TensorTable with columns cast to tensors of target framework.

        Parameters
        ----------
        tensor_type : Union[Type, TensorColumn]
            Tensor type or TensorColumn to convert to

        Returns
        -------
        TensorTable
            A new TensorTable with columns cast to framework tensors

        Raises
        ------
        ValueError
            If tensor type provided does not match supported types
        """
        framework_columns = {}
        supported_columns = [NumpyColumn, CupyColumn, TorchColumn, TensorflowColumn]
        for _column in supported_columns:
            column_tensor_type = _column.array_type()
            if column_tensor_type:
                framework_columns[column_tensor_type] = _column

        enabled_tensor_types = ", ".join(
            f"'{_class.__module__}.{_class.__name__}'" for _class in framework_columns
        )
        enabled_column_types = ", ".join(
            f"'merlin.table.{_class.__name__}'" for _class in framework_columns.values()
        )

        if not isinstance(tensor_type, type):
            raise ValueError(
                f"tensor_type argument must be a type. Received: {type(tensor_type)} \n"
                f"Supported values are: {enabled_tensor_types}. \n"
                f"Or TensorColumn Types: {enabled_column_types}"
            )

        if issubclass(tensor_type, TensorColumn):
            target_col_type = tensor_type
        else:
            try:
                target_col_type = framework_columns[tensor_type]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported tensor type '{tensor_type}'. \n"
                    f"Supported values are: {enabled_tensor_types}. \n"
                    f"Or TensorColumn Types: {enabled_column_types}"
                ) from exc

        # if the current table already contains columns of the target type
        # no conversion required
        if self.column_type == target_col_type:
            return self

        # construct new table with columns cast to target framework type
        columns = {}
        for column_name in self.columns:
            columns[column_name] = convert_col(self[column_name], target_col_type)
        table = TensorTable(columns)

        return table

    def _convert_arrays_to_columns(self, columns, _unsafe=False):
        grouped_columns = group_values_offsets(columns or {})
        cols_dict = {}
        for name, column in grouped_columns.items():
            if isinstance(column, TensorColumn):
                cols_dict[name] = column
            elif isinstance(column, tuple):
                cols_dict[name] = create_tensor_column(column[0], column[1], _unsafe=_unsafe)
            else:
                cols_dict[name] = create_tensor_column(column, _unsafe=_unsafe)

        return cols_dict

    def _validate_columns(self, cols_dict):
        col_types = {type(col_obj) for col_obj in cols_dict.values()}
        if len(col_types) >= 2:
            raise TypeError(
                "Columns supplied to TensorTable must be backed by arrays/tensors "
                "from the same framework. Found arrays/tensors that correspond to "
                f"types {list(col_types)}."
            )

        col_devices = {col_obj.device for col_obj in cols_dict.values()}
        if len(col_devices) >= 2:
            raise ValueError(
                "Columns supplied to TensorTable must be backed by arrays/tensors "
                f" on the same device. Found arrays/tensors on devices {list(col_devices)}."
            )

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
    def columns(self) -> List[str]:
        """
        Return the names of the columns

        Exists for compatibility with the DataFrameLike protocol
        """
        return list(self.keys())

    @columns.setter
    def columns(self, col_names):
        renamed_columns = {}
        for col, col_name in zip(self.columns, col_names):
            renamed_columns[col_name] = self._columns[col]
        self._columns = renamed_columns

    @property
    def column_type(self):
        return type(list(self.values())[0])

    def dtypes(self):
        """
        Returns a list of the dtypes of all columns in the Tensor Table column
        dictionary

        Exists for compatibility with the DataFrameLike protocol
        """
        return [column.dtype for column in self.values()]

    def pop(self, column_name):
        return self._columns.pop(column_name)

    def to_dict(self):
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

    def cpu(self):
        """
        Move this TensorTable and its columns to CPU

        Returns
        -------
        TensorTable
            A new TensorTable containing the same columns but on CPU
        """
        columns = {col_name: col_values.cpu() for col_name, col_values in self.items()}
        return TensorTable(columns)

    def gpu(self):
        """
        Move this TensorTable and its columns to GPU

        Returns
        -------
        TensorTable
            A new TensorTable containing the same columns but on GPU
        """
        columns = {col_name: col_values.gpu() for col_name, col_values in self.items()}
        return TensorTable(columns)

    def to_df(self):
        """
        Convert to a dataframe
        """
        return df_from_tensor_table(self)


@create_tensor_column.register_lazy("tensorflow")
def _register_create_tf_column():
    import tensorflow as tf

    eager_tensor_type = type(tf.random.uniform((1,)))

    @create_tensor_column.register(tf.Tensor)
    @create_tensor_column.register(eager_tensor_type)
    def _create_tensor_column_tf(values, offsets=None, _unsafe=False):
        return TensorflowColumn(values, offsets, _unsafe=_unsafe)


@create_tensor_column.register_lazy("torch")
def _register_create_torch_column():
    import torch as th

    @create_tensor_column.register(th.Tensor)
    def _create_tensor_column_torch(values, offsets=None, _unsafe=False):
        return TorchColumn(values, offsets, _unsafe=_unsafe)


@create_tensor_column.register_lazy("numpy")
def _register_create_numpy_column():
    import numpy as np

    @create_tensor_column.register(np.ndarray)
    def _create_tensor_column_numpy(values, offsets=None, _unsafe=False):
        return NumpyColumn(values, offsets, _unsafe=_unsafe)


@create_tensor_column.register_lazy("cupy")
def _register_create_cupy_column():
    import cupy as cp

    @create_tensor_column.register(cp.ndarray)
    def _create_tensor_column_cupy(values, offsets=None, _unsafe=False):
        return CupyColumn(values, offsets, _unsafe=_unsafe)
