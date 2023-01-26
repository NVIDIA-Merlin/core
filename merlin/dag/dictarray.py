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
from typing import Dict, Optional

import numpy as np
import pandas as pd

from merlin.core.dispatch import (
    build_cudf_list_column,
    build_pandas_list_column,
    get_lib,
    is_list_dtype,
)
from merlin.dag.column import Column


class DictArray:
    """
    A simple dataframe-like wrapper around a dictionary of values. Matches the Transformable
    protocol for (limited) interchangeability with actual dataframes in Merlin DAGs.
    """

    def __init__(self, values: Optional[Dict] = None):
        super().__init__()
        values = values or {}

        self._columns = {key: _make_column(value) for key, value in values.items()}

    @property
    def columns(self):
        return list(self._columns.keys())

    # TODO: Make this a property once the underlying Transformable protocol has been adjusted
    def dtypes(self):
        return {key: value.dtype for key, value in self._columns.items()}

    def __len__(self):
        return len(self._columns)

    def __iter__(self):
        return iter(self._columns)

    def __eq__(self, other):
        return self.values() == other.values() and self.dtypes() == other.dtypes()

    def __setitem__(self, key, value):
        self._columns[key] = _make_column(value)

    def __getitem__(self, key):
        if isinstance(key, list):
            return DictArray({k: self._columns[k] for k in key})
        else:
            return self._columns[key]

    def __delitem__(self, key):
        del self._columns[key]

    def keys(self):
        """
        Shortcut to get the dictionary keys
        """
        return self._columns.keys()

    def items(self):
        """
        Shortcut to get the dictionary items
        """
        return self._columns.items()

    def values(self):
        """
        Shortcut to get the dictionary values
        """
        return self._columns.values()

    def update(self, other):
        """
        Shortcut to update the dictionary items
        """
        self._columns.update(other)

    def copy(self):
        """
        Create a new DictArray with the same data and dtypes
        """
        return DictArray(self._columns.copy())

    def to_df(self):
        """
        Create a DataFrame from the DictArray
        """
        df_lib = get_lib()
        df = df_lib.DataFrame()
        for col_name, column in self.items():
            if column.is_list:
                values_series = df_lib.Series(column.values.flatten()).reset_index(drop=True)

                if isinstance(values_series, pd.Series):
                    row_lengths_series = df_lib.Series(column.row_lengths)
                    df[col_name] = build_pandas_list_column(values_series, row_lengths_series)
                else:
                    values_size = df_lib.Series(column.values.shape[0]).reset_index(drop=True)
                    offsets_series = (
                        df_lib.Series(column.offsets)
                        .append(values_size)
                        .reset_index(drop=True)
                        .astype("int32")
                    )
                    df[col_name] = build_cudf_list_column(values_series, offsets_series)
            else:
                df[col_name] = df_lib.Series(column.values)
        return df

    @classmethod
    def from_df(cls, df):
        """
        Create a DictArray from a DataFrame
        """
        array_dict = {}
        for col in df.columns:
            if is_list_dtype(df[col]):
                series = df[col]
                if isinstance(series, pd.Series):
                    array_dict[col] = (series.to_numpy(), get_series_offsets(series))
                else:
                    # cudf series that is a list to_pandas to keep values in non-flattened format
                    array_dict[col] = (series.to_pandas().to_numpy(), get_series_offsets(series))
            else:
                array_dict[col] = df[col].to_numpy()
        return cls(array_dict)


def get_series_offsets(series):
    if not is_list_dtype(series):
        # no offsets
        return np.asarray([])
    row_lengths = []
    if isinstance(series, pd.Series):
        for row in series:
            row_lengths.append(len(row))
        return np.asarray(row_lengths)
    else:
        return series._column.offsets.values[1:] - series._column.offsets.values[:-1]


def _make_column(value):
    # If it's already a column, there's nothing to do
    if isinstance(value, Column):
        return value

    # Handle (values, lengths) tuples
    if isinstance(value, tuple):
        values, row_lengths = value
        return Column(values, row_lengths=row_lengths)

    # Otherwise, assume value is a Numpy/Cupy array
    if hasattr(value, "shape"):
        if len(value.shape) > 2:
            raise ValueError("Values with dimensions higher than 2 aren't supported.")
        elif len(value.shape) == 2:
            num_values = value.shape[0]
            row_lengths = [value.shape[1]] * num_values
            # raise ValueError(value, row_lengths)
            return Column(value, row_lengths=row_lengths)

    column = Column(value) if not isinstance(value, Column) else value
    return column
