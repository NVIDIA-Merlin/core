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
from typing import Any, Dict, List, Protocol, runtime_checkable

from merlin.features.array.base import MerlinArray


@runtime_checkable
class DataFrame(Protocol):
    """
    This Protocol matches either real Pandas/cuDF dataframes
    or the VirtualDataFrame class defined above when an object
    is checked with `is_instance(obj, Features)` which returns
    `True` for any object that defines all of the methods below
    """

    @property
    def columns(self):
        ...

    def __getitem__(self, key):
        ...

    def __setitem__(self, key, value):
        ...

    def __iter__(self):  # pylint: disable=non-iterator-returned
        ...


class VirtualDataFrame:
    """
    A dataframe-like representation that simulates a real dataframe, given a dictionary
    of array-like objects (e.g. numpy.ndarray, tensorflow.Tensor) from external frameworks
    supported by Merlin that represent the "column" data of the VirtualDataFrame.
    """

    def __init__(self, data: Dict[str, Any] = None):
        data = data or {}
        converted_cols = {}
        for col_name, col_data in data.items():
            converted_col = (
                col_data if isinstance(col_data, MerlinArray) else MerlinArray.build(col_data)
            )
            converted_cols[col_name] = converted_col

        self._col_data = converted_cols

    @classmethod
    def from_df(cls, other_df: DataFrame) -> "VirtualDataFrame":
        """
        Create virtual dataframe from another dataframe

        Parameters
        ----------
        other_df : DataFrameType
            A pandas or cudf dataframe.

        Returns
        -------
        VirtualDataFrame
            A VirtualDataFrame that has the same data as the DataFrame supplied.
        """

        col_series = {col_name: other_df[col_name] for col_name in other_df.columns}
        return VirtualDataFrame(col_series)

    def columns_to(self, array_type) -> "VirtualDataFrame":
        """
        Convert the columns of this dataframe to a
        framework array-like type (e.g. cupy.ndarray)

        Parameters
        ----------
        array_type : array-like type
            An array type from an external framework

        Returns
        -------
        VirtualDataFrame
            A new VirtualDataFrame with the columns converted to the new type
        """
        merlin_array_type = MerlinArray.array_types[array_type]
        converted_cols = {}
        for col_name in self:
            converted_cols[col_name] = merlin_array_type(self[col_name])
        return VirtualDataFrame(converted_cols)

    @property
    def columns(self) -> List[str]:
        return list(self._col_data.keys())

    def __getitem__(self, col_items):
        if isinstance(col_items, list):
            results = {name: self[name].array for name in col_items}
            return VirtualDataFrame(results)
        else:
            return self._col_data[col_items].array

    def __setitem__(self, col_name, col_values):
        converted_col = (
            col_values if isinstance(col_values, MerlinArray) else MerlinArray.build(col_values)
        )
        self._col_data[col_name] = converted_col

    def __len__(self):
        return len(self.columns)

    def __iter__(self):
        for col_name in self.columns:
            yield col_name

    def __repr__(self):
        return str(self._col_data)
