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
from typing import Any, Dict, List

from merlin.features.array.base import MerlinArray


class VirtualDataframe:
    def __init__(self, data: Dict[str, Any] = None):
        data = data or {}
        converted_cols = {}
        for col_name, col_values in data.items():
            converted_col = (
                col_values if isinstance(col_values, MerlinArray) else MerlinArray.build(col_values)
            )
            converted_cols[col_name] = converted_col

        self._col_data = converted_cols

    @classmethod
    def from_df(cls, other_df):
        col_series = {col_name: other_df[col_name] for col_name in other_df.columns}
        return VirtualDataframe(col_series)

    def to(self, array_type):
        merlin_array_type = MerlinArray.array_types[array_type]
        converted_cols = {}
        for col_name, col_array in self._col_data.items():
            converted_cols[col_name] = merlin_array_type(col_array)
        return VirtualDataframe(converted_cols)

    @property
    def columns(self) -> List[str]:
        return list(self._col_data.keys())

    def __getitem__(self, col_items):
        if isinstance(col_items, list):
            results = {name: self._col_data[name].array for name in col_items}
            return VirtualDataframe(results)
        else:
            return self._col_data[col_items].array

    def __len__(self):
        return len(self.columns)

    def __iter__(self):
        for name, tensor in self._col_data.items():
            yield name, tensor

    def __repr__(self):
        dict_rep = {}
        for k, v in self._col_data.items():
            dict_rep[k] = v
        return str(dict_rep)
