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
from typing import Dict

from merlin.core.protocols import SeriesLike, Transformable


class Column(SeriesLike):
    def __init__(self, values, dtype):
        self.values = values
        self.dtype = dtype

    def __getitem__(self, index):
        return self.values[index]

    def __eq__(self, other):
        return self.data == other.data and self.dtype == other.dtype


class DictArray(Transformable):
    def __init__(self, data: Dict, dtypes: Dict):
        self.data = data
        self.dtypes = dtypes

    @property
    def columns(self):
        return list(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __eq__(self, other):
        return self.data == other.data and self.dtypes == other.dtypes

    def __setitem__(self, key, value):
        # TODO: Update dtypes here too?
        self.data[key] = value

    def __getitem__(self, key):
        if isinstance(key, list):
            return DictArray(
                data={k: self.data[k] for k in key},
                dtypes={k: self.dtypes[k] for k in key},
            )
        else:
            return Column(self.data[key], self.dtypes[key])

    def __delitem__(self, key):
        del self.data[key]

    def _grab_keys(self, source, keys):
        return {k: source[k] for k in keys}

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def update(self, other):
        self.data.update(other)

    def copy(self):
        return DictArray(self.data.copy(), self.dtypes.copy())
