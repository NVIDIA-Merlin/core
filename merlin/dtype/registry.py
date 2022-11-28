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
import dataclasses
from typing import Dict, Union


class DTypeMapping:
    def __init__(self, mapping, base_class=None):
        self.from_merlin_ = mapping
        self.to_merlin_ = {}
        self.base_class = base_class

        for key, values in mapping.items():
            if not isinstance(values, list):
                values = [values]
            for value in values:
                self.to_merlin_[value] = key

    def matches_external(self, dtype):
        return self._matches(dtype, self.to_merlin_, self.base_class)

    def matches_merlin(self, dtype):
        return self._matches(dtype, self.from_merlin_)

    def to_merlin(self, external_dtype, shape = None):
        merlin_dtype = self.to_merlin_[external_dtype]
        return dataclasses.replace(merlin_dtype, **{"shape": shape})

    def from_merlin(self, merlin_dtype):
        # Ignore the shape when matching dtypes
        shapeless_merlin_dtype = dataclasses.replace(merlin_dtype, **{"shape": None})
        # Always translate to the first external dtype in the list
        return self.from_merlin_[shapeless_merlin_dtype][0]

    def _matches(self, dtype, mapping, base_class = None):
        if base_class and not isinstance(dtype, base_class):
            return False

        return dtype in mapping.keys()


class DTypeMappingRegistry:
    def __init__(self):
        self.mappings = {}

    def __iter__(self):
        return iter(self.mappings)

    def register(self, name: str, mapping: Union[Dict, DTypeMapping]):
        if not isinstance(mapping, DTypeMapping):
            mapping = DTypeMapping(mapping)

        self.mappings[name] = mapping


_dtype_registry = DTypeMappingRegistry()
