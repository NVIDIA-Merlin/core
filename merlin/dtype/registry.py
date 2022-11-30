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

    def to_merlin(self, external_dtype, shape=None):
        merlin_dtype = self.to_merlin_[external_dtype]
        return merlin_dtype

    def from_merlin(self, merlin_dtype):
        # Always translate to the first external dtype in the list
        return self.from_merlin_[merlin_dtype][0]

    def _matches(self, dtype, mapping, base_class=None):
        # If the mapping requires that the dtype is a subclass
        # of a particular base class and it isn't, then we
        # can immediately fail to match and exit.
        if base_class and not isinstance(dtype, base_class):
            return False

        # Some external dtype objects are not hashable, so they
        # can't be used as dictionary keys. In that case, match
        # against the dtype class instead.
        hashable_dtype = dtype
        try:
            hash(dtype)
        except TypeError:
            hashable_dtype = type(dtype)

        return hashable_dtype in mapping.keys()


class DTypeMappingRegistry:
    def __init__(self):
        self.mappings = {}

    def __iter__(self):
        return iter(self.mappings)

    def register(self, name: str, mapping: Union[Dict, DTypeMapping]):
        if not isinstance(mapping, DTypeMapping):
            mapping = DTypeMapping(mapping)

        self.mappings[name] = mapping

    def to_merlin(self, external_dtype, shape=None):
        for mapping in self._unique_mappings:
            if mapping.matches_external(external_dtype):
                return mapping.to_merlin(external_dtype, shape)

        raise TypeError(
            f"Merlin doesn't provide a mapping from {external_dtype} ({type(external_dtype)}) "
            "to a Merlin dtype. If you'd like to provide one, you can use "
            "`merlin.dtype.register()`."
        )

    def from_merlin(self, merlin_dtype, mapping_name):
        mapping = self.mappings[mapping_name]
        if mapping.matches_merlin(merlin_dtype):
            return mapping.to_merlin(merlin_dtype)

        raise TypeError(
            f"Merlin doesn't provide a mapping from {merlin_dtype} to a {mapping_name} dtype. "
            "If you'd like to provide one, you can use `merlin.dtype.register()`."
        )

    @property
    def _unique_mappings(self):
        """
        This is a workaround that allows us to register the same dtype mapping
        under multiple names (e.g. "tf" and "tensorflow".)

        Returns
        -------
        Set[DTypeMapping]
            The set of unique dtype mappings that have been registered
        """
        return set(self.mappings.values())


_dtype_registry = DTypeMappingRegistry()
