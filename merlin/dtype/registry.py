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

from merlin.dtype.mapping import DTypeMapping


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
