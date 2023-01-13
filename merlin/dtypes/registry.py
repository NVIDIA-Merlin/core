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

from merlin.dtypes.mapping import DTypeMapping


class DTypeMappingRegistry:
    """
    A registry of mappings between Merlin dtypes and the dtypes of many external frameworks
    """

    def __init__(self):
        self.mappings = {}

    def __iter__(self):
        return iter(self.mappings)

    def register(self, name: str, mapping: Union[Dict, DTypeMapping]):
        """
        Register a mapping between Merlin and external dtypes by name

        Parameters
        ----------
        name : str
            Name of the new mapping to register
        mapping : Union[Dict, DTypeMapping]
            Mapping between Merlin and external dtypes
        """
        if not isinstance(mapping, DTypeMapping):
            mapping = DTypeMapping(mapping)

        self.mappings[name] = mapping

    def from_merlin(self, merlin_dtype, mapping_name):
        """
        Map a Merlin dtype to an external dtype

        Parameters
        ----------
        merlin_dtype : DType
            A Merlin dtype object
        mapping_name : str
            The name of the external framework mapping to apply

        Returns
        -------
        Any
            An external framework dtype object

        Raises
        ------
        TypeError
            If the Merlin dtype can't be mapped to an external dtype from the requested framework
        """
        mapping = self.mappings[mapping_name]
        if mapping.matches_merlin(merlin_dtype):
            return mapping.to_merlin(merlin_dtype)

        raise TypeError(
            f"Merlin doesn't provide a mapping from {merlin_dtype} to a {mapping_name} dtype. "
            "If you'd like to provide one, you can use `merlin.dtype.register()`."
        )

    def to_merlin(self, external_dtype):
        """
        Map an external dtype to a Merlin dtype

        Parameters
        ----------
        external_dtype : Any
            A dtype object from an external framework

        Returns
        -------
        DType
            A Merlin DType object

        Raises
        ------
        TypeError
            If the external dtype can't be mapped to a Merlin dtype
        """
        for framework, mapping in self.mappings.items():
            if mapping.matches_external(external_dtype):
                return mapping.to_merlin(external_dtype)

        raise TypeError(
            f"Merlin doesn't provide a mapping from {external_dtype} ({type(external_dtype)}) "
            "to a Merlin dtype. If you'd like to provide one, you can use "
            "`merlin.dtype.register()`."
        )

    def to_merlin_via_numpy(self, external_dtype):
        """
        Map an external dtype to a Merlin dtype by converting the external type to Numpy first

        This is sometimes useful for external framework dtypes that don't have a clear
        one-to-one mapping with a Merlin dtype, like cuDF's CategoricalDtype. We can often do
        some additional preprocessing on the external framework's dtype to determine the
        Numpy dtype of the elements, and then use that as an intermediary to find the
        corresponding Merlin dtype.

        Parameters
        ----------
        external_dtype : Any
            A dtype object from an external framework

        Returns
        -------
        DType
            A Merlin DType object

        Raises
        ------
        TypeError
            If the external dtype can't be mapped to a Merlin dtype
        """
        numpy_dtype = None

        for mapping in self.mappings.values():
            if mapping.translator and mapping.translator.matches(external_dtype):
                numpy_dtype = mapping.translator.to_numpy(external_dtype)
                break

        return self.to_merlin(numpy_dtype)


_dtype_registry = DTypeMappingRegistry()
