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
from typing import List, Optional

import numpy as np


class NumpyPreprocessor:
    """
    Allows converting framework dtypes to numpy dtypes before mapping to Merlin types
    """

    def __init__(
        self,
        framework,
        translation_fn,
        attrs: Optional[List[str]] = None,
        classes: Optional[List[type]] = None,
    ):
        self.framework = framework
        self.translation_fn = translation_fn
        self.attrs = attrs or []
        self.classes = classes or []

    def matches(self, raw_dtype) -> bool:
        """
        Check if this preprocessor has a translation available for a dtype

        Parameters
        ----------
        raw_dtype : external framework dtype
            The dtype to be translated to numpy

        Returns
        -------
        bool
            True if this preprocessor can convert the external dtype to Numpy
        """
        for attr in self.attrs:
            if hasattr(raw_dtype, attr):
                return True
        for cls_ in self.classes:
            if isinstance(raw_dtype, cls_):
                return True
        return False

    def to_numpy(self, raw_dtype) -> np.dtype:
        """
        Translate an external framework dtype to Numpy

        Parameters
        ----------
        raw_dtype : external framework dtype
            The dtype to be translated to numpy

        Returns
        -------
        np.dtype
            The result of translating raw_dtype to Numpy
        """
        return self.translation_fn(raw_dtype)


class DTypeMapping:
    """
    A mapping between Merlin dtypes and the dtypes of one external framework
    """

    def __init__(self, mapping=None, base_class=None, translator=None):
        self.from_merlin_ = mapping or {}
        self.to_merlin_ = {}
        self.base_class = base_class
        self.translator = translator

        for key, values in self.from_merlin_.items():
            if not isinstance(values, list):
                values = [values]
            for value in values:
                self.to_merlin_[value] = key

    def matches_external(self, external_dtype) -> bool:
        """
        Check if this mapping can translate the supplied external dtype

        Parameters
        ----------
        external_dtype : Any
            An external framework dtype

        Returns
        -------
        bool
            True if the external dtype can be translated
        """
        return self._matches(external_dtype, self.to_merlin_, self.base_class)

    def matches_merlin(self, merlin_dtype) -> bool:
        """
        Check if this mapping can translate the supplied Merlin dtype

        Parameters
        ----------
        merlin_dtype : DType
            A Merlin DType object

        Returns
        -------
        bool
            True if the Merlin dtype can be translated
        """
        return self._matches(merlin_dtype, self.from_merlin_)

    def to_merlin(self, external_dtype):
        """
        Translate an external dtype to a Merlin dtype

        Parameters
        ----------
        external_dtype : Any
            An external framework dtype

        Returns
        -------
        DType
            A Merlin DType object
        """
        merlin_dtype = self.to_merlin_[external_dtype]
        return merlin_dtype

    def from_merlin(self, merlin_dtype):
        """
        Translate a Merlin dtype to an external dtype

        Parameters
        ----------
        merlin_dtype : DType
            A Merlin DType object

        Returns
        -------
        Any
            An external framework dtype
        """
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
        try:
            return dtype in mapping.keys()
        except TypeError:
            return type(dtype) in mapping.keys()
