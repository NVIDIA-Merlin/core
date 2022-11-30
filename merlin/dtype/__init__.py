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

# flake8: noqa
import sys
from types import ModuleType
from typing import Any, Optional, Tuple

import numpy as np

from merlin.dtype import mappings
from merlin.dtype.dtypes import *
from merlin.dtype.dtypes import DType

from merlin.dtype.registry import _dtype_registry


# Convenience alias for this method
register = _dtype_registry.register


# This class implements the "call" method for the *module*, which
# allows us to use both `dtype(value)` and `dtype.int32` syntaxes,
# even though we can't directly add a callable to the `merlin`
# namespace (since it's implicit and doesn't allow an `__init__.py`
# file)
class DTypeModule(ModuleType):
    def __call__(self, external_dtype: Any, shape: Optional[Tuple] = None):
        # If the supplied dtype is already a Merlin dtype, then there's
        # nothing for us to do and we can exit early
        if isinstance(external_dtype, DType):
            return external_dtype

        # We can't raise an error when the supplied dtype is None, because
        # that will break when we load the module, so instead return None
        # which will either work fine if the dtype is None because it isn't
        # really being used, or surface issues downstream when something tries
        # to use it
        if external_dtype is None:
            return None

        try:
            # First attempt to apply all the registered dtype mappings
            return _dtype_registry.to_merlin(external_dtype)

        except TypeError as base_exc:
            # If we don't find a match, fall back to converting to
            # a numpy dtype and trying to match that
            try:
                numpy_dtype = _numpy_dtype(external_dtype)
                return _dtype_registry.to_merlin(numpy_dtype)
            except TypeError:
                ...

            raise base_exc


# We promise that the class defined above is actually a module
dtype = sys.modules[__name__]
dtype.__class__ = DTypeModule


def _numpy_dtype(raw_dtype):
    # Many Pandas dtypes have equivalent numpy dtypes
    if hasattr(raw_dtype, "numpy_dtype"):
        return np.dtype(raw_dtype.numpy_dtype)
    # cuDF categorical columns have varying element types
    elif hasattr(raw_dtype, "_categories"):
        return raw_dtype._categories.dtype
    # Rely on Numpy to do conversions from strings to dtypes (for now)
    elif isinstance(raw_dtype, str):
        return np.dtype(raw_dtype)
    # Tensorflow dtypes can convert themselves (in case we missed a mapping)
    elif hasattr(raw_dtype, "as_numpy_dtype"):
        return raw_dtype.as_numpy_dtype
