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

from merlin.dtype import mappings
from merlin.dtype.mappings.numpy import _numpy_dtype
from merlin.dtype.dtypes import *
from merlin.dtype.dtypes import DType

from merlin.dtype.registry import _dtype_registry


# Convenience alias for this method
register = _dtype_registry.register


def _to_merlin(external_dtype):
    # If the supplied dtype is already a Merlin dtype, then there's
    # nothing for us to do and we can exit early
    if isinstance(external_dtype, DType):
        return external_dtype

    return _dtype_registry.to_merlin(external_dtype)


# This class implements the "call" method for the *module*, which
# allows us to use both `dtype(value)` and `dtype.int32` syntaxes,
# even though we can't directly add a callable to the `merlin`
# namespace (since it's implicit and doesn't allow an `__init__.py`
# file)
class DTypeModule(ModuleType):
    def __call__(self, external_dtype: Any, shape: Optional[Tuple] = None):
        # We can't raise an error when the supplied dtype is None, because
        # that will break when we load the module, so instead return None
        # which will either:
        # - work fine if the dtype is None because it isn't really being used
        # - surface issues downstream when something else tries to use it
        if external_dtype is None:
            return None

        try:
            # First attempt to apply all the registered Merlin dtype mappings
            return _to_merlin(external_dtype)
        except TypeError as base_exc:
            # If we don't find a match, fall back to converting to
            # a numpy dtype and trying to match that
            try:
                return _to_merlin(_numpy_dtype(external_dtype))
            except TypeError:
                ...

            raise base_exc


# We promise that the class defined above is actually a module
dtype = sys.modules[__name__]
dtype.__class__ = DTypeModule


