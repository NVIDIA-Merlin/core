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

from merlin.dtypes import mappings
from merlin.dtypes.aliases import *
from merlin.dtypes.base import DType
from merlin.dtypes.mappings.numpy import _numpy_dtype
from merlin.dtypes.registry import _dtype_registry

# Convenience alias for registering dtypes
register = _dtype_registry.register


def dtype(external_dtype):
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

        # If we fail to find a match even after we try converting to
        # numpy, re-raise the original exception since it has more
        # information about the original external dtype that's causing
        # the problem. (We want to highlight that one, not whatever
        # numpy dtype it was converted to.)
        raise base_exc


def _to_merlin(external_dtype):
    # If the supplied dtype is already a Merlin dtype, then there's
    # nothing for us to do and we can exit early
    if isinstance(external_dtype, DType):
        return external_dtype

    return _dtype_registry.to_merlin(external_dtype)
