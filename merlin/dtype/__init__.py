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
import dataclasses
import sys
from types import ModuleType
from typing import Any, Optional, Tuple

from merlin.dtype.dtypes import *
from merlin.dtype.dtypes import DType
from merlin.dtype.mappings import _dtype_registry

# Convenience alias for this method
register = _dtype_registry.register


# This class implements the "call" method for the *module*, which
# allows us to use both `dtype(value)` and `dtype.int32` syntaxes,
# even though we can't directly add a callable to the `merlin`
# namespace (since it's implicit and doesn't allow an `__init__.py`
# file)
class DTypeModule(ModuleType):
    def __call__(self, value: Any, shape: Optional[Tuple] = None):
        if isinstance(value, DType):
            return value

        for _, mapping in _dtype_registry.mappings.items():
            if value in mapping.to_merlin:
                return dataclasses.replace(mapping.to_merlin[value], **{"shape": shape})

        raise TypeError(
            f"Merlin doesn't have a mapping from {value} to a Merlin dtype. "
            "If you'd like to provide one, you can use `merlin.dtype.register()`."
        )


sys.modules[__name__].__class__ = DTypeModule
