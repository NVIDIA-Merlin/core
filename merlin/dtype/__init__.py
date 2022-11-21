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

import sys
from copy import copy
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

import numpy as np


class ElementType(Enum):
    Bool = "bool"
    Int = "int"
    UInt = "uint"
    Float = "float"
    String = "string"
    DateTime = "datetime"
    Object = "object"


@dataclass
class DType:
    name: str
    # TODO: Rename elemsize to bits or bytes for clarity
    elemtype: ElementType
    elemsize: Optional[int] = None
    signed: Optional[bool] = None
    shape: Optional[Tuple] = None

    @property
    def is_list(self):
        return self.shape is not None and len(self.shape) > 1

    @property
    def is_ragged(self):
        return self.is_list and None in self.shape


int32 = DType("int32", ElementType.Int, 32, signed=True)
int64 = DType("int64", ElementType.Int, 64, signed=True)
uint32 = DType("uint32", ElementType.UInt, 32)
uint64 = DType("uint64", ElementType.UInt, 64)
float32 = DType("float32", ElementType.Float, 32, signed=True)
float64 = DType("float64", ElementType.Float, 64, signed=True)
datetime64us = DType("datetime64[us]", ElementType.DateTime, 64)
datetime64ns = DType("datetime64[ns]", ElementType.DateTime, 64)
string = DType("str", ElementType.String)
boolean = DType("bool", ElementType.Bool)
object_ = DType("object", ElementType.Object)

_mapping_registry = []


# Is there ever a case where we'd want to preempt the built-in mappings?
def register(mapping: Dict[str, DType]):
    _mapping_registry.append(mapping)


# Make these mappings immutable?
python_dtypes = {int: int64, float: float64, str: string}
register(python_dtypes)

numpy_dtypes = {
    np.int32: int32,
    np.dtype("int32"): int32,
    np.int64: int64,
    np.dtype("int64"): int64,
    np.float32: float32,
    np.dtype("float32"): float32,
    np.float64: float64,
    np.dtype("float64"): float64,
    np.datetime64: datetime64ns,
    np.dtype("datetime64[ns]"): datetime64ns,
    np.dtype("datetime64[us]"): datetime64us,
    np.str: string,
    np.dtype("O"): object_,
}
register(numpy_dtypes)


# This class implements the "call" method for the *module*, which
# allows us to use both `dtype(value)` and `dtype.int32` syntaxes,
# even though we can't directly add a callable to the `merlin`
# namespace (since it's implicit and doesn't allow an `__init__.py`
# file)
class DTypeModule(ModuleType):
    def __call__(self, value: Any, shape: Optional[Tuple] = None):
        if isinstance(value, DType):
            return value
        for mapping in _mapping_registry:
            try:
                if value in mapping:
                    merlin_type = copy(mapping[value])
                    if shape is not None:
                        merlin_type.shape = shape
                    return merlin_type
            except TypeError:
                pass

        raise TypeError(
            f"Merlin doesn't have a mapping from {value} to a Merlin dtype. "
            "If you'd like to provide one, you can use `merlin.dtype.register()`."
        )


sys.modules[__name__].__class__ = DTypeModule
