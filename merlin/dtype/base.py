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

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Tuple

from merlin.dtype.registry import _dtype_registry


class ElementType(Enum):
    Bool = "bool"
    Int = "int"
    UInt = "uint"
    Float = "float"
    String = "string"
    DateTime = "datetime"
    Object = "object"


class ElementUnit(Enum):
    Year = "year"
    Month = "month"
    Day = "day"
    Hour = "hour"
    Minute = "minute"
    Second = "second"
    Millisecond = "millisecond"
    Microsecond = "microsecond"
    Nanosecond = "nanosecond"


@dataclass(eq=True, frozen=True)
class DType:
    name: str
    elemtype: ElementType
    elemsize: Optional[int] = None
    elemunit: Optional[ElementUnit] = None
    signed: Optional[bool] = None
    shape: Optional[Tuple] = None

    # These properties refer to what's in a single row of the DataFrame/DictArray
    @property
    def is_list(self):
        return self.shape is not None and len(self.shape) > 1

    @property
    def is_ragged(self):
        return self.is_list and None in self.shape[1:]

    def to(self, mapping_name):
        try:
            mapping = _dtype_registry.mappings[mapping_name]
        except KeyError:
            raise ValueError(
                f"Merlin doesn't have a registered dtype mapping for '{mapping_name}'. "
                "If you'd like to register a new dtype mapping, use `merlin.dtype.register()`. "
                "If you're expecting this mapping to already exist, has the library or package "
                "that defines the mapping been imported successfully?"
            )

        # Ignore the shape when matching dtypes
        dtype = replace(self, **{"shape": None})

        # Always translate to the first external dtype in the list
        try:
            return mapping.from_merlin[dtype][0]
        except KeyError:
            raise ValueError(
                f"The registered dtype mapping for {mapping_name} doesn't contain type {dtype.name}. "
            )
