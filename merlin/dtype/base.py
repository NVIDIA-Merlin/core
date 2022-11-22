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


@dataclass(eq=True, frozen=True)
class DType:
    name: str
    elemtype: ElementType
    elemsize: Optional[int] = None
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
        mapping = _dtype_registry.mappings[mapping_name]

        # Ignore the shape when matching dtypes
        dtype = replace(dtype, **{"shape": None})

        # Always translate to the first external dtype in the list
        return mapping.from_merlin[dtype][0]
