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
from abc import ABC, abstractproperty
from enum import Enum
from typing import Any

import merlin.dtypes as md
from merlin.core.protocols import SeriesLike


class Device(Enum):
    CPU = 0
    GPU = 1


# This should always contains arrays or tensors, not series
class TensorColumn(ABC):
    """
    A simple wrapper around an array of values
    """

    def __init__(self, values: Any, offsets: Any = None, dtype=None):
        super().__init__()

        self.values = values
        self.offsets = offsets
        self.dtype = md.dtype(dtype or values.dtype)

    @abstractproperty
    def device(self) -> Device:
        ...
