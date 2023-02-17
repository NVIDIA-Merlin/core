#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import merlin.dtypes as md


class Device(Enum):
    CPU = 0
    GPU = 1


# This should always contains arrays or tensors, not series
class TensorColumn(ABC):
    """
    A simple wrapper around an array of values and an optional array of offsets
    """

    @classmethod
    @abstractmethod
    def array_type(cls):
        return None

    @classmethod
    @abstractmethod
    def supported_devices(cls):
        return []

    def __init__(self, values: Any, offsets: Any = None, dtype=None, _ref=None, _device=None):
        self.values = values
        self.offsets = offsets
        self.dtype = md.dtype(dtype or values.dtype)
        self._ref = _ref
        self._device = _device

    @property
    def device(self) -> Device:
        return self._device


@dataclass(frozen=True)
class TransferColumn:
    values: Any
    offsets: Any
    ref: TensorColumn


@dataclass(frozen=True)
class DlpackColumn(TransferColumn):
    pass


@dataclass(frozen=True)
class CudaArrayColumn(TransferColumn):
    pass


@dataclass(frozen=True)
class ArrayColumn(TransferColumn):
    pass
