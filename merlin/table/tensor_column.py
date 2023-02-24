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
from typing import Any, Type

import merlin.dtypes as md
from merlin.core.protocols import SeriesLike


class Device(Enum):
    CPU = 0
    GPU = 1


# This should always contains arrays or tensors, not series
class TensorColumn(ABC, SeriesLike):
    """
    A simple wrapper around an array of values and an optional array of offsets
    """

    @classmethod
    @abstractmethod
    def array_type(cls) -> Type:
        """
        The type of the arrays backing this column
        """
        return None

    @classmethod
    @abstractmethod
    def supported_devices(cls):
        """
        List of device types supported by this column type
        """
        return []

    def __init__(self, values: Any, offsets: Any = None, dtype=None, _ref=None, _device=None):
        self._validate_values_offsets(values, offsets)

        self._values = values
        self._offsets = offsets

        self._dtype = md.dtype(dtype or values.dtype)
        self._ref = _ref
        self._device = _device

    @property
    def device(self) -> Device:
        return self._device

    @property
    def values(self):
        return self._values

    @property
    def offsets(self):
        return self._offsets

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, index):
        # get correct indexes if offsets exists
        if self._offsets is None:
            index = len(self._values) + index if index < 0 else index
            return self._values[index]
        # There should be a better way to get negative indexing to work
        index = len(self._offsets) - 1 + index if index < 0 else index
        start = self._offsets[index]
        end = self._offsets[index + 1]
        return self._values[start:end]

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return False
        return (
            _arrays_eq(self._values, other._values)
            and _arrays_eq(self._offsets, other._offsets)
            and self.dtype == other.dtype
            and self.device == other.device
        )

    def _validate_values_offsets(self, values, offsets):
        self._raise_type_error("values", values)

        if offsets is not None:
            self._raise_type_error("offsets", offsets)

            # The offsets have to be in increasing order
            # None of the offsets should be greater than the length of the values
            # The last offset should be the length of the values

            if len(values) != offsets[-1]:
                raise ValueError("The last offset must match the length of the values.")

            if any(offsets[1:] - offsets[:-1] < 0):
                raise ValueError("Offsets must be in increasing order")

    def _raise_type_error(self, field_name, array):
        expected_type = self.__class__.array_type()
        if not isinstance(  # pylint:disable=isinstance-second-argument-not-valid-type
            array, expected_type
        ):
            raise TypeError(
                f"{self.__class__} expected arrays of type {expected_type}, "
                f"but received {field_name} of type {type(array)}"
            )


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


def _arrays_eq(array1, array2):
    if array1 is None and array2 is None:
        return True

    if array1 is None or array2 is None:
        return False

    return len(array1) == len(array2) and all(array1 == array2)
