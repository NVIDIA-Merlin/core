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
import functools
import itertools
import operator
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Type

import merlin.dtypes as md
from merlin.dispatch.lazy import lazy_singledispatch
from merlin.dtypes import DType, Shape


class Device(Enum):
    CPU = 0
    GPU = 1


@lazy_singledispatch
def create_tensor_column(values, *args, offsets=None, _unsafe=False, **kwargs):
    """
    Create the appropriate TensorColumn subclass from the type of the supplied values and offsets
    """
    raise NotImplementedError


class TensorColumn:
    """
    A simple wrapper around an array of values and an optional array of offsets

    Should always contain arrays or tensors, not dataframe series
    """

    @classmethod
    def array_type(cls) -> Type:
        """
        The type of the arrays backing this column
        """
        raise NotImplementedError

    @classmethod
    def supported_devices(cls) -> List[Device]:
        """
        List of device types supported by this column type
        """
        raise NotImplementedError

    def __new__(cls, values, *args, offsets=None, **kwargs):
        if cls == TensorColumn:
            return create_tensor_column(values, *args, offsets=offsets, **kwargs)
        else:
            return object.__new__(cls)

    def __init__(
        self, values: Any, offsets: Any = None, dtype=None, _ref=None, _device=None, _unsafe=False
    ):
        values, offsets = self._convert_nested_lists(values, offsets)
        if _ref and _ref.values.shape != values.shape:
            values = self._reshape_values(values, _ref.values.shape)

        if not _unsafe:
            self._validate_values_offsets(values, offsets)

        self._values = values
        self._offsets = offsets
        self._dtype = dtype or values.dtype
        self._ref = _ref
        self._device = _device
        self._shape = None

    def cpu(self):
        """
        Move this column's data to host (i.e. CPU) memory

        Should be overridden by TensorColumn sub-classes with an appropriate implementation
        for their individual frameworks.

        Raises
        ------
        NotImplementedError
            If a sub-class doesn't provide an implementation
        """
        raise NotImplementedError

    def gpu(self):
        """
        Move this column's data to device (i.e. GPU) memory

        Should be overridden by TensorColumn sub-classes with an appropriate implementation
        for their individual frameworks.

        Raises
        ------
        NotImplementedError
            If a sub-class doesn't provide an implementation
        """
        raise NotImplementedError

    @property
    def shape(self) -> Shape:
        if not self._shape:
            self._shape = self._construct_shape(self.values, self.offsets)
        return self._shape

    @property
    def is_list(self) -> Shape:
        return self.shape.is_list

    @property
    def is_ragged(self) -> Shape:
        return self.shape.is_ragged

    @property
    def is_fixed(self):
        return self.shape.is_fixed

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
        if not isinstance(self._dtype, DType):
            self._dtype = md.dtype(self._dtype).with_shape(self.shape)
        return self._dtype

    @property
    def _flatten_values(self):
        raise NotImplementedError

    def _reshape_values(self, values, shape):
        raise NotImplementedError

    def __len__(self):
        if self.offsets is not None:
            return len(self.offsets) - 1
        else:
            return len(self.values)

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
        if not isinstance(other, type(self)):
            return False
        return (
            _arrays_eq(self._values, other._values)
            and _arrays_eq(self._offsets, other._offsets)
            and self.dtype == other.dtype
            and self.device == other.device
        )

    def _construct_shape(self, values, offsets):
        if offsets is not None:
            num_rows = len(offsets) - 1
            row_lengths = offsets[1:] - offsets[:-1]
            num_cols = int(row_lengths[0]) if all(row_lengths == row_lengths[0]) else None
            shape = [num_rows, num_cols]
            if len(values.shape) > 1:
                embedding_shape = values.shape[1:]
                shape.extend(embedding_shape)
            shape = Shape(tuple(shape))
        else:
            shape = Shape(values.shape)
        return shape

    def _convert_nested_lists(self, values, offsets):
        if not isinstance(values, list):
            return values, offsets

        if offsets is not None:
            raise ValueError(
                "Providing nested values is not supported with offsets. "
                "Either provide flattened values with offsets, or nested "
                "values without offsets."
            )

        flat_values = functools.reduce(operator.iconcat, values, [])
        flat_offsets = list(itertools.accumulate([0] + [len(val) for val in values]))

        constructor = self.__class__.array_constructor()
        return constructor(flat_values), constructor(flat_offsets)

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
class _TransferColumn:
    values: Any
    offsets: Any
    ref: TensorColumn


@dataclass(frozen=True)
class _DlpackColumn(_TransferColumn):
    pass


def _arrays_eq(array1, array2):
    if array1 is None and array2 is None:
        return True

    if array1 is None or array2 is None:
        return False

    return len(array1) == len(array2) and all(array1 == array2)
