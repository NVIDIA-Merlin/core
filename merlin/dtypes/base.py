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
from typing import Optional, Tuple, Union

from merlin.dtypes.registry import _dtype_registry
from merlin.dtypes.shape import Shape


class ElementType(Enum):
    """
    Merlin DType base types

    Since a Merlin DType may describe a list, these are the either the types of
    scalars or the types of list elements.
    """

    Bool = "bool"
    Int = "int"
    UInt = "uint"
    Float = "float"
    String = "string"
    DateTime = "datetime"
    Object = "object"
    Unknown = "unknown"
    Struct = "struct"


class ElementUnit(Enum):
    """
    Dtype units, used only for datetime types

    Since a Merlin DType may describe a list, these are the either the units of
    scalars or the units of list elements.
    """

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
    """
    Merlin dtypes are objects of this dataclass
    """

    name: str
    element_type: ElementType
    element_size: Optional[int] = None
    element_unit: Optional[ElementUnit] = None
    signed: Optional[bool] = None
    shape: Optional[Shape] = None

    def __post_init__(self):
        if not self.shape:
            object.__setattr__(self, "shape", Shape())

    def to(self, mapping_name: str):
        """
        Convert this Merlin dtype to another framework's dtypes

        Parameters
        ----------
        mapping_name : str
            Name of the framework dtype mapping to apply

        Returns
        -------
        Any
            An external framework dtype object

        Raises
        ------
        ValueError
            If there is no registered mapping for the given framework name
        ValueError
            The registered mapping for the given framework name doesn't map
            this Merlin dtype to a framework dtype
        """
        try:
            mapping = _dtype_registry.mappings[mapping_name]
        except KeyError as exc:
            raise ValueError(
                f"Merlin doesn't have a registered dtype mapping for '{mapping_name}'. "
                "If you'd like to register a new dtype mapping, use `merlin.dtype.register()`. "
                "If you're expecting this mapping to already exist, has the library or package "
                "that defines the mapping been imported successfully?"
            ) from exc

        try:
            return mapping.from_merlin(self.without_shape)
        except KeyError as exc:
            raise ValueError(
                f"The registered dtype mapping for {mapping_name} doesn't contain type {self.name}."
            ) from exc

    @property
    def to_numpy(self):
        return self.to("numpy")

    @property
    def to_python(self):
        return self.to("python")

    # These properties refer to a single scalar (potentially a list element)
    @property
    def is_integer(self):
        return self.element_type.value == "int"

    @property
    def is_float(self):
        return self.element_type.value == "float"

    def with_shape(self, shape: Union[Tuple, Shape]):
        """
        Create a copy of this dtype with a new shape

        Parameters
        ----------
        shape : Union[Tuple, Shape]
            Object to set as shape of dtype, must be either a tuple or Shape.

        Returns
        -------
        DType
            A copy of this dtype containing the provided shape value

        Raises
        ------
        TypeError
            If value is not either a tuple or a Shape
        """
        if isinstance(shape, tuple):
            shape = Shape(shape)

        if not isinstance(shape, Shape):
            raise TypeError(
                f"Provided value {shape} (of type {type(shape)}) for DType.shape property "
                "is not of type Shape."
            )

        return replace(self, shape=shape)

    @property
    def without_shape(self):
        """
        Create a copy of this object without the shape

        Returns
        -------
        DType
            A copy of this object with the shape removed
        """
        if self.shape.dims is None:
            return self

        return replace(self, shape=Shape())
