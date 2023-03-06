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


class DefaultShapes(Enum):
    LIST = (-1, None)
    SCALAR = (-1,)


def Dimension(size=None):
    """Create a dimension from a size.

    A size can be one of:
      - None    : a ragged dimension of unknown size
      - int     : a fixed dimension of some size (-1 = unknown)
      - 2-tuple : the bounds of a ragged dimension (fixed if min == max)
    """
    if isinstance(size, (FixedDimension, RaggedDimension)):
        return size
    elif isinstance(size, tuple) and len(size) == 2:
        if size[0] == size[1]:
            return FixedDimension(size[0], size[1])
        return RaggedDimension(size[0], size[1])
    elif isinstance(size, int):
        if size == -1:
            return FixedDimension()
        return FixedDimension(size, size)
    elif size is None:
        return RaggedDimension()
    else:
        raise ValueError(
            f"Invalid dimension format: {size}. Each dimension is expected "
            " to be None, a single integer, or a tuple with length 2."
        )


@dataclass(frozen=True)
class BaseDimension:
    """
    The range of potential sizes for a single dimension of a field or column
    """

    min: int = 0
    max: Optional[int] = None

    def __post_init__(self):
        if self.min is None:
            raise ValueError("The minimum size of a dimension cannot be None. ")

        if not isinstance(self.min, int):
            raise ValueError("The minimmum size must be an integer. " f"Provided min: {self.min}")

        if self.max and not isinstance(self.max, int):
            raise ValueError("The maximum size must be an integer. " f"Provided max: {self.max}")

        if self.min < 0:
            raise ValueError(
                "The minimum size of a dimension must be non-negative. " f"Provided min: {self.min}"
            )

        if self.max and self.max < 0:
            raise ValueError(
                "The maximum size of a dimension must be at least one. " f"Provided max: {self.max}"
            )

        if self.max and self.max < self.min:
            raise ValueError(
                "The maximum size of a dimension must be at least as large as the minimum size. "
                f"Provided min: {self.min} max: {self.max}"
            )

    @property
    def is_bounded(self):
        """Is the dimension bounded in size?"""
        return self.max is not None

    @property
    def is_uniform(self):
        """Is the dimension uniform in size?"""
        return self.is_bounded and self.min == self.max

    @property
    def is_variable(self):
        """Can the size of the dimension vary between instances of tensors."""
        return not self.is_uniform

    @property
    def is_unknown(self):
        return self.min == 0 and self.max is None

    def with_min(self, value):
        return replace(self, min=value)

    def with_max(self, value):
        return replace(self, max=value)


class RaggedDimension(BaseDimension):
    @property
    def is_fixed(self):
        return False

    @property
    def size(self):
        return None


class FixedDimension(BaseDimension):
    @property
    def is_fixed(self):
        return True

    @property
    def size(self):
        if self.is_uniform:
            return self.max
        else:
            return -1


@dataclass(frozen=True)
class Shape:
    """
    The range of potential sizes for all the dimensions of a field or column
    """

    dims: Optional[Union[Tuple, "Shape"]] = None

    def __post_init__(self):
        if isinstance(self.dims, DefaultShapes):
            object.__setattr__(self, "dims", self.dims.value)

        if isinstance(self.dims, Shape):
            object.__setattr__(self, "dims", self.dims.dims)

        if self.dims is not None:
            new_dims = []
            for i, dim in enumerate(self.dims):
                new_dim = Dimension(dim)
                new_dims.append(new_dim)

            object.__setattr__(self, "dims", tuple(new_dims))

    def __eq__(self, other):
        """
        Make `dims is None` a wildcard when determining equality

        This definition of equality allows an unknown shape with `dims is None` to be
        considered equal or compatible with a known shape with `dims is not None`.
        """
        if not isinstance(other, Shape):
            return False

        if self.dims is None or other.dims is None:
            return True

        return self.dims == other.dims

    def __getitem__(self, idx):
        return self.dims[idx]

    def __iter__(self):
        return self.dims

    def with_dim(self, index, value):
        new_dims = list(self.dims)
        new_dims[index] = value
        return replace(self, dims=tuple(new_dims))

    def with_dim_min(self, index, value):
        return self.with_dim(index, self.dims[index].with_min(value))

    def with_dim_max(self, index, value):
        return self.with_dim(index, self.dims[index].with_max(value))

    @property
    def min(self) -> Tuple:
        return tuple(dim.min for dim in self.dims)

    @property
    def max(self) -> Tuple:
        return tuple(dim.max for dim in self.dims)

    @property
    def is_bounded(self):
        return all(dim.is_bounded for dim in self.dims)

    @property
    def is_fixed(self):
        return all(dim.is_fixed for dim in self.dims)

    @property
    def is_uniform(self):
        return all(dim.is_uniform for dim in self.dims)

    @property
    def is_variable(self):
        return not self.is_uniform

    @property
    def is_list(self):
        return self.dims is not None and len(self.dims) > 1

    @property
    def is_ragged(self):
        return self.is_list and any(not dim.is_fixed for dim in self.dims[1:])

    @property
    def as_tuple(self):
        return (
            tuple((dim.size if dim.is_fixed else (dim.min, dim.max) for dim in self.dims))
            if self.dims
            else None
        )

    @property
    def is_unknown(self):
        return self.dims is None
