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

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Dimension:
    """
    The range of potential sizes for a single dimension of a field or column
    """

    min: int = 0
    max: Optional[int] = None

    def __post_init__(self):
        if self.min is None:
            raise ValueError("The minimum size of a dimension cannot be None. ")

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
        return self.max is not None

    @property
    def is_fixed(self):
        return self.is_bounded and self.min == self.max

    @property
    def is_variable(self):
        return not self.is_fixed


@dataclass(frozen=True)
class Shape:
    """
    The range of potential sizes for all the dimensions of a field or column
    """

    dims: Optional[Tuple] = None

    def __post_init__(self):
        if self.dims is not None:
            new_dims = []
            for i, dim in enumerate(self.dims):
                if isinstance(dim, Dimension):
                    new_dim = dim
                elif isinstance(dim, tuple) and len(dim) == 2:
                    new_dim = Dimension(dim[0], dim[1])
                elif isinstance(dim, int):
                    new_dim = Dimension(dim, dim)
                else:
                    raise ValueError(
                        f"Invalid shape tuple format: {self.dims}. Each dimension is expected "
                        " to be either a single integer or a length 2 tuple."
                    )
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
    def is_variable(self):
        return not self.is_fixed

    @property
    def is_list(self):
        return self.dims is not None and len(self.dims) > 1

    @property
    def is_ragged(self):
        return self.is_list and any(dim.min != dim.max for dim in self.dims[1:])
