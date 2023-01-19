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
import pytest

import merlin.dtypes as md
from merlin.dtypes.shape import Dimension, Shape

# Dimension


def test_empty_dimension():
    dim = Dimension()
    assert dim.min == 0
    assert dim.max is None


def test_min_max_val_dimension():
    dim = Dimension(2, 3)
    assert dim.min == 2
    assert dim.max == 3


def test_fixed_min_with_unbounded_max():
    dim = Dimension(2)
    assert dim.min == 2
    assert dim.max is None

    dim = Dimension(2, None)
    assert dim.min == 2
    assert dim.max is None


def test_min_is_none_raises_error():
    with pytest.raises(ValueError):
        Dimension(None)

    with pytest.raises(ValueError):
        Dimension(None, 1)


def test_bounds_must_be_non_negative():
    with pytest.raises(ValueError):
        Dimension(-1, 2)

    with pytest.raises(ValueError):
        Dimension(2, -1)


def test_max_less_than_min():
    with pytest.raises(ValueError):
        Dimension(2, 1)


def test_is_bounded():
    dim = Dimension()
    assert dim.is_bounded is False

    dim = Dimension(2)
    assert dim.is_bounded is False

    dim = Dimension(2, 2)
    assert dim.is_bounded is True

    dim = Dimension(2, 4)
    assert dim.is_bounded is True

    dim = Dimension(2, None)
    assert dim.is_bounded is False


def test_is_fixed():
    dim = Dimension()
    assert dim.is_fixed is False

    dim = Dimension(2)
    assert dim.is_fixed is False

    dim = Dimension(2, 2)
    assert dim.is_fixed is True

    dim = Dimension(2, 4)
    assert dim.is_fixed is False

    dim = Dimension(2, None)
    assert dim.is_fixed is False


def test_is_variable():
    dim = Dimension()
    assert dim.is_variable is True

    dim = Dimension(2)
    assert dim.is_variable is True

    dim = Dimension(2, 2)
    assert dim.is_variable is False

    dim = Dimension(2, 4)
    assert dim.is_variable is True

    dim = Dimension(2, None)
    assert dim.is_variable is True


# Shape


def test_shape_without_args_represents_unknown():
    shape = Shape()
    assert shape.dims is None
    assert shape.is_list is False
    assert shape.is_ragged is False


def test_shape_with_empty_tuple_represents_scalar():
    shape = Shape(())
    assert shape.dims == ()
    assert shape.is_list is False
    assert shape.is_ragged is False


def test_flat_tuple_creates_fixed_shape():
    shape = Shape((1, 2, 3))
    assert shape.is_fixed is True


def test_nested_tuple_creates_variable_shape():
    shape = Shape(((5, 5), (2, 2), (3, 3)))
    assert shape.is_variable is False

    shape = Shape(((1, 3), (2, 2), (3, 4)))
    assert shape.is_variable is True

    shape = Shape(((1, 3), (2, 4), (4, 4)))
    assert shape.is_variable is True

    shape = Shape(((1, 3), (2, 4), (4, 7)))
    assert shape.is_variable is True


def test_mixed_tuple_creates_variable_shape():
    shape = Shape((5, (2, 3), 4))
    assert shape.is_variable is True


def test_nested_tuple_error():
    with pytest.raises(ValueError):
        Shape((5, (2, None), (4, 5, 6)))

    with pytest.raises(ValueError):
        Shape((5.3, (2, None), (4, 6)))

    with pytest.raises(ValueError):
        Shape(("asdf", (2, None), (4, 6)))


def test_shape_properties():
    shape = Shape((5,))
    assert shape.is_fixed is True
    assert shape.is_variable is False
    assert shape.is_bounded is True
    assert shape.is_ragged is False
    assert shape.is_list is False

    shape = Shape((5, 1))
    assert shape.is_fixed is True
    assert shape.is_variable is False
    assert shape.is_bounded is True
    assert shape.is_ragged is False
    assert shape.is_list is True

    shape = Shape(((5, None), 2, 4))
    assert shape.is_fixed is False
    assert shape.is_variable is True
    assert shape.is_bounded is False
    assert shape.is_ragged is False
    assert shape.is_list is True

    shape = Shape((5, (2, None), 4))
    assert shape.is_fixed is False
    assert shape.is_variable is True
    assert shape.is_bounded is False
    assert shape.is_ragged is True
    assert shape.is_list is True

    shape = Shape((5, 2, (4, None)))
    assert shape.is_fixed is False
    assert shape.is_variable is True
    assert shape.is_bounded is False
    assert shape.is_ragged is True
    assert shape.is_list is True


# DType


def test_dtype_has_a_shape():
    assert md.int32.shape == Shape()


def test_dtype_with_shape():
    dtype = md.int32.with_shape((3, 4, 5))
    assert dtype.shape != (3, 4, 5)
    assert dtype.shape == Shape((3, 4, 5))
