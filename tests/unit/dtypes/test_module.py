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
import numpy
import pytest

from merlin import dtype

###############################
#        Registration         #
###############################


def test_type_mappings_can_be_registered():
    class TestType:
        pass

    test_type = dtype.DType("test", dtype.ElementType.Int, 4096, signed=True)

    dtype.register("test", {test_type: TestType})
    merlin_dtype = dtype(TestType)
    assert merlin_dtype == test_type


###############################
#        Conversions          #
###############################


@pytest.mark.parametrize("python_type, merlin_type", [(int, dtype.int64)])
def test_python_types_convert_correctly(python_type, merlin_type):
    assert dtype(python_type) == merlin_type


@pytest.mark.parametrize("numpy_type, merlin_type", [(numpy.int64, dtype.int64)])
def test_numpy_types_convert_correctly(numpy_type, merlin_type):
    assert dtype(numpy_type) == merlin_type


###############################
#         Type Errors         #
###############################


def test_returns_none_dtype_for_none_input():
    result = dtype(None)
    assert result is None


def test_unknown_types_raise_error():
    class UnknownType:
        pass

    with pytest.raises(TypeError):
        dtype(UnknownType)


###############################
#           Shapes            #
###############################


def test_dtypes_can_hold_shapes():
    shape = (10, 10)
    merlin_type = dtype(int, shape)
    assert merlin_type.shape == shape


@pytest.mark.parametrize("shape, is_list", [((10, 10), True), ((10,), False), (None, False)])
def test_dtypes_know_if_theyre_lists(shape, is_list):
    merlin_type = dtype(int, shape)
    assert merlin_type.is_list == is_list


@pytest.mark.parametrize(
    "shape, is_ragged",
    [((10, None), True), ((None, 10), False), ((10, 10), False), ((10,), False), (None, False)],
)
def test_dtypes_know_if_theyre_ragged(shape, is_ragged):
    merlin_type = dtype(int, shape)
    assert merlin_type.is_ragged == is_ragged
