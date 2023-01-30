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

import merlin.dtypes as md


@pytest.mark.parametrize("python_type, merlin_type", [(int, md.int64)])
def test_python_types_convert_correctly(python_type, merlin_type):
    assert md.dtype(python_type) == merlin_type


@pytest.mark.parametrize("numpy_type, merlin_type", [(numpy.int64, md.int64)])
def test_numpy_types_convert_correctly(numpy_type, merlin_type):
    assert md.dtype(numpy_type) == merlin_type


def test_type_mappings_can_be_registered():
    class TestType:
        pass

    test_type = md.DType("test", md.ElementType.Int, 4096, signed=True)

    md.register("test", {test_type: TestType})
    merlin_dtype = md.dtype(TestType)
    assert merlin_dtype == test_type


def test_unknown_types_raise_error():
    class UnknownType:
        pass

    with pytest.raises(TypeError):
        md.dtype(UnknownType)
