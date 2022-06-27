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

from merlin.features.array.numpy import MerlinNumpyArray
from merlin.features.compat import numpy

pytest.importorskip("numpy")


def test_merlin_array_keeps_data_in_memory():
    np_array = numpy.array([1, 2, 3, 4, 5])
    array1 = MerlinNumpyArray(np_array)
    array2 = MerlinNumpyArray(array1)
    del array1
    assert numpy.array_equal(array2.array, np_array)
