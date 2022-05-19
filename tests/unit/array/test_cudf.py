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

import cudf
import cupy as cp

from merlin.array.cudf import MerlinCudfArray
from merlin.array.cupy import MerlinCupyArray


def test_cudf_merlinarray_to_cupy_merlinarray():

    cudf_series = cudf.Series([1, 2, 3, 4])

    cudf_array = MerlinCudfArray(cudf_series)
    cp_array = cudf_array.to(MerlinCupyArray)
    assert isinstance(cp_array.data, cp.ndarray)
    assert (cudf_array.data.to_numpy() == cp.asnumpy(cp_array.data)).all()
