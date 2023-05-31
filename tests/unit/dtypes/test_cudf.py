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
import pytest

import merlin.dtypes as md
from merlin.core.compat import cudf


@pytest.mark.skipif(not cudf, reason="CUDF is required to test its dtypes")
def test_cudf_struct_dtype():
    struct_dtype = cudf.StructDtype({"a": "int64", "b": "string"})
    merlin_dtype = md.dtype(struct_dtype)
    assert merlin_dtype == md.struct

    merlin_dtype = md.struct
    cudf_dtype = merlin_dtype.to("cudf")
    assert cudf_dtype == cudf.StructDtype
