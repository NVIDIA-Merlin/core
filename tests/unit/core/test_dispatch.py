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
import numpy as np
import pytest

from merlin.core.dispatch import HAS_GPU, is_list_dtype, list_val_dtype, make_df

if HAS_GPU:
    _CPU = [True, False]
else:
    _CPU = [True]


@pytest.mark.parametrize("cpu", _CPU)
def test_list_dtypes(tmpdir, cpu):
    df = make_df(device="cpu" if cpu else "gpu")
    df["vals"] = [
        [[0, 1, 2], [3, 4], [5]],
    ]

    assert is_list_dtype(df["vals"])
    assert list_val_dtype(df["vals"]) == np.dtype(np.int64)
