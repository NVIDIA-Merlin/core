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

from merlin.core.protocols import DataFrameLike, DictLike, SeriesLike, Transformable
from merlin.dag.dictarray import Column, DictArray


@pytest.mark.parametrize("protocol", [SeriesLike])
def test_column_matches_protocols(protocol):
    obj = Column([], np.int32)

    assert isinstance(obj, protocol)


@pytest.mark.parametrize("protocol", [DictLike, DataFrameLike, Transformable])
def test_dictarray_matches_protocols(protocol):
    obj = DictArray({}, {})

    assert isinstance(obj, protocol)
