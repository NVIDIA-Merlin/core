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

from merlin.core.dispatch import make_df
from merlin.dag.dictarray import DataFrameLike, DictArray, DictLike, Transformable


@pytest.mark.parametrize("protocol", [DictLike])
def test_dictionary_is_dictlike(protocol):
    obj = {}

    assert isinstance(obj, protocol)


@pytest.mark.parametrize("device", [None, "cpu"])
@pytest.mark.parametrize("protocol", [DictLike, DataFrameLike, Transformable])
def test_dataframes_match_protocols(protocol, device):
    obj = make_df({}, device=device)

    assert isinstance(obj, DictLike)


@pytest.mark.parametrize("device", [None, "cpu"])
def test_dataframes_are_dataframelike(device):
    obj = make_df({}, device=device)

    assert isinstance(obj, DataFrameLike)


@pytest.mark.parametrize("protocol", [DictLike, DataFrameLike, Transformable])
def test_dictarray_matches_protocols(protocol):
    obj = DictArray({}, {})

    assert isinstance(obj, protocol)
