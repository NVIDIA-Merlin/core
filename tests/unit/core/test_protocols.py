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

from merlin.core.compat import HAS_GPU, cudf
from merlin.core.dispatch import make_df, make_series
from merlin.core.protocols import DataFrameLike, DictLike, SeriesLike, Transformable

if HAS_GPU and cudf:
    _DEVICES = ["cpu", None]
else:
    _DEVICES = ["cpu"]


@pytest.mark.parametrize("protocol", [DictLike])
def test_dictionary_is_dictlike(protocol):
    obj = {}

    assert isinstance(obj, protocol)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("protocol", [DictLike, DataFrameLike, Transformable])
def test_dataframes_match_protocols(protocol, device):
    obj = make_df({}, device=device)

    assert isinstance(obj, protocol)


@pytest.mark.parametrize("device", _DEVICES)
def test_series_are_serieslike(device):
    obj = make_series([0], device=device)

    assert isinstance(obj, SeriesLike)
