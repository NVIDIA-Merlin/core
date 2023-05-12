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
import pandas as pd
import pytest

from merlin.core.compat import HAS_GPU, cudf
from merlin.core.dispatch import make_df
from merlin.io import Dataset


class TestDatasetCpu:
    def test_true(self):
        dataset = Dataset(make_df({"a": [1, 2, 3]}), cpu=True)
        assert dataset.cpu is True
        assert isinstance(dataset.compute(), pd.DataFrame)

    @pytest.mark.skipif(not (cudf and HAS_GPU), reason="requires cuDF and GPU")
    def test_default_cudf(self):
        dataset = Dataset(make_df({"a": [1, 2, 3]}))
        assert dataset.cpu is False
        assert isinstance(dataset.compute(), cudf.DataFrame)

    @pytest.mark.skipif(cudf and HAS_GPU, reason="requires environment without cuDF and GPU")
    def test_default_pandas(self):
        dataset = Dataset(make_df({"a": [1, 2, 3]}))
        assert dataset.cpu is True
        assert isinstance(dataset.compute(), pd.DataFrame)

    @pytest.mark.skipif(not (cudf and HAS_GPU), reason="requires cuDF and GPU")
    def test_false_with_cudf_and_gpu(self):
        dataset = Dataset(make_df({"a": [1, 2, 3]}), cpu=False)
        assert dataset.cpu is False
        assert isinstance(dataset.compute(), cudf.DataFrame)

    @pytest.mark.skipif(cudf or HAS_GPU, reason="requires environment without cuDF or GPU")
    def test_false_missing_cudf_or_gpu(self):
        with pytest.raises(RuntimeError):
            Dataset(make_df({"a": [1, 2, 3]}), cpu=False)


def test_infer_list_dtype_unknown():
    df = pd.DataFrame({"col": [[], []]})
    dataset = Dataset(df, cpu=True)
    assert dataset.schema["col"].dtype.element_type.value == "unknown"
