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
import pyarrow as pa
import pytest

from merlin.core.compat import cudf
from merlin.io.worker import fetch_table_data


class TestFetchTableData:
    def test_pandas_reader(self, tmpdir):
        df = pd.DataFrame({"feature": [1, 2, 3]})
        path = f"{tmpdir}/data.parquet"
        df.to_parquet(path)
        table_cache = {}
        returned_df = fetch_table_data(
            table_cache,
            path,
            reader=pd.read_parquet,
        )
        assert isinstance(returned_df, pd.DataFrame)
        pd.testing.assert_frame_equal(returned_df, df)
        assert not table_cache

    def test_pandas_reader_cache_host(self, tmpdir):
        df = pd.DataFrame({"feature": [1, 2, 3]})
        path = f"{tmpdir}/data.parquet"
        df.to_parquet(path)
        table_cache = {}
        returned_df = fetch_table_data(
            table_cache,
            path,
            reader=pd.read_parquet,
            cache="host",
        )
        assert isinstance(returned_df, pd.DataFrame)
        pd.testing.assert_frame_equal(returned_df, df)
        pd.testing.assert_frame_equal(table_cache[path], df)

    def test_pandas_dataframe_table_cache(self, tmpdir):
        df = pd.DataFrame({"feature": [1, 2, 3]})
        path = f"{tmpdir}/data.parquet"
        df.to_parquet(path)
        table_cache = {path: df}
        returned_df = fetch_table_data(table_cache, path)
        assert isinstance(returned_df, pd.DataFrame)
        pd.testing.assert_frame_equal(returned_df, df)
        pd.testing.assert_frame_equal(table_cache[path], df)

    def test_pandas_arrow_table_cache(self, tmpdir):
        df = pd.DataFrame({"feature": [1, 2, 3]})
        path = f"{tmpdir}/data.parquet"
        df.to_parquet(path)
        table_cache = {path: pa.Table.from_pandas(df)}
        returned_df = fetch_table_data(
            table_cache,
            path,
            reader=pd.read_parquet,
        )
        assert isinstance(returned_df, pd.DataFrame)
        pd.testing.assert_frame_equal(returned_df, df)
        assert table_cache == {path: pa.Table.from_pandas(df)}

    def test_pandas_cats_only(self, tmpdir):
        df = pd.DataFrame({"feature": [1, 2, 3]})
        path = f"{tmpdir}/data.parquet"
        df.to_parquet(path)
        table_cache = {path: pa.Table.from_pandas(df)}
        returned_df = fetch_table_data(
            table_cache,
            path,
            reader=pd.read_parquet,
            cats_only=True,
        )
        assert isinstance(returned_df, pd.DataFrame)
        expected_df = pd.DataFrame({"labels": [0, 1, 2], "feature": [1, 2, 3]})
        pd.testing.assert_frame_equal(returned_df, expected_df)

    @pytest.mark.skipif(not cudf, reason="cuDF required")
    def test_default_cudf(self, tmpdir):
        df = cudf.DataFrame({"feature": [1, 2, 3]})
        path = f"{tmpdir}/data.parquet"
        df.to_parquet(path)
        table_cache = {}
        returned_df = fetch_table_data(
            table_cache,
            path,
        )
        assert isinstance(returned_df, cudf.DataFrame)
        cudf.testing.assert_frame_equal(returned_df, df)
        assert not table_cache

    @pytest.mark.skipif(not cudf, reason="cuDF required")
    def test_cudf_host_cache(self, tmpdir):
        df = cudf.DataFrame({"feature": [1, 2, 3]})
        path = f"{tmpdir}/data.parquet"
        df.to_parquet(path)
        table_cache = {}
        returned_df = fetch_table_data(table_cache, path, cache="host")
        assert isinstance(returned_df, cudf.DataFrame)
        cudf.testing.assert_frame_equal(returned_df, df)
        assert table_cache == {path: df.to_arrow()}

    @pytest.mark.skipif(not cudf, reason="cuDF required")
    def test_cudf_device_cache(self, tmpdir):
        df = cudf.DataFrame({"feature": [1, 2, 3]})
        path = f"{tmpdir}/data.parquet"
        df.to_parquet(path)
        table_cache = {}
        returned_df = fetch_table_data(table_cache, path, cache="device")
        assert isinstance(returned_df, cudf.DataFrame)
        cudf.testing.assert_frame_equal(returned_df, df)
        assert table_cache == {path: df}

    @pytest.mark.skipif(not cudf, reason="cuDF required")
    def test_cudf_table_cache(self, tmpdir):
        df = cudf.DataFrame({"feature": [1, 2, 3]})
        path = f"{tmpdir}/data.parquet"
        df.to_parquet(path)
        table_cache = {path: df.to_arrow()}
        returned_df = fetch_table_data(
            table_cache,
            path,
        )
        assert isinstance(returned_df, cudf.DataFrame)
        cudf.testing.assert_frame_equal(returned_df, df)
        assert table_cache == {path: df.to_arrow()}
