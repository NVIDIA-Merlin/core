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
from typing import Dict

import numpy as np

from merlin.core.dispatch import make_df
from merlin.dag import Graph
from merlin.dag.base_operator import BaseOperator
from merlin.dag.executors import LocalExecutor
from merlin.schema.schema import ColumnSchema, Schema


def test_local_executor_with_dataframe():
    df = make_df({"a": [1, 2, 3], "b": [4, 5, 6]})
    schema = Schema([ColumnSchema("a", dtype=np.int64), ColumnSchema("b", dtype=np.int64)])
    operator = ["a"] >> BaseOperator()
    graph = Graph(operator)
    graph.construct_schema(schema)

    executor = LocalExecutor()
    result = executor.transform(df, [graph.output_node])

    assert all(result["a"].to_pandas() == df["a"].to_pandas())
    assert "b" not in result.columns


def test_local_executor_with_multiple_dataframe():
    df0 = make_df({"a": [1, 2, 3]})
    df1 = make_df({"a": [4, 5, 6]})
    schema = Schema([ColumnSchema("a", dtype=np.int64)])
    operator = ["a"] >> BaseOperator()
    graph = Graph(operator)
    graph.construct_schema(schema)

    executor = LocalExecutor()
    result = executor.transform_multi((df0, df1), [graph.output_node])

    assert all(result[0]["a"].to_pandas() == df0["a"].to_pandas())
    assert "b" not in result[0].columns

    assert all(result[1]["a"].to_pandas() == df1["a"].to_pandas())
    assert "b" not in result[1].columns


# ==========================================================================


class Column:
    def __init__(self, data, dtype):
        self.data = data
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def __eq__(self, other):
        return self.data == other.data and self._dtype == other._dtype


class DataFrameLike:
    def __init__(self, data: Dict, dtypes: Dict):
        self.data = data
        self.dtypes = dtypes

    @property
    def columns(self):
        return list(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        return self.data == other.data and self.dtypes == other.dtypes

    def __getitem__(self, key):
        if isinstance(key, list):
            return DataFrameLike(
                data={k: self.data[k] for k in key},
                dtypes={k: self.dtypes[k] for k in key},
            )
        else:
            return Column(self.data[key], self.dtypes[key])

    def _grab_keys(self, source, keys):
        return {k: source[k] for k in keys}


def test_local_executor_with_dataframe_like():
    df = DataFrameLike({"a": [1, 2, 3], "b": [4, 5, 6]}, dtypes={"a": np.int64, "b": np.int64})
    schema = Schema([ColumnSchema("a", dtype=np.int64), ColumnSchema("b", dtype=np.int64)])
    operator = ["a"] >> BaseOperator()
    graph = Graph(operator)
    graph.construct_schema(schema)

    executor = LocalExecutor()
    result = executor.transform(df, [graph.output_node])

    assert result["a"] == df["a"]
    assert "b" not in result.columns


def test_local_executor_with_multiple_dataframe_like():
    df0 = DataFrameLike({"a": [1, 2, 3]}, dtypes={"a": np.int64})
    df1 = DataFrameLike({"a": [4, 5, 6]}, dtypes={"a": np.int64})

    schema = Schema([ColumnSchema("a", dtype=np.int64)])
    operator = ["a"] >> BaseOperator()
    graph = Graph(operator)
    graph.construct_schema(schema)

    executor = LocalExecutor()
    result = executor.transform_multi((df0, df1), [graph.output_node])

    assert result[0]["a"] == df0["a"]
    assert "b" not in result[0].columns

    assert result[1]["a"] == df1["a"]
    assert "b" not in result[0].columns