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
import pandas as pd

from merlin.core.dispatch import make_df
from merlin.dag import DictArray, Graph
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

    result_a = result["a"].to_pandas() if not isinstance(result["a"], pd.Series) else result["a"]
    df_a = df["a"].to_pandas() if not isinstance(df["a"], pd.Series) else result["a"]

    assert all(result_a == df_a)
    assert "b" not in result.columns


def test_local_executor_with_dataframe_like():
    df = DictArray(
        {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}, dtypes={"a": np.int64, "b": np.int64}
    )
    schema = Schema([ColumnSchema("a", dtype=np.int64), ColumnSchema("b", dtype=np.int64)])
    operator = ["a"] >> BaseOperator()
    graph = Graph(operator)
    graph.construct_schema(schema)

    executor = LocalExecutor()
    result = executor.transform(df, [graph.output_node])

    assert all(result["a"] == df["a"])
    assert "b" not in result.columns
