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
import numpy as np
import pandas as pd

from merlin.core.dispatch import make_df
from merlin.dag import Graph
from merlin.dag.base_operator import BaseOperator
from merlin.dag.executors import LocalExecutor
from merlin.dag.ops.logging import Logging
from merlin.dag.ops.subgraph import Subgraph
from merlin.schema.schema import ColumnSchema, Schema
from merlin.telemetry import configure_telemetry_provider
from merlin.telemetry.structlog import StructlogProvider


def test_structlog_records_events():
    # Create input dataset
    df = make_df(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
            "d": [10, 11, 12],
            "x": [13, 14, 15],
            "str": ["a", "b", "c"],
        }
    )

    # Build a graph
    schema = Schema(
        [
            ColumnSchema("a", dtype=np.int64),
            ColumnSchema("b", dtype=np.int64),
            ColumnSchema("c", dtype=np.int64),
            ColumnSchema("d", dtype=np.int64),
            ColumnSchema("x", dtype=np.int64),
            ColumnSchema("str", dtype=str),
        ]
    )

    sg1 = ["a", "b"] >> BaseOperator()
    sg2 = ["a", "c"] >> BaseOperator()
    sg3 = ["a", "d"] >> BaseOperator()

    combined1 = Subgraph("sub1", sg1) + Subgraph("sub2", sg2)
    combined2 = Subgraph("combined1", combined1) + Subgraph("sub3", sg3)
    combined3 = Subgraph("combined2", combined2) + (["x", "str"] >> BaseOperator())
    outer_subgraph = Subgraph("combined3", combined3)

    logging = Logging(["a", "b", "str"])
    output_node = "*" >> outer_subgraph >> logging

    graph = Graph(output_node)
    graph.construct_schema(schema)

    # Execute the graph with an executor
    configure_telemetry_provider(StructlogProvider())
    executor = LocalExecutor()
    result = executor.transform(df, [graph.output_node])

    # Check the results
    result_a = result["a"].to_pandas() if not isinstance(result["a"], pd.Series) else result["a"]
    df_a = df["a"].to_pandas() if not isinstance(df["a"], pd.Series) else result["a"]

    assert all(result_a == df_a)
    assert "b" not in result.columns

    # TODO: Figure out what to assert on to check that spans were recorded
    #       Use some kind of mock for this to turn it into a unit test?
