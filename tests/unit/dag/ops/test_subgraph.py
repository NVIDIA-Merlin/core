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

from merlin.core.protocols import Transformable
from merlin.dag.executors import DaskExecutor, LocalExecutor
from merlin.dag.graph import Graph
from merlin.dag.operator import Operator
from merlin.dag.ops.stat_operator import StatOperator
from merlin.dag.ops.subgraph import Subgraph
from merlin.dag.selector import ColumnSelector
from merlin.io import Dataset
from merlin.schema import Schema


@pytest.mark.parametrize("engine", ["parquet"])
def test_subgraph(df):
    ops = ["x"] >> Operator() >> Operator()
    subgraph_op = Subgraph("subgraph", ops)
    main_graph_ops = ["x", "y"] >> Operator() >> subgraph_op >> Operator()

    main_graph = Graph(main_graph_ops)

    main_graph.construct_schema(Schema(list(df.columns)))

    result_df = LocalExecutor().transform(df, main_graph)
    assert (result_df == df[["x"]]).all()[0]

    assert main_graph.subgraph("subgraph") == subgraph_op.graph


@pytest.mark.parametrize("engine", ["parquet"])
def test_subgraph_fit(dataset):
    class FitTestOp(StatOperator):
        def fit(self, col_selector: ColumnSelector, ddf):
            self.stats = {"fit": True}

        def clear(self):
            self.stats = {}

        def fit_finalize(self, dask_stats):
            return self.stats

    fit_test_op = FitTestOp()
    subgraph_op = Subgraph("subgraph", ["x"] >> fit_test_op)
    main_graph_ops = ["x", "y"] >> Operator() >> subgraph_op >> Operator()

    main_graph = Graph(main_graph_ops)
    main_graph.construct_schema(dataset.schema)

    executor = DaskExecutor()
    executor.fit(dataset, main_graph)
    result_df = executor.transform(dataset.to_ddf(), main_graph)

    assert (result_df.compute() == dataset.to_ddf().compute()[["x"]]).all()[0]
    assert main_graph.subgraph("subgraph").output_node.op.stats["fit"] is True


@pytest.mark.parametrize("engine", ["parquet"])
def test_subgraph_looping(dataset):
    class LoopingTestOp(Operator):
        def transform(
            self, col_selector: ColumnSelector, transformable: Transformable
        ) -> Transformable:
            return transformable[col_selector.names] + 1.0

    subgraph = ["x"] >> LoopingTestOp()
    subgraph_op = Subgraph(
        "subgraph",
        subgraph,
        loop_until=lambda transformable: (transformable["x"] > 5.0).all(),
    )
    main_graph_ops = ["x", "y"] >> Operator() >> subgraph_op >> Operator()

    main_graph = Graph(main_graph_ops)
    main_graph.construct_schema(dataset.schema)

    df = dataset.to_ddf().compute()
    df["x"] = df["x"] * 0.0
    dataset = Dataset(df)

    executor = DaskExecutor()
    executor.fit(dataset, main_graph)
    result_df = executor.transform(dataset.to_ddf(), main_graph)

    assert (result_df.compute()[["x"]] > 5.0).all()[0]
