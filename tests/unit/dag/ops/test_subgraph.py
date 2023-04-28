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

from merlin.dag.base_operator import BaseOperator
from merlin.dag.executors import DaskExecutor, LocalExecutor
from merlin.dag.graph import Graph
from merlin.dag.ops.subgraph import Subgraph
from merlin.dag.selector import ColumnSelector
from merlin.dag.stat_operator import StatOperator
from merlin.io import Dataset
from merlin.schema import Schema


@pytest.mark.parametrize("engine", ["parquet"])
def test_subgraph(df):
    ops = ["x"] >> BaseOperator() >> BaseOperator()
    subgraph_op = Subgraph("subgraph", ops)
    main_graph_ops = ["x", "y"] >> BaseOperator() >> subgraph_op >> BaseOperator()

    main_graph = Graph(main_graph_ops)

    main_graph.construct_schema(Schema(list(df.columns)))

    result_df = LocalExecutor().transform(df, main_graph)
    assert result_df == df[["x"]]

    assert main_graph.subgraph("subgraph") == subgraph_op.graph


@pytest.mark.parametrize("engine", ["parquet"])
def test_subgraph_fit(dataset):
    class FitTestOp(StatOperator):
        def fit(self, col_selector: ColumnSelector, dataset: Dataset):
            self.stats = {"fit": True}

        def clear(self):
            self.stats = {}

        def fit_finalize(self, dask_stats):
            return self.stats

    subgraph_op = Subgraph("subgraph", ["x"] >> FitTestOp())
    main_graph_ops = ["x", "y"] >> BaseOperator() >> subgraph_op >> BaseOperator()

    main_graph = Graph(main_graph_ops)
    main_graph.construct_schema(dataset.schema)

    executor = DaskExecutor()
    executor.fit(dataset, main_graph)
    result_df = executor.transform(dataset, main_graph)

    assert result_df.to_ddf().compute() == dataset.to_ddf().compute()[["x"]]
    assert main_graph.subgraph("subgraph").output_node.op.stats["fit"] is True
