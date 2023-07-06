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
from __future__ import annotations

from typing import Callable

from merlin.core.protocols import Transformable
from merlin.dag import Node
from merlin.dag.executors import DaskExecutor, LocalExecutor
from merlin.dag.graph import Graph
from merlin.dag.ops.stat_operator import StatOperator
from merlin.dag.selector import ColumnSelector
from merlin.io.dataset import Dataset
from merlin.schema import Schema


class Subgraph(StatOperator):
    """
    Operator that executes a Merlin Graph as a sub-graph of another Graph
    """

    def __init__(self, name, output_node, loop_until: Callable = None):
        self.name = name
        self.graph = output_node
        self.loop_until = loop_until

        if not isinstance(output_node, Graph):
            self.graph = Graph(output_node)

        super().__init__()

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        """Simply returns the selected output columns from the input dataframe

        The main functionality of this operator has to do with computing the schemas
        for selection nodes in the Workflow graph, so very little has to happen in the
        `transform` method.

        Parameters
        -----------
        columns: list of str or list of list of str
            The columns to apply this operator to
        transformable: Transformable
            A pandas or cudf dataframe that this operator will work on

        Returns
        -------
        DataFrame
            Returns a transformed dataframe for this operator
        """
        executor = LocalExecutor()
        transformable = executor.transform(transformable, self.graph)
        while self.loop_until and not self.loop_until(transformable):
            transformable = executor.transform(transformable, self.graph)
        return transformable

    def fit(
        self, col_selector: ColumnSelector, dataset: Dataset
    ):  # pylint: disable=arguments-renamed
        DaskExecutor().fit(dataset, self.graph)

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        """
        Return the schemas of columns

        Parameters
        ----------
        root_schema : Schema
            Schema of the columns from the input dataset
        parents_schema : Schema
            Schema of the columns from the parent nodes
        deps_schema : Schema
            Schema of the columns from the dependency nodes
        selector : ColumnSelector
            Existing column selector for this node in the graph (often None)

        Returns
        -------
        Schema
            Schema of selected columns from input, parents, and dependencies
        """
        if not self.graph.input_schema:
            self.graph = self.graph.construct_schema(root_schema)

        return self.graph.input_schema

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ) -> Schema:
        """Given a set of schemas and a column selector for the input columns,
        returns a set of schemas for the transformed columns this operator will produce

        Parameters
        -----------
        input_schema: Schema
            The schemas of the columns to apply this operator to
        col_selector: ColumnSelector
            The column selector to apply to the input schema
        Returns
        -------
        Schema
            The schemas of the columns produced by this operator
        """
        return self.graph.output_schema

    def clear(self):
        """Removes calculated statistics from each node in the graph

        See Also
        --------
        StatOperator.clear
        """
        for stat in Graph.get_nodes_by_op_type([self.graph.output_node], StatOperator):
            stat.op.clear()

    def column_mapping(self, col_selector):
        """Applies logic to obtain correct column names, delegated to operators,
        in subgraph.
        """
        return self.graph.column_mapping

    def fit_finalize(self, dask_stats):
        return dask_stats

    def set_storage_path(self, new_path, copy=False):
        for stat in Graph.get_nodes_by_op_type([self.graph.output_node], StatOperator):
            stat.op.set_storage_path(new_path, copy=copy)

    @property
    def is_subgraph(self):
        """Property that identifies if operator is a subgraph"""
        return True

    def __add__(self, other):
        node = Node.construct_from(self)
        return node + other
