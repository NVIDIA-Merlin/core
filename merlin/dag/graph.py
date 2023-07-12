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

import logging
from collections import deque
from typing import Dict

from merlin.dag.node import (
    Node,
    _combine_schemas,
    iter_nodes,
    postorder_iter_nodes,
    preorder_iter_nodes,
)
from merlin.dag.operator import Operator
from merlin.dag.ops.stat_operator import StatOperator
from merlin.schema import Schema

LOG = logging.getLogger("merlin")


class Graph:
    """
    Represents an DAG composed of Nodes, each of which contains an operator that
    transforms dataframes or dataframe-like data
    """

    def __init__(self, output_node: Node):
        if isinstance(output_node, Operator):
            output_node = Node.construct_from(output_node)

        self.output_node = output_node

        parents_with_deps = self.output_node.parents_with_dependencies
        parents_with_deps.append(output_node)

        self.subgraphs: Dict[str, "Graph"] = {}
        _find_subgraphs(output_node, self.subgraphs)
        for node in list(postorder_iter_nodes(self.output_node, flatten_subgraphs=True)):
            node.op.load_artifacts("")

    def subgraph(self, name: str) -> "Graph":
        if name not in self.subgraphs:
            raise ValueError(f"No subgraph named {name}. Options are: {self.subgraphs.keys()}")
        return self.subgraphs[name]

    @property
    def input_dtypes(self):
        if self.input_schema:
            return {
                name: col_schema.dtype
                for name, col_schema in self.input_schema.column_schemas.items()
            }
        else:
            return {}

    @property
    def output_dtypes(self):
        if self.output_schema:
            return {
                name: col_schema.dtype
                for name, col_schema in self.output_schema.column_schemas.items()
            }
        else:
            return {}

    @property
    def column_mapping(self):
        nodes = preorder_iter_nodes(self.output_node)
        column_mapping = self.output_node.column_mapping
        for node in list(nodes)[1:]:
            node_map = node.column_mapping
            for output_col, input_cols in column_mapping.items():
                early_inputs = []
                for input_col in input_cols:
                    early_inputs += node_map.get(input_col, [input_col])
                column_mapping[output_col] = early_inputs

        return column_mapping

    def construct_schema(self, root_schema: Schema, preserve_dtypes=False) -> "Graph":
        """
        Given the schema of a dataset to transform, determine the output schema of the graph

        Parameters
        ----------
        root_schema : Schema
            The schema of a dataset to be transformed with this DAG
        preserve_dtypes : bool, optional
            Whether to keep any dtypes that may already be present in the schemas, by default False

        Returns
        -------
        Graph
            This DAG after the schemas have been filled in
        """
        nodes = list(postorder_iter_nodes(self.output_node))

        self._compute_node_schemas(root_schema, nodes, preserve_dtypes)
        self._validate_node_schemas(root_schema, nodes, preserve_dtypes)

        return self

    def _compute_node_schemas(self, root_schema, nodes, preserve_dtypes=False):
        for node in nodes:
            node.compute_schemas(root_schema, preserve_dtypes=preserve_dtypes)

    def _validate_node_schemas(self, root_schema, nodes, strict_dtypes=False):
        for node in nodes:
            node.validate_schemas(root_schema, strict_dtypes=strict_dtypes)

    @property
    def input_schema(self):
        # leaf_node input and output schemas are the same (aka selection)
        # subgraphs can also be leaf nodes now, so input and output are different
        return _combine_schemas(self.leaf_nodes, input_schemas=True)

    @property
    def leaf_nodes(self):
        return [node for node in postorder_iter_nodes(self.output_node) if not node.parents]

    @property
    def output_schema(self):
        return self.output_node.output_schema

    def _input_columns(self):
        input_cols = []
        for node in iter_nodes([self.output_node]):
            upstream_output_cols = []

            for upstream_node in node.parents_with_dependencies:
                upstream_output_cols += upstream_node.output_columns.names

            upstream_output_cols = _get_unique(upstream_output_cols)
            input_cols += list(set(node.input_columns.names) - set(upstream_output_cols))

        return _get_unique(input_cols)

    def remove_inputs(self, to_remove):
        """
        Removes columns from a Graph

        Starting at the leaf nodes, trickle down looking for columns to remove,
        when found remove but then must propagate the removal of any other
        output columns derived from that column.

        Parameters
        -----------
        graph : Graph
            The graph to remove columns from
        to_remove : array_like
            A list of input column names to remove from the graph

        Returns
        -------
        Graph
            The same graph with columns removed
        """
        nodes_to_process = deque([(node, to_remove) for node in self.leaf_nodes])

        while nodes_to_process:
            node, columns_to_remove = nodes_to_process.popleft()
            if node.input_schema and len(node.input_schema):
                output_columns_to_remove = node.remove_inputs(columns_to_remove)

                for child in node.children:
                    nodes_to_process.append(
                        (child, list(set(to_remove + output_columns_to_remove)))
                    )

                    if not len(node.input_schema):
                        node.remove_child(child)

            # remove any dependencies that do not have an output schema
            node.dependencies = [
                dep for dep in node.dependencies if dep.output_schema and len(dep.output_schema)
            ]

            if not node.input_schema or not len(node.input_schema):
                for parent in node.parents:
                    parent.remove_child(node)
                for dependency in node.dependencies:
                    dependency.remove_child(node)
                del node

        return self

    @classmethod
    def get_nodes_by_op_type(cls, nodes, op_type):
        return set(node for node in iter_nodes(nodes) if isinstance(node.op, op_type))

    def clear_stats(self):
        """Removes calculated statistics from each node in the graph

        See Also
        --------
        StatOperator.clear
        """
        for stat in Graph.get_nodes_by_op_type([self.output_node], StatOperator):
            stat.op.clear()


def _get_schemaless_nodes(nodes):
    schemaless_nodes = []
    for node in iter_nodes(nodes):
        if node.input_schema is None:
            schemaless_nodes.append(node)

    return set(schemaless_nodes)


def _get_unique(cols):
    # Need to preserve order in unique-column list
    return list({x: x for x in cols}.keys())


def _find_subgraphs(output_node, subgraphs):
    for node in postorder_iter_nodes(output_node):
        op = node.op
        if op.is_subgraph and op.name:
            if op.name in subgraphs:
                raise ValueError(
                    f"Found two subgraphs called {op.name}. "
                    "Each subgraph must have a unique name."
                )
            subgraphs[op.name] = op.graph
            _find_subgraphs(op.graph.output_node, subgraphs)
