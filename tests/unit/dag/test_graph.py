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

from merlin.dag import Graph, Node
from merlin.dag.base_operator import BaseOperator
from merlin.dag.selector import ColumnSelector
from merlin.schema.schema import ColumnSchema, Schema


def test_remove_dependencies():
    # Construct a simple graph with a structure like:
    #   ["y"] ----> ["x", "y"] ---\
    #                              --- > ["o"]
    #   ["z"] --------------------/

    # When removing "y", we should see all of the dependencies
    # and parents removed from the list of leaf nodes.

    dep_node = Node(selector=ColumnSelector(["y"]))
    dep_node.input_schema = Schema([ColumnSchema("y")])
    dep_node.output_schema = Schema([ColumnSchema("y")])

    node_xy = Node(selector=ColumnSelector(["x", "y"]))
    node_xy.input_schema = Schema([ColumnSchema("x"), ColumnSchema("y")])
    node_xy.output_schema = Schema([ColumnSchema("z")])

    plus_node = Node(selector=ColumnSelector(["z", "y"]))
    plus_node.input_schema = Schema([ColumnSchema("y"), ColumnSchema("z")])
    plus_node.output_schema = Schema([ColumnSchema("o")])
    plus_node.add_parent(dep_node)
    plus_node.add_parent(node_xy)
    plus_node.add_dependency(dep_node)

    graph_with_dependency = Graph(plus_node)
    assert len(graph_with_dependency.leaf_nodes) == 2
    graph_with_dependency.remove_inputs(["y"])
    assert len(graph_with_dependency.leaf_nodes) == 1


def test_subgraph():
    sg1 = ["a", "b"] >> BaseOperator()
    sg2 = ["a", "c"] >> BaseOperator()
    sg3 = ["a", "d"] >> BaseOperator()

    combined = sg1 + sg2 + sg3
    graph = Graph(combined, subgraphs={"sub1": sg1, "sub2": sg2})
    assert graph.subgraph("sub1").output_node == sg1
    assert graph.subgraph("sub2").output_node == sg2

    with pytest.raises(ValueError):
        graph.subgraph("sg3")
