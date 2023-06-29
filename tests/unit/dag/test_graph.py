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

from merlin.core.dispatch import make_df
from merlin.dag import Graph, Node, iter_nodes, postorder_iter_nodes, preorder_iter_nodes
from merlin.dag.base_operator import BaseOperator
from merlin.dag.executors import LocalExecutor
from merlin.dag.ops.subgraph import Subgraph
from merlin.dag.ops.udf import UDF
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

    combined = Subgraph("sub1", sg1) + Subgraph("sub2", sg2) + sg3
    graph = Graph(
        combined,
    )
    assert graph.subgraph("sub1").output_node == sg1
    assert graph.subgraph("sub2").output_node == sg2

    with pytest.raises(ValueError):
        graph.subgraph("sub3")


def test_subgraph_with_summed_subgraphs():
    sg1 = ["a", "b"] >> BaseOperator()
    sg2 = ["a", "c"] >> BaseOperator()
    sg3 = ["a", "d"] >> BaseOperator()

    combined1 = Subgraph("sub1", sg1) + Subgraph("sub2", sg2)
    combined2 = Subgraph("combined1", combined1) + Subgraph("sub3", sg3)
    combined3 = Subgraph("combined2", combined2) + (["x"] >> BaseOperator())
    output = Subgraph("combined3", combined3)

    graph = Graph(output)

    assert graph.subgraph("sub1").output_node == sg1
    assert graph.subgraph("sub2").output_node == sg2
    assert graph.subgraph("sub3").output_node == sg3
    assert graph.subgraph("combined1").output_node == combined1
    assert graph.subgraph("combined2").output_node == combined2
    assert graph.subgraph("combined3").output_node == combined3

    post_len = len(list(postorder_iter_nodes(graph.output_node)))
    pre_len = len(list(preorder_iter_nodes(graph.output_node)))
    iter_node_list = list(iter_nodes([graph.output_node]))
    iter_len = len(iter_node_list)

    assert post_len == pre_len
    assert iter_len == post_len
    assert iter_len == pre_len


def test_selector_concat_parents():
    df = make_df({"a": [1, 1, 1, 1, 1, 1], "b": [1, 1, 1, 1, 1, 1]})

    graph = Graph((["a", "b"] >> UDF(lambda x: x + 1)) + ["a"])

    schema = Schema(["a", "b"])

    graph.construct_schema(schema)
    result2 = LocalExecutor().transform(df, graph)
    assert (result2["b"] == [2, 2, 2, 2, 2, 2]).all()
    assert (result2["a"] == [1, 1, 1, 1, 1, 1]).all()


def test_selector_concat_parents_inverted():
    df = make_df({"a": [1, 1, 1, 1, 1, 1], "b": [1, 1, 1, 1, 1, 1]})

    graph = Graph((["a"] >> UDF(lambda x: x + 1)) + ["a", "b"])

    schema = Schema(["a", "b"])

    graph.construct_schema(schema)
    result2 = LocalExecutor().transform(df, graph)
    assert (result2["b"] == [1, 1, 1, 1, 1, 1]).all()
    assert (result2["a"] == [1, 1, 1, 1, 1, 1]).all()


def test_selector_concat_parents_open():
    df = make_df({"a": [1, 1, 1, 1, 1, 1], "b": [1, 1, 1, 1, 1, 1]})

    graph = Graph((["a"] >> UDF(lambda x: x + 1)) + ["b"])

    schema = Schema(["a", "b"])

    graph.construct_schema(schema)
    result2 = LocalExecutor().transform(df, graph)
    assert (result2["b"] == [1, 1, 1, 1, 1, 1]).all()
    assert (result2["a"] == [2, 2, 2, 2, 2, 2]).all()
