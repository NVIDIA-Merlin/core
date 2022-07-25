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
from merlin.dag import BaseOperator, Graph, Node
from merlin.dag.ops.selection import SelectionOp
from merlin.dag.selector import ColumnSelector
from merlin.schema.schema import ColumnSchema, Schema


def test_remove_inputs():
    # construct a basic graph , mimicking the structure shown here
    # https://github.com/NVIDIA-Merlin/NVTabular/issues/1632
    selector = ColumnSelector(["a", "b", "c"])
    operator = BaseOperator()
    op_node = selector >> operator
    output_node = op_node + "target"
    graph = Graph(output_node)
    graph.construct_schema(Schema(["a", "b", "c", "target"]))

    # remove the 'target' column from the inputs
    graph.remove_inputs(["target"])

    # make sure the 'target' is removed everywhere.
    to_examine = [graph.output_node]
    while to_examine:
        current = to_examine.pop()

        assert "target" not in current.selector.names
        assert "target" not in current.output_schema

        if current.op and isinstance(current.op, SelectionOp):
            assert "target" not in current.op.selector.names

        to_examine.extend(current.parents_with_dependencies)


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
