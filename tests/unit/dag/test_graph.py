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
from merlin.dag import BaseOperator
from merlin.dag.graph import Graph
from merlin.dag.ops.selection import SelectionOp
from merlin.dag.selector import ColumnSelector
from merlin.schema import Schema


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
