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

from merlin.dag.base_operator import BaseOperator as Operator
from merlin.dag.graph import Graph
from merlin.dag.selector import ColumnSelector
from merlin.schema import Schema


@pytest.mark.parametrize("engine", ["parquet"])
def test_graph_validates_schemas(dataset, engine):
    ops = ["a", "b", "c"] >> Operator()
    graph = Graph(ops)

    with pytest.raises(ValueError) as exc_info:
        graph.construct_schema(dataset.schema)

    assert "Missing column" in str(exc_info.value)


@pytest.mark.parametrize("engine", ["parquet"])
def test_compute_selector_validates_schemas(dataset, engine):
    op = Operator()
    schema = Schema(["a", "b"])
    selector = ColumnSelector(["c"])

    with pytest.raises(ValueError) as exc_info:
        op.compute_selector(schema, selector, ColumnSelector(), ColumnSelector())

    assert "Missing column" in str(exc_info.value)


@pytest.mark.parametrize("engine", ["parquet"])
def test_compute_input_schema_validates_schemas(dataset, engine):
    op = Operator()
    schema = Schema(["a", "b"])
    selector = ColumnSelector(["c"])

    with pytest.raises(ValueError) as exc_info:
        op.compute_input_schema(schema, Schema(), Schema(), selector)

    assert "Missing column" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        op.compute_input_schema(Schema(), schema, Schema(), selector)

    assert "Missing column" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        op.compute_input_schema(Schema(), Schema(), schema, selector)

    assert "Missing column" in str(exc_info.value)


@pytest.mark.parametrize("engine", ["parquet"])
def test_compute_output_schema_validates_schemas(dataset, engine):
    op = Operator()
    schema = Schema(["a", "b"])
    selector = ColumnSelector(["c"])

    with pytest.raises(ValueError) as exc_info:
        op.compute_output_schema(schema, selector)

    assert "Missing column" in str(exc_info.value)
