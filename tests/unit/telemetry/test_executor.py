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
from contextlib import contextmanager
from typing import Optional

from merlin.core.dispatch import make_df
from merlin.core.protocols import Transformable
from merlin.dag import BaseOperator, Graph
from merlin.dag.executors import LocalExecutor
from merlin.dag.selector import ColumnSelector
from merlin.schema import Schema
from merlin.telemetry import TelemetryProvider, telemetry


def configure_telemetry(provider: TelemetryProvider):
    import merlin

    merlin.telemetry.TELEMETRY_PROVIDER = provider


class TestTelemetry(TelemetryProvider):
    def __init__(self):
        self.spans = []

    def span(self, name):
        self.spans.append(name)


configure_telemetry(TestTelemetry())


class InstrumentedExecutor(LocalExecutor):
    def __init__(self, telemetry=None):
        super().__init__()
        self.telemetry = telemetry

    def transform(
        self,
        transformable,
        graph,
        output_dtypes=None,
        additional_columns=None,
        capture_dtypes=False,
        strict=False,
        target_format=None,
    ):
        self.telemetry.span(f"{self.__class__.__name__} start")
        result = super().transform(
            transformable,
            graph,
            output_dtypes,
            additional_columns,
            capture_dtypes,
            strict,
            target_format,
        )
        self.telemetry.span(f"{self.__class__.__name__} end")
        return result

    def run_op_transform(self, node, input_data, selection):
        if self.telemetry:
            self.telemetry.span(f"{node.op.__class__.__name__} start")

        transformed_data = super().run_op_transform(node, input_data, selection)

        if self.telemetry:
            self.telemetry.span(f"{node.op.__class__.__name__} end")

        return transformed_data


@contextmanager
def record_span(telemetry: TelemetryProvider, name: str):
    telemetry.span(f"{name} cm start")
    try:
        yield
    finally:
        telemetry.span(f"{name} cm end")


class InstrumentedOperator(BaseOperator):
    @telemetry
    def transform(
        self,
        col_selector: ColumnSelector,
        transformable: Transformable,
        telemetry: Optional[TelemetryProvider] = None,
    ) -> Transformable:
        telemetry.span(f"{self.__class__.__name__} decorator start")
        result = super().transform(col_selector, transformable)
        telemetry.span(f"{self.__class__.__name__} decorator end")

        return result


class ContextManagerOperator(BaseOperator):
    @telemetry
    def transform(
        self,
        col_selector: ColumnSelector,
        transformable: Transformable,
        telemetry: Optional[TelemetryProvider] = None,
    ) -> Transformable:
        with record_span(telemetry, self.__class__.__name__):
            result = super().transform(col_selector, transformable)

        return result


def test_executor_with_telemetry():
    # Construct an operator graph
    output_node = "*" >> BaseOperator()
    graph = Graph(output_node)

    # Instantiate an executor and provide telemetry to it
    telem_obj = TestTelemetry()
    configure_telemetry(telem_obj)
    executor = InstrumentedExecutor(telemetry=telem_obj)

    # Execute the graph's transform with that executor
    df = make_df({"a": [1, 2, 3]})
    graph.construct_schema(Schema(["a"]))
    executor.transform(df, graph)

    # Assert against the contents of the telemetry object
    assert telem_obj.spans
    assert telem_obj.spans == [
        "InstrumentedExecutor start",
        "SelectionOp start",
        "SelectionOp end",
        "BaseOperator start",
        "BaseOperator end",
        "InstrumentedExecutor end",
    ]


def test_operator_with_decorator_telemetry():
    # Construct an operator graph
    output_node = "*" >> InstrumentedOperator()
    graph = Graph(output_node)

    # Instantiate an executor and provide telemetry to it
    telem_obj = TestTelemetry()
    configure_telemetry(telem_obj)
    executor = InstrumentedExecutor(telemetry=telem_obj)

    # Execute the graph's transform with that executor
    df = make_df({"a": [1, 2, 3]})
    graph.construct_schema(Schema(["a"]))
    executor.transform(df, graph)

    # Assert against the contents of the telemetry object
    assert telem_obj.spans
    assert telem_obj.spans == [
        "InstrumentedExecutor start",
        "SelectionOp start",
        "SelectionOp end",
        "InstrumentedOperator start",
        "InstrumentedOperator decorator start",
        "InstrumentedOperator decorator end",
        "InstrumentedOperator end",
        "InstrumentedExecutor end",
    ]


def test_operator_with_cm_telemetry():
    # Construct an operator graph
    output_node = "*" >> ContextManagerOperator()
    graph = Graph(output_node)

    # Instantiate an executor and provide telemetry to it
    telem_obj = TestTelemetry()
    configure_telemetry(telem_obj)
    executor = InstrumentedExecutor(telemetry=telem_obj)

    # Execute the graph's transform with that executor
    df = make_df({"a": [1, 2, 3]})
    graph.construct_schema(Schema(["a"]))
    executor.transform(df, graph)

    # Assert against the contents of the telemetry object
    assert telem_obj.spans
    assert telem_obj.spans == [
        "InstrumentedExecutor start",
        "SelectionOp start",
        "SelectionOp end",
        "ContextManagerOperator start",
        "ContextManagerOperator cm start",
        "ContextManagerOperator cm end",
        "ContextManagerOperator end",
        "InstrumentedExecutor end",
    ]
