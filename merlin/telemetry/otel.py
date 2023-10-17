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
from contextlib import AbstractContextManager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from merlin.telemetry.provider import TelemetryProvider

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
TRACER = trace.get_tracer(__name__)


class OtelProvider(TelemetryProvider):
    """
    Merlin telemetry provider that integrates Open Telemetry instrumentation.
    """

    def __init__(self, tracer=TRACER):
        self.tracer = tracer

    def span(self, name: str) -> AbstractContextManager:
        """
        Create a span that is recorded within an OpenTelemetry trace.

        Parameters
        ----------
        name : str
            Identifier for the recorded span

        Returns
        -------
        AbstractContextManager
            A context manager that records a span around the code executed within
        """
        return self.tracer.start_as_current_span(name)
