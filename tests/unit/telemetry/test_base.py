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

from merlin.telemetry import Telemetry, telemetry


def configure_telemetry(telemeter: Telemetry):
    import merlin

    merlin.telemetry.TELEMETRY = telemeter


class TestTelemetry(Telemetry):
    def __init__(self):
        self.spans = []

    def span(self, name):
        self.spans.append(name)


configure_telemetry(TestTelemetry())


def test_span_gets_called():
    @telemetry
    def x(telemetry=None):
        telemetry.span("newspan")
        return telemetry

    result = x()
    assert "newspan" in result.spans


def test_span_gets_called_with_other_arg():
    @telemetry
    def x(other_arg, telemetry=None):
        telemetry.span(other_arg)
        return telemetry

    result = x("newspan")
    assert "newspan" in result.spans
