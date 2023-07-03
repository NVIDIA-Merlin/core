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
from merlin.telemetry.provider import TelemetryProvider


TELEMETRY_PROVIDER = None


def configure_telemetry_provider(provider: TelemetryProvider):
    import merlin

    merlin.telemetry.TELEMETRY_PROVIDER = provider


def get_telemetry_provider() -> TelemetryProvider:
    import merlin

    return merlin.telemetry.TELEMETRY_PROVIDER


# @telemetry decorator should not have an arg
# we should only need to add a telemetry param to operator transform methods when we're using telemetry directly
# we shouldn't have to add additional properties to the operators
# the executor should be able to supply a telemetry object when running ops (somehow)


def telemetry(func):
    def wrapper(*args, **kwargs):
        return func(*args, telemetry=TELEMETRY_PROVIDER, **kwargs)

    wrapper.telemetry = True

    return wrapper
