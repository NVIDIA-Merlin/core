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
from merlin.telemetry.provider import NullTelemetryProvider, TelemetryProvider

TELEMETRY_PROVIDER: TelemetryProvider = NullTelemetryProvider()


def configure_telemetry_provider(provider: TelemetryProvider):
    """
    Set the global telemetry provider for Merlin

    Parameters
    ----------
    provider : TelemetryProvider

    """
    import merlin

    merlin.telemetry.TELEMETRY_PROVIDER = provider


def get_telemetry_provider() -> TelemetryProvider:
    """
    Get the global telemetry provider for Merlin

    Returns
    -------
    TelemetryProvider
        The telemetry provider currently in use
    """
    import merlin

    return merlin.telemetry.TELEMETRY_PROVIDER


def telemetry(func):
    """
    A decorator that automatically records a span around the provided function

    Parameters
    ----------
    func : Callable
        Function to be wrapped by decorator.
    """

    def wrapper(*args, **kwargs):
        telemetry = get_telemetry_provider()
        with telemetry.span(f"{func.__qualname__}()"):
            return func(*args, **kwargs)

    return wrapper
