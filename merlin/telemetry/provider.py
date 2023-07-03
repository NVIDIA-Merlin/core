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

from contextlib import AbstractContextManager, nullcontext


class TelemetryProvider:
    """
    Abstract base class for Merlin telemetry providers that integrate external telemetry tools
    """

    def span(self, name: str) -> AbstractContextManager:
        """
        Create a span that is recorded within a trace.

        Parameters
        ----------
        name : str
            Identifier for the recorded span

        Returns
        -------
        AbstractContextManager
            A context manager that records a span around the code executed within
        """
        return nullcontext()

    def log(self, event: dict):
        """
        _summary_

        Parameters
        ----------
        event : dict
            _description_

        Returns
        -------
        _type_
            _description_
        """
