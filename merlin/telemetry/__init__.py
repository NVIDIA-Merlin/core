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
from typing import Optional

from merlin.telemetry.base import Telemetry
from merlin.telemetry.otel import OtelTelemetry

TELEMETRY = None


# @telemetry decorator should not have an arg
# we should only need to add a telemetry param to operator transform methods when we're using telemetry directly
# we shouldn't have to add additional properties to the operators
# the executor should be able to supply a telemetry object when running ops (somehow)


def telemetry(func):
    def wrapper(*args, **kwargs):
        return func(*args, telemetry=TELEMETRY, **kwargs)

    wrapper.telemetry = True

    return wrapper


# class BaseOperator:
#     @property
#     def TELEMETRY(self):
#         ...

#     def telemetry(self, func):
#         def wrapper(*args, **kwargs):
#             return func(*args, telemetry=self.TELEMETRY, **kwargs)

#         wrapper.telemetry = True

#         return wrapper

# class MyOperator(BaseOperator):

#     def transform(self, col_):
#         ...

# # executor().transform(transformable, nodes, tracer=tracer)


# class MyExecutor:
#     def __init__(self, telem: Telemetry):
#         self.telemetry = telem

#     def transform(self, fn):
#         if hasattr(fn, "telemetry"):
#             return fn(telemetry=self.telemetry)
#         else:
#             return fn()


# @telemetry
# def my_fn(telemetry: Optional[Telemetry] = None):
#     if telemetry:
#         return telemetry.span


# def my_other_fn():
#     return 42


# my_exec = MyExecutor(Telemetry())
# my_exec.transform(my_fn)

# my_exec.transform(my_other_fn)
