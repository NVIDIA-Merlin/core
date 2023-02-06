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
from functools import singledispatch
from typing import Any, Union

import merlin.dtypes as md
from merlin.core.compat import cupy as cp
from merlin.core.compat import numpy as np
from merlin.core.compat import tensorflow as tf
from merlin.core.compat import tf_ops
from merlin.core.compat import torch as th
from merlin.core.protocols import SeriesLike
from merlin.dag.table.tensor_column import Device, TensorColumn


class CupyColumn(TensorColumn):
    @classmethod
    def cast(cls, other):
        column = cls(to_cupy(other.values), to_cupy(other.offsets))
        column._ref = (other.values, other.offsets)
        return column

    def __init__(self, values: cp.ndarray, offsets: cp.ndarray = None, dtype=None):
        super().__init__(values, offsets, dtype)

    @property
    def device(self) -> Device:
        return Device.GPU


def to_cupy(tensor):
    return _to_cupy(tensor)


@singledispatch
def _to_cupy(tensor):
    raise NotImplementedError


if cp:

    @_to_cupy.register
    def cupy_to_cupy(tensor: cp.ndarray):
        return cp.asarray(tensor)


if np:

    @_to_cupy.register
    def numpy_to_cupy(tensor: np.ndarray):
        return cp.asarray(tensor)


if th:

    @_to_cupy.register
    def torch_to_cupy(tensor: th.Tensor):
        return cp.asarray(tensor.numpy())


if tf:

    @_to_cupy.register(tf.Tensor)
    @_to_cupy.register(tf_ops.EagerTensor)
    def tf_to_cupy(tensor: Union[tf.Tensor, tf_ops.EagerTensor]):
        return cp.asarray(tensor.numpy())
