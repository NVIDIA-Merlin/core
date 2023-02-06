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


class NumpyColumn(TensorColumn):
    @classmethod
    def cast(cls, other):
        column = cls(to_numpy(other.values), to_numpy(other.offsets))
        column._ref = (other.values, other.offsets)
        return column

    def __init__(self, values: np.ndarray, offsets: np.ndarray = None, dtype=None):
        super().__init__(values, offsets, dtype)

    @property
    def device(self) -> Device:
        return Device.CPU


# This wrapper method allows us to use explicit conditional dispatch
# in conjunction with type-based single-dispatch (which may be useful
# for frameworks that don't have simple types to work with)
def to_numpy(tensor):
    return _to_numpy(tensor)


@singledispatch
def _to_numpy(tensor):
    raise NotImplementedError


if np:

    @_to_numpy.register
    def numpy_to_numpy(tensor: np.ndarray):
        return tensor


if cp:

    @_to_numpy.register
    def cupy_to_numpy(tensor: cp.ndarray):
        return cp.asnumpy(tensor)


if th:

    @_to_numpy.register
    def torch_to_numpy(tensor: th.Tensor):
        return tensor.numpy()


if tf:

    @_to_numpy.register(tf.Tensor)
    @_to_numpy.register(tf_ops.EagerTensor)
    def tf_to_numpy(tensor: Union[tf.Tensor, tf_ops.EagerTensor]):
        return tensor.numpy()
