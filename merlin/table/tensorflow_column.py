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
from dataclasses import dataclass
from typing import Any

from merlin.core.compat import tensorflow as tf
from merlin.table.conversions import _from_dlpack_cpu, _from_dlpack_gpu, _to_dlpack
from merlin.table.tensor_column import Device, TensorColumn


@dataclass(frozen=True)
class TensorflowDlpackWrapper:
    capsule: Any

    def __dlpack__(self):
        return self.capsule


class TensorflowColumn(TensorColumn):
    @classmethod
    def array_type(cls):
        return tf.Tensor

    @classmethod
    def supported_devices(cls):
        return [Device.CPU, Device.GPU]

    @classmethod
    def cast(cls, other):
        return convert_column(other, target_type=cls)

    def __init__(self, values: tf.Tensor, offsets: tf.Tensor = None, dtype=None, _ref=None):
        values_device = self._tf_device(values)
        offsets_device = self._tf_device(offsets)
        if values_device != offsets_device:
            raise ValueError(
                f"Values and offsets were detected on different devices: "
                f"values ({values_device}) and offsets ({offsets_device})."
            )
        super().__init__(values, offsets, dtype, _device=values_device, _ref=_ref)

    def to(self, target_type):
        return convert_column(self, target_type=target_type)

    def _tf_device(self, tensor):
        return Device.GPU if "GPU" in tensor.device else Device.CPU


@_to_dlpack.register_lazy("tensorflow")
def register_to_dlpack_from_tf():
    import tensorflow as tf

    eager_tensor_type = type(tf.random.uniform((1,)))

    @_to_dlpack.register(tf.Tensor)
    @_to_dlpack.register(eager_tensor_type)
    def _to_dlpack_from_tf_tensor(tensor):
        return TensorflowDlpackWrapper(tf.experimental.dlpack.to_dlpack(tensor))


@_from_dlpack_cpu.register_lazy("tensorflow")
def register_from_dlpack_cpu_to_tf():
    import tensorflow as tf

    eager_tensor_type = type(tf.random.uniform((1,)))

    @_from_dlpack_cpu.register(tf.Tensor)
    @_from_dlpack_cpu.register(eager_tensor_type)
    def _from_dlpack_cpu_to_tf(target_type, array):
        return tf.experimental.dlpack.from_dlpack(array.__dlpack__())


@_from_dlpack_gpu.register_lazy("tensorflow")
def register_from_dlpack_gpu_to_tf():
    import tensorflow as tf

    eager_tensor_type = type(tf.random.uniform((1,)))

    @_from_dlpack_gpu.register(tf.Tensor)
    @_from_dlpack_gpu.register(eager_tensor_type)
    def _from_dlpack_gpu_to_tf(target_type, array):
        return tf.experimental.dlpack.from_dlpack(array.__dlpack__())


# @_to_array_interface.register_lazy("tensorflow")
# def register_to_array_interface_numpy():
#     import numpy as np

#     @_to_array_interface.register(np.ndarray)
#     def _to_array_interface_numpy(array):
#         return array

# @_from_array_interface.register_lazy("tensorflow")
# def register_from_array_interface_numpy():
#     import numpy as np

#     @_from_array_interface.register(np.ndarray)
#     def _from_array_interface_numpy(target_type, array_interface):
#         return np.asarray(array_interface)
