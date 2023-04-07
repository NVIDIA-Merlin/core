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
from typing import Callable, Type

from merlin.core.compat.torch import torch as th
from merlin.table.conversions import _from_dlpack_cpu, _from_dlpack_gpu, _to_dlpack
from merlin.table.tensor_column import Device, TensorColumn


class TorchColumn(TensorColumn):
    """
    A SeriesLike column backed by Torch tensors
    """

    @classmethod
    def array_type(cls) -> Type:
        """
        The type of the arrays backing this column
        """
        return th.Tensor

    @classmethod
    def array_constructor(cls) -> Callable:
        return th.tensor

    @classmethod
    def supported_devices(cls):
        """
        List of device types supported by this column type
        """
        return [Device.CPU, Device.GPU]

    def __init__(
        self, values: "th.Tensor", offsets: "th.Tensor" = None, dtype=None, _ref=None, _unsafe=False
    ):
        values_device = self._th_device(values)
        if offsets is not None:
            offsets_device = self._th_device(offsets)
            if values_device != offsets_device:
                raise ValueError(
                    f"Values and offsets were detected on different devices: "
                    f"values ({values_device}) and offsets ({offsets_device})."
                )

        super().__init__(values, offsets, dtype, _device=values_device, _ref=_ref, _unsafe=_unsafe)

    def cpu(self):
        """
        Move this column's data to host (i.e. CPU) memory

        Returns
        -------
        TorchColumn
            A copy of this column backed by Torch CPU tensors
        """
        if self.device is Device.CPU:
            return self

        values = self.values.cpu()
        offsets = self.offsets.cpu() if self.offsets is not None else None

        return TorchColumn(values, offsets)

    def gpu(self):
        """
        Move this column's data to device (i.e. GPU) memory

        Returns
        -------
        TorchColumn
            A copy of this column backed by Torch GPU tensors
        """
        if self.device is Device.GPU:
            return self

        values = self.values.cuda()
        offsets = self.offsets.cuda() if self.offsets is not None else None

        return TorchColumn(values, offsets)

    @property
    def device(self) -> Device:
        return self._th_device(self.values)

    @property
    def _flatten_values(self):
        return th.flatten(self.values)

    def _reshape_values(self, values, shape):
        return th.reshape(values, shape)

    def _th_device(self, tensor):
        return Device.GPU if tensor.is_cuda else Device.CPU


@_to_dlpack.register_lazy("torch")
def _register_to_dlpack_from_torch():
    import torch as th

    @_to_dlpack.register(th.Tensor)
    def _to_dlpack_from_torch_tensor(tensor):
        return tensor


@_from_dlpack_cpu.register_lazy("torch")
def _register_from_dlpack_cpu_to_torch():
    import torch as th

    @_from_dlpack_cpu.register(th.Tensor)
    def _from_dlpack_cpu_to_torch(target_type, array):
        return th.utils.dlpack.from_dlpack(array)


@_from_dlpack_gpu.register_lazy("torch")
def _register_from_dlpack_gpu_to_torch():
    import torch as th

    @_from_dlpack_gpu.register(th.Tensor)
    def _from_dlpack_gpu_to_torch(target_type, array):
        return th.utils.dlpack.from_dlpack(array.__dlpack__())
