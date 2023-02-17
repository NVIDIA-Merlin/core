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
from merlin.core.compat import cupy as cp
from merlin.table.conversions import _from_dlpack_gpu, _to_dlpack
from merlin.table.tensor_column import Device, TensorColumn


class CupyColumn(TensorColumn):
    @classmethod
    def array_type(cls):
        return cp.ndarray

    @classmethod
    def supported_devices(cls):
        return [Device.GPU]

    @classmethod
    def cast(cls, other):
        column = cls(to_cupy(other.values), to_cupy(other.offsets))
        column._ref = (other.values, other.offsets)
        return column

    def __init__(self, values: cp.ndarray, offsets: cp.ndarray = None, dtype=None, _ref=None):
        super().__init__(values, offsets, dtype, _ref)

    @property
    def device(self) -> Device:
        return Device.GPU


@_to_dlpack.register_lazy("cupy")
def register_to_dlpack_from_cupy():
    import cupy as cp

    @_to_dlpack.register(cp.ndarray)
    def _to_dlpack_from_tf_tensor(tensor):
        return tensor


@_from_dlpack_gpu.register_lazy("cupy")
def register_from_dlpack_gpu_to_cupy():
    import cupy as cp

    @_from_dlpack_gpu.register(cp.ndarray)
    def _from_dlpack_gpu_to_cupy(to, array):
        return cp.fromDlpack(array.__dlpack__())
