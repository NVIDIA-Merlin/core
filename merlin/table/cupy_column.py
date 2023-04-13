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

from merlin.core.compat import cupy as cp
from merlin.table.conversions import _from_dlpack_gpu, _to_dlpack
from merlin.table.tensor_column import Device, TensorColumn


class CupyColumn(TensorColumn):
    """
    A SeriesLike column backed by CuPy arrays
    """

    @classmethod
    def array_type(cls) -> Type:
        """
        The type of the arrays backing this column
        """
        return cp.ndarray

    @classmethod
    def array_constructor(cls) -> Callable:
        return cp.asarray

    @classmethod
    def supported_devices(cls):
        """
        List of device types supported by this column type
        """
        return [Device.GPU]

    def __init__(
        self,
        values: "cp.ndarray",
        offsets: "cp.ndarray" = None,
        dtype=None,
        _ref=None,
        _unsafe=False,
    ):
        super().__init__(values, offsets, dtype, _ref=_ref, _device=Device.GPU, _unsafe=_unsafe)

    def cpu(self):
        """
        Move this column's data to host (i.e. CPU) memory

        Returns
        -------
        NumpyColumn
            A copy of this column backed by NumPy arrays
        """
        from merlin.table import NumpyColumn

        values = cp.asnumpy(self.values)
        offsets = cp.asnumpy(self.offsets) if self.offsets is not None else None

        return NumpyColumn(values, offsets)

    def gpu(self):
        """
        Move this column's data to device (i.e. GPU) memory

        Returns
        -------
        CupyColumn
            This column, unchanged and backed by CuPy arrays
        """
        return self

    @property
    def _flatten_values(self):
        return self.values.flatten()

    def _reshape_values(self, values, shape):
        return cp.reshape(values, shape)


@_to_dlpack.register_lazy("cupy")
def _register_to_dlpack_from_cupy():
    import cupy as cp

    @_to_dlpack.register(cp.ndarray)
    def _to_dlpack_from_cp_tensor(tensor):
        if tensor.dtype == cp.dtype("bool"):
            tensor = tensor.astype(cp.dtype("int8"))
        return tensor


@_from_dlpack_gpu.register_lazy("cupy")
def _register_from_dlpack_gpu_to_cupy():
    import cupy as cp

    @_from_dlpack_gpu.register(cp.ndarray)
    def _from_dlpack_gpu_to_cupy(to, array) -> cp.ndarray:
        return cp.from_dlpack(array)
