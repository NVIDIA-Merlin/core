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
from merlin.core.compat import numpy as np
from merlin.table.conversions import _from_dlpack_cpu, _to_dlpack
from merlin.table.tensor_column import Device, TensorColumn


class NumpyColumn(TensorColumn):
    """
    A SeriesLike column backed by NumPy arrays
    """

    @classmethod
    def array_type(cls) -> Type:
        """
        The type of the arrays backing this column
        """
        return np.ndarray

    @classmethod
    def array_constructor(cls) -> Callable:
        return np.array

    @classmethod
    def supported_devices(cls):
        """
        List of device types supported by this column type
        """
        return [Device.CPU]

    def __init__(
        self,
        values: "np.ndarray",
        offsets: "np.ndarray" = None,
        dtype=None,
        _ref=None,
        _unsafe=False,
    ):
        super().__init__(values, offsets, dtype, _ref=_ref, _device=Device.CPU, _unsafe=_unsafe)

    def cpu(self):
        """
        Move this column's data to host (i.e. CPU) memory

        Returns
        -------
        NumpyColumn
            This column, unchanged and backed by NumPy arrays
        """
        return self

    def gpu(self):
        """
        Move this column's data to device (i.e. GPU) memory

        Returns
        -------
        CupyColumn
            A copy of this column backed by CuPy arrays
        """

        from merlin.table import CupyColumn

        values = cp.asarray(self.values)
        offsets = cp.asarray(self.offsets) if self.offsets is not None else None

        return CupyColumn(values, offsets)

    @property
    def _flatten_values(self):
        return self.values.flatten()

    def _reshape_values(self, values, shape):
        return np.reshape(values, shape)


@_from_dlpack_cpu.register_lazy("numpy")
def _register_from_dlpack_cpu_to_numpy():
    import numpy as np

    @_from_dlpack_cpu.register(np.ndarray)
    def _from_dlpack_cpu_to_numpy(to, array):
        try:
            # private `_from_dlpack` method added in 1.22.0
            return np._from_dlpack(array)
        except AttributeError:
            pass
        try:
            # public `from_dlpack` method added in 1.23.0
            return np.from_dlpack(array)
        except AttributeError as exc:
            raise NotImplementedError(
                "NumPy does not implement the DLPack Standard until version 1.22.0, "
                f"currently running {np.__version__}"
            ) from exc


@_to_dlpack.register_lazy("numpy")
def _register_from_numpy_to_dlpack_cpu():
    import numpy as np

    @_to_dlpack.register(np.ndarray)
    def _to_dlpack_cpu_from_numpy(array):
        if array.dtype == np.dtype("bool"):
            array = array.astype(np.dtype("int8"))
        return array
