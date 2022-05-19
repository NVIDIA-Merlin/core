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

try:
    import cupy as cp
except ImportError:
    cupy = None

try:
    import numba
    from numba.cuda.cudadrv.devicearray import DeviceNDArray as NumbaArray
except ImportError:
    numba = None
    NumbaArray = None

from merlin.array.base import MerlinArray
from merlin.array.protocols import CudaArrayConvertible, DlpackConvertible, NumbaConvertible


class MerlinCupyArray(MerlinArray, CudaArrayConvertible, DlpackConvertible, NumbaConvertible):
    """
    Thin wrapper around a Cupy array that implements conversion
    via CUDA array interface, DLPack, and Numba.
    """

    @classmethod
    def _from_cuda_array(cls, cuda_array) -> "MerlinArray":
        return cls(cp.asarray(cuda_array))

    def _to_cuda_array(self):
        return self.data

    @classmethod
    def _from_dlpack(cls, dlpack_capsule) -> "MerlinCupyArray":
        return cls(cp.fromDlpack(dlpack_capsule))

    def _to_dlpack(self):
        return self.data.toDlpack()

    @classmethod
    def _from_numba(cls, numba_array) -> "MerlinCupyArray":
        return cls(cp.asarray(numba_array))

    def _to_numba(self) -> NumbaArray:
        return numba.cuda.to_device(self.data)
