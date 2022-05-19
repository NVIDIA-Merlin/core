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
from typing import Protocol, runtime_checkable

from numba.cuda.cudadrv.devicearray import DeviceNDArray as NumbaArray


@runtime_checkable
class CudaArrayConvertible(Protocol):
    """
    Dlpack is the preferred intermediate format for converting array data
    between frameworks, since it provides zero-copy data transfer.
    """

    @classmethod
    def _from_cuda_array(cls, cuda_array):
        ...

    def _to_cuda_array(self):
        ...


@runtime_checkable
class DlpackConvertible(Protocol):
    """
    Dlpack is the preferred intermediate format for converting array data
    between frameworks, since it provides zero-copy data transfer.
    """

    @classmethod
    def _from_dlpack(cls, dlpack_capsule):
        ...

    def _to_dlpack(self):
        ...


@runtime_checkable
class NumbaConvertible(Protocol):
    """
    Numpy does not support dlpack until version 1.22.3, so Numba provides
    an alternate intermediate format for converting between array types
    where dlpack support is not available.
    """

    @classmethod
    def _from_numba(cls, numba_array):
        ...

    def _to_numba(self) -> NumbaArray:
        ...


CONVERSION_PROTOCOLS = {
    CudaArrayConvertible: ("_to_cuda_array", "_from_cuda_array"),
    DlpackConvertible: ("_to_dlpack", "_from_dlpack"),
    NumbaConvertible: ("_to_numba", "_from_numba"),
}
