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
    import numba
    from numba.cuda.cudadrv.devicearray import DeviceNDArray as NumbaArray
except ImportError:
    numba = None
    NumbaArray = None

from merlin.array.base import MerlinArray
from merlin.array.protocols import NumbaConvertible


class MerlinNumpyArray(MerlinArray, NumbaConvertible):
    """
    Thin wrapper around a Numpy array that implements conversion via Numba.
    """

    @classmethod
    def _from_numba(cls, numba_array) -> "MerlinNumpyArray":
        return cls(numba_array.copy_to_host())

    def _to_numba(self) -> NumbaArray:
        return numba.cuda.to_device(self.data)
