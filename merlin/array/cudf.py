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
    import cudf
except ImportError:
    cudf = None

from merlin.array.base import MerlinArray
from merlin.array.protocols import CudaArrayConvertible, DlpackConvertible


class MerlinCudfArray(MerlinArray, DlpackConvertible, CudaArrayConvertible):
    """
    Thin wrapper around a CuDF Series that implements conversion via DLPack.
    """

    @classmethod
    def _from_dlpack(cls, dlpack_capsule) -> "MerlinCudfArray":
        return cls(cudf.io.from_dlpack(dlpack_capsule))

    def _to_dlpack(self):
        return self.data.to_dlpack()

    @classmethod
    def _from_cuda_array(cls, cuda_array) -> "MerlinArray":
        return cls(cudf.Series(cuda_array))

    def _to_cuda_array(self):
        return self.data.values
