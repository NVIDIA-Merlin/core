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
import cupy as cp

from merlin.features.array.base import MerlinArray


class MerlinCupyArray(MerlinArray):
    """MerlinCupyArray"""

    def build_from_cuda_array(self, other):
        """build_from_cuda_array"""
        return cp.asarray(other)

    def build_from_array(self, other):
        """build_from_array"""
        return cp.asarray(other)

    def build_from_dlpack_capsule(self, capsule):
        """build_from_dlpack_capsule"""
        try:
            return cp.from_dlpack(capsule)
        except AttributeError:
            return cp.fromDlpack(capsule)
