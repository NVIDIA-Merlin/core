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
    import tensorflow.experimental.dlpack as tf_dlpack
except ImportError:
    tf_dlpack = None

from merlin.array.base import DlpackConvertible, MerlinArray


class MerlinTensorflowArray(MerlinArray, DlpackConvertible):
    """
    Thin wrapper around a Tensorflow tensor that implements conversion via DLPack.
    """

    @classmethod
    def _from_dlpack(cls, dlpack_capsule) -> "MerlinTensorflowArray":
        return cls(tf_dlpack.from_dlpack(dlpack_capsule))

    def _to_dlpack(self):
        return tf_dlpack.to_dlpack(self.data)
