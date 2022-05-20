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
import tensorflow as tf
import tensorflow.experimental.dlpack as tf_dlpack

from merlin.array.interfaces.base import MerlinArray


class MerlinTensorflowArray(MerlinArray):
    """
    _summary_

    Parameters
    ----------
    MerlinArray : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    @classmethod
    def build_from_cuda_array(cls, other):
        """build_from_cuda_array"""
        raise NotImplementedError("Tensorflow does not implement the CUDA Array Interface")

    @classmethod
    def build_from_array(cls, other):
        """build_from_array"""
        return tf.convert_to_tensor(other)

    @classmethod
    def build_from_dlpack_capsule(cls, capsule):
        """build_from_dlpack_capsule"""
        return tf_dlpack.from_dlpack(capsule)
