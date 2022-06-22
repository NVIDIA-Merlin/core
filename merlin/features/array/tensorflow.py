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
from merlin.features.array.base import ArrayPackageNotInstalled, MerlinArray
from merlin.features.array.compat import tensorflow

if not tensorflow:

    class _TensorflowNotInstalled(ArrayPackageNotInstalled):
        @classmethod
        def package_name(cls):
            return "tensorflow"

else:

    class _MerlinTensorflowArray(MerlinArray):
        """
        Thin wrapper around a tensorflow.Tensor that can be constructed from other framework types.
        """

        @classmethod
        def array_type(cls):
            return tensorflow.Tensor

        @classmethod
        def convert_to_array(cls, other):
            return other.numpy()

        @classmethod
        def convert_to_cuda_array(cls, other):
            raise NotImplementedError

        @classmethod
        def convert_to_dlpack_capsule(cls, other):
            return tensorflow.experimental.dlpack.to_dlpack(other)

        def build_from_cuda_array(self, other):
            """
            Build a tf.Tensor from an object that implements the Cuda Array Interface.

            Parameters
            ----------
            other : array-like
                The array-like object to build the Tensor from

            Returns
            -------
            tf.Tensor
                The Tensor built from the array-like object
            """
            raise NotImplementedError("Tensorflow does not implement the CUDA Array Interface")

        def build_from_array(self, other):
            """
            Build a tf.Tensor from an object that implements the Numpy Array Interface.

            Parameters
            ----------
            other : array-like
                The array-like object to build the Tensor from

            Returns
            -------
            tf.Tensor
                The Tensor built from the array-like object
            """
            return tensorflow.convert_to_tensor(other)

        def build_from_dlpack_capsule(self, capsule):
            """
            Build a tf.Tensor from an object that implements the DLPack Standard.

            Parameters
            ----------
            other : array-like
                The array-like object to build the Tensor from

            Returns
            -------
            tf.Tensor
                The Tensor built from the array-like object
            """
            return tensorflow.experimental.dlpack.from_dlpack(capsule)


# This makes mypy type checking work by avoiding
# duplicate definitions of MerlinCudfArray
MerlinTensorflowArray = _MerlinTensorflowArray if tensorflow else _TensorflowNotInstalled
