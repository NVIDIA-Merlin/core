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
from merlin.features.array.base import MerlinArray
from merlin.features.array.compat import numpy

if numpy:

    class _MerlinNumpyArray(MerlinArray):
        """
        Thin wrapper around a numpy.ndarray that can be constructed from other framework types.
        """

        @classmethod
        def array_type(cls):
            return numpy.ndarray

        @classmethod
        def convert_to_array(cls, other):
            return other

        @classmethod
        def convert_to_cuda_array(cls, other):
            raise NotImplementedError

        @classmethod
        def convert_to_dlpack_capsule(cls, other):
            raise NotImplementedError

        def build_from_cuda_array(self, other):
            """
            Build a numpy.ndarray from an object that implements the Cuda Array Interface.

            Parameters
            ----------
            other : array-like
                The array-like object to build the ndarray from

            Returns
            -------
            numpy.ndarray
                The ndarray built from the array-like object
            """
            return numpy.array(other)

        def build_from_array(self, other):
            """
            Build a numpy.ndarray from an object that implements the Numpy Array Interface.

            Parameters
            ----------
            other : array-like
                The array-like object to build the ndarray from

            Returns
            -------
            numpy.ndarray
                The ndarray built from the array-like object
            """
            return numpy.array(other)

        def build_from_dlpack_capsule(self, capsule):
            """
            Build a numpy.ndarray from an object that implements the DLPack Standard.

            Parameters
            ----------
            other : array-like
                The array-like object to build the ndarray from

            Returns
            -------
            numpy.ndarray
                The ndarray built from the array-like object
            """
            try:
                return numpy._from_dlpack(capsule)
            except AttributeError as exc:
                raise NotImplementedError(
                    "NumPy does not implement the DLPack Standard until version 1.22.0, "
                    f"currently running {numpy.__version__}"
                ) from exc


MerlinNumpyArray = None if not numpy else _MerlinNumpyArray
