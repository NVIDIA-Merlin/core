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
from merlin.features.array.compat import cupy


class MerlinCupyArray(MerlinArray):
    """
    Thin wrapper around a cupy.ndarray that can be constructed from other framework types.
    """

    def build_from_cuda_array(self, other):
        """
        Build a cupy.ndarray from an object that implements the Cuda Array Interface.

        Parameters
        ----------
        other : array-like
            The array-like object to build the ndarray from

        Returns
        -------
        cupy.ndarray
            The ndarray built from the array-like object
        """
        return cupy.asarray(other)

    def build_from_array(self, other):
        """
        Build a cupy.ndarray from an object that implements the Numpy Array Interface.

        Parameters
        ----------
        other : array-like
            The array-like object to build the ndarray from

        Returns
        -------
        cupy.ndarray
            The ndarray built from the array-like object
        """
        return cupy.asarray(other)

    def build_from_dlpack_capsule(self, capsule):
        """
        Build a cupy.ndarray from a PyCapsule object created according to the DLPack Standard.

        Parameters
        ----------
        other : array-like
            The array-like object to build the ndarray from

        Returns
        -------
        cupy.ndarray
            The ndarray built from the array-like object
        """
        try:
            return cupy.from_dlpack(capsule)
        except AttributeError:
            return cupy.fromDlpack(capsule)
