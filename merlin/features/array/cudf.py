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
from merlin.features.array.compat import cudf


class _MerlinCudfArray(MerlinArray):
    """
    Thin wrapper around a cudf.Series that can be constructed from other framework types.
    """

    @classmethod
    def array_type(cls):
        return cudf.Series

    @classmethod
    def convert_to_array(cls, other):
        return other.to_numpy()

    @classmethod
    def convert_to_cuda_array(cls, other):
        raise NotImplementedError

    @classmethod
    def convert_to_dlpack_capsule(cls, other):
        return other.to_dlpack()

    def build_from_cuda_array(self, other):
        """
        Build a cudf.Series from an object that implements the Cuda Array Interface.

        Parameters
        ----------
        other : array-like
            The array-like object to build the Series from

        Returns
        -------
        cudf.Series
            The Series built from the array-like object
        """
        return cudf.Series(other)

    def build_from_array(self, other):
        """
        Build a cudf.Series from an object that implements the Numpy Array Interface.

        Parameters
        ----------
        other : array-like
            The array-like object to build the Series from

        Returns
        -------
        cudf.Series
            The Series built from the array-like object
        """
        return cudf.Series(other)

    def build_from_dlpack_capsule(self, capsule):
        """
        Build a cudf.Series from an object that implements the DLPack Standard.

        Parameters
        ----------
        other : array-like
            The array-like object to build the Series from

        Returns
        -------
        cudf.Series
            The Series built from the array-like object
        """
        return cudf.io.from_dlpack(capsule)


MerlinCudfArray = None if not cudf else _MerlinCudfArray
