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
from abc import ABC, abstractmethod

from merlin.features.array import CudaArrayInterface, DLPackInterface, NumpyArrayInterface

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import cudf
except ImportError:
    cudf = None


class MerlinArray(ABC):
    """
    Base class for Merlin array wrapper classes that implement cross-framework array conversions.
    """

    def __init__(self, array):
        self.array = self._build_from(array)

    def _build_from(self, other):
        """
        Build a MerlinArray sub-class object from an array-like object.

        Parameters
        ----------
        other : "array-like"
            An array-like object from a framework such as Numpy, CuPy, cuDF, Tensorflow, etc

        Returns
        -------
        MerlinArray
            A MerlinArray wrapping a framework array converted
            from the source framework to the subclass framework.

        Raises
        ------
        TypeError
            If no appropriate conversion interface can be identified.
        """
        interface_methods = {}

        if cp is not None:
            interface_methods[cp.ndarray] = self.build_from_cp_array

        if tf is not None:
            interface_methods[tf.Tensor] = self.build_from_tf_tensor

        if cudf is not None:
            interface_methods[cudf.Series] = self.build_from_cudf_series

        interface_methods.update(
            {
                CudaArrayInterface: self.build_from_cuda_array,
                DLPackInterface: self.build_from_dlpack,
                NumpyArrayInterface: self.build_from_array,
            }
        )

        for interface, method in interface_methods.items():
            if isinstance(other, interface):
                try:
                    return method(other)
                except NotImplementedError:
                    continue

        raise TypeError(
            f"Can't create {self.__class__} array from type {type(other)}, "
            "which doesn't support any of the available conversion interfaces."
        )

    @abstractmethod
    def build_from_cuda_array(self, other: CudaArrayInterface):
        """
        Build a MerlinArray from an array-like object that implements the CUDA Array Interface

        Parameters
        ----------
        other : CudaArrayInterface
            An array-like object that implements the CUDA Array Interface
        """
        ...

    @abstractmethod
    def build_from_array(self, other: NumpyArrayInterface):
        """
        Build a MerlinArray from an array-like object that implements the Numpy Array Interface

        Parameters
        ----------
        other : NumpyArrayInterface
            An array-like object that implements the Numpy Array Interface
        """
        ...

    @abstractmethod
    def build_from_dlpack_capsule(self, capsule):
        """
        Build a MerlinArray from a PyCapsule object created with the DLPack interface

        Parameters
        ----------
        other : PyCapsule
            A PyCapsule object created with the DLPack interface
        """
        ...

    def build_from_dlpack(self, other):
        """
        Build a MerlinArray from an array-like object that implements the DLPack Standard

        Parameters
        ----------
        other : PyCapsule
            An array-like object that implements the DLPack Standard
        """
        return self.build_from_dlpack_capsule(other.__dlpack__())

    def build_from_tf_tensor(self, other):
        """build_from_tf_tensor"""
        try:
            capsule = tf.experimental.dlpack.to_dlpack(other)
            return self.build_from_dlpack_capsule(capsule)
        except NotImplementedError:
            ...

        return self.build_from_array(other.numpy())

    def build_from_cp_array(self, other):
        """build_from_cp_array"""
        try:
            return self.build_from_dlpack_capsule(other.toDlpack())
        except NotImplementedError:
            ...

        return self.build_from_array(cp.asnumpy(other))

    def build_from_cudf_series(self, other):
        """build_from_cudf_series"""
        try:
            return self.build_from_dlpack_capsule(other.to_dlpack())
        except NotImplementedError:
            ...

        return self.build_from_array(other.to_numpy())
