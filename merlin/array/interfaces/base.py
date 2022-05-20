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
from abc import ABC, abstractclassmethod

from merlin.array.interfaces import CudaArrayInterface, DLPackInterface, NumpyArrayInterface

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
        self.data = self.__class__.build_from(array)

    @classmethod
    def build_from(cls, other):
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
            interface_methods[cp.ndarray] = cls.build_from_cp_array

        if tf is not None:
            interface_methods[tf.Tensor] = cls.build_from_tf_tensor

        if cudf is not None:
            interface_methods[cudf.Series] = cls.build_from_cudf_series

        interface_methods.update(
            {
                CudaArrayInterface: cls.build_from_cuda_array,
                DLPackInterface: cls.build_from_dlpack,
                NumpyArrayInterface: cls.build_from_array,
            }
        )

        for interface, method in interface_methods.items():
            if isinstance(other, interface):
                try:
                    return method(other)
                except NotImplementedError:
                    continue

        raise TypeError(
            f"Can't create {cls} array from type {type(other)}, "
            "which doesn't support any of the available conversion interfaces."
        )

    @abstractclassmethod
    def build_from_cuda_array(cls, other: CudaArrayInterface):
        """
        Build a MerlinArray from an array-like object that implements the CUDA Array Interface

        Parameters
        ----------
        other : CudaArrayInterface
            An array-like object that implements the CUDA Array Interface
        """
        ...

    @abstractclassmethod
    def build_from_array(cls, other: NumpyArrayInterface):
        """
        Build a MerlinArray from an array-like object that implements the Numpy Array Interface

        Parameters
        ----------
        other : NumpyArrayInterface
            An array-like object that implements the Numpy Array Interface
        """
        ...

    @abstractclassmethod
    def build_from_dlpack_capsule(cls, capsule):
        """
        Build a MerlinArray from a PyCapsule object created with the DLPack interface

        Parameters
        ----------
        other : PyCapsule
            A PyCapsule object created with the DLPack interface
        """
        ...

    @classmethod
    def build_from_dlpack(cls, other):
        """
        Build a MerlinArray from an array-like object that implements the DLPack Standard

        Parameters
        ----------
        other : PyCapsule
            An array-like object that implements the DLPack Standard
        """
        return cls.build_from_dlpack_capsule(other.__dlpack__())

    @classmethod
    def build_from_tf_tensor(cls, other):
        """build_from_tf_tensor"""
        try:
            capsule = tf.experimental.dlpack.to_dlpack(other)
            return cls.build_from_dlpack_capsule(capsule)
        except NotImplementedError:
            ...

        return cls.build_from_array(other.numpy())

    @classmethod
    def build_from_cp_array(cls, other):
        """build_from_cp_array"""
        try:
            return cls.build_from_dlpack_capsule(other.toDlpack())
        except NotImplementedError:
            ...

        return cls.build_from_array(cp.asnumpy(other))

    @classmethod
    def build_from_cudf_series(cls, other):
        """build_from_cudf_series"""
        try:
            return cls.build_from_dlpack_capsule(other.to_dlpack())
        except NotImplementedError:
            ...

        return cls.build_from_array(other.to_numpy())
