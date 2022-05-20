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
from merlin.array.interfaces import (
    ArrayInterface,
    CudaArrayInterface,
    DLPackInterface,
    NumpyArrayInterface,
)

try:
    import tensorflow as tf
except ImportError:
    tf = None


class MerlinArray:
    """
    Base class for Merlin array wrapper classes that implement cross-framework array conversions.
    """

    def __init__(self, array):
        self.data = array

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
        if isinstance(other, CudaArrayInterface):
            return cls.build_from_cuda_array(other)
        elif isinstance(other, ArrayInterface):
            return cls.build_from_array(other)
        elif isinstance(other, DLPackInterface):
            return cls.build_from_dlpack(other)
        elif tf is not None and isinstance(other, tf.Tensor):
            capsule = tf.experimental.dlpack.to_dlpack(other)
            return cls.build_from_dlpack_capsule(capsule)
        else:
            raise TypeError(
                f"Can't create {cls} array from type {type(other)}, "
                "which doesn't support any of the available conversion interfaces."
            )

    @classmethod
    def build_from_cuda_array(cls, other: CudaArrayInterface):
        """
        Build a MerlinArray from an array-like object that implements the CUDA Array Interface

        Parameters
        ----------
        other : CudaArrayInterface
            An array-like object that implements the CUDA Array Interface
        """
        ...

    @classmethod
    def build_from_array(cls, other: NumpyArrayInterface):
        """
        Build a MerlinArray from an array-like object that implements the Numpy Array Interface

        Parameters
        ----------
        other : NumpyArrayInterface
            An array-like object that implements the Numpy Array Interface
        """
        ...

    @classmethod
    def build_from_dlpack(cls, capsule):
        """
        Build a MerlinArray from an array-like object that implements the DLPack Standard

        Parameters
        ----------
        other : PyCapsule
            An array-like object that implements the DLPack Standard
        """
        ...

    @classmethod
    def build_from_dlpack_capsule(cls, capsule):
        """
        Build a MerlinArray from a PyCapsule object created with the DLPack interface

        Parameters
        ----------
        other : PyCapsule
            A PyCapsule object created with the DLPack interface
        """
        ...
