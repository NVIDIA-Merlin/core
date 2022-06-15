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
from typing import Callable, Dict

from merlin.features.array.interfaces import (
    CudaArrayInterface,
    DLPackInterface,
    NumpyArrayInterface,
)


class MerlinArray(ABC):
    """
    Base class for Merlin array wrapper classes that implement cross-framework array conversions.
    """

    array_types: Dict[type, type] = {}
    array_interfaces: Dict[type, Callable] = {}

    def __init__(self, array):
        self._ref = array
        self.array = self._build_from(array)

    @classmethod
    def __init_subclass__(cls) -> None:
        # Create a lookup table that matches array types (e.g. cupy.ndarray)
        # with MerlinArray sub-classes (e.g. CupyMerlinArray)
        if hasattr(cls, "array_type"):
            cls.array_types[cls.array_type()] = cls

        # Since each sub-class defines its own implementation of these methods
        # for building from various interoperability interfaces, we need to
        # populate this lookup table separately for each sub-class
        cls.array_interfaces = {
            CudaArrayInterface: cls.build_from_cuda_array,
            DLPackInterface: cls.build_from_dlpack,
            NumpyArrayInterface: cls.build_from_array,
        }

        return super().__init_subclass__()

    # MerlinArray.build(array_object) -> MerlinCupyArray
    @classmethod
    def build(cls, other):
        for array_type, array_class in cls.array_types.items():
            if isinstance(other, array_type):
                return array_class(other)

        raise TypeError(f"Unknown type of array: {type(other)}")

    # MerlinCupyArray(array_object)
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
        cls = self.__class__

        # First, try directly indexing into the dictionary to find the right type
        for array_type, array_class in cls.array_types.items():
            if isinstance(other, array_type):
                try:
                    return self.build_from_cuda_array(array_class.convert_to_cuda_array(other))
                except (NotImplementedError, TypeError):
                    ...

                try:
                    return self.build_from_dlpack_capsule(
                        array_class.convert_to_dlpack_capsule(other)
                    )
                except (NotImplementedError, TypeError):
                    ...

                try:
                    return self.build_from_array(array_class.convert_to_array(other))
                except (NotImplementedError, TypeError):
                    ...

        # If that fails, then check against the interfaces
        for array_interface, array_method in cls.array_interfaces.items():
            if isinstance(other, array_interface):
                return array_method(cls, other)

        raise TypeError(
            f"Can't create {cls} array from type {type(other)}, "
            "which doesn't support any of the available conversion interfaces."
        )

    @classmethod
    @abstractmethod
    def array_type(cls) -> type:
        ...

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

    @classmethod
    @abstractmethod
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
    @abstractmethod
    def build_from_dlpack_capsule(cls, capsule):
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
