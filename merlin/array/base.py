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
from typing import Protocol, runtime_checkable

import cudf
import cupy as cp
import numba
import tensorflow.experimental.dlpack as tf_dlpack
from numba.cuda.cudadrv.devicearray import DeviceNDArray as NumbaArray


@runtime_checkable
class CudaArrayConvertible(Protocol):
    """
    Dlpack is the preferred intermediate format for converting array data
    between frameworks, since it provides zero-copy data transfer.
    """

    @classmethod
    def _from_cuda_array(cls, cuda_array) -> "MerlinArray":
        ...

    def _to_cuda_array(self):
        ...


@runtime_checkable
class DlpackConvertible(Protocol):
    """
    Dlpack is the preferred intermediate format for converting array data
    between frameworks, since it provides zero-copy data transfer.
    """

    @classmethod
    def _from_dlpack(cls, dlpack_capsule) -> "MerlinArray":
        ...

    def _to_dlpack(self):
        ...


@runtime_checkable
class NumbaConvertible(Protocol):
    """
    Numpy does not support dlpack until version 1.22.3, so Numba provides
    an alternate intermediate format for converting between array types
    where dlpack support is not available.
    """

    @classmethod
    def _from_numba(cls, numba_array) -> "MerlinArray":
        ...

    def _to_numba(self) -> NumbaArray:
        ...


CONVERSION_PROTOCOLS = {
    CudaArrayConvertible: ("_to_cuda_array", "_from_cuda_array"),
    DlpackConvertible: ("_to_dlpack", "_from_dlpack"),
    NumbaConvertible: ("_to_numba", "_from_numba"),
}


class MerlinArray:
    """
    Base class for an array of data in the Merlin framework.

    Subclasses must support converting to other subclasses via one of the Protocols defined above.
    """

    def __init__(self, array):
        self.data = array

    def to(self, target_type: type):
        """
        Convert one kind of MerlinArray to another kind of MerlinArray, using
        whichever intermediate format is supported between both MerlinArray types.

        Parameters
        ----------
        target_type : type
            A subclass of MerlinArray that specifies the destination format.

        Returns
        -------
        MerlinArray
            A new MerlinArray instance with the data converted to the target format.

        Raises
        ------
        TypeError
            If there's no supported intermediate format between the two MerlinArray types,
            as determined by the Protocols implemented by each type.
        """
        for protocol, method_names in CONVERSION_PROTOCOLS.items():
            to_name, from_name = method_names
            if isinstance(self, protocol) and issubclass(target_type, protocol):
                to_method = getattr(self, to_name)
                from_method = getattr(target_type, from_name)
                return from_method(to_method())

        raise TypeError(
            f"Types {type(self)} and {target_type} have no interchange format "
            "directly compatible with both classes."
        )


class MerlinTensorflowArray(MerlinArray, DlpackConvertible):
    """
    Thin wrapper around a Tensorflow tensor that implements conversion via DLPack.
    """

    @classmethod
    def _from_dlpack(cls, dlpack_capsule) -> "MerlinTensorflowArray":
        return cls(tf_dlpack.from_dlpack(dlpack_capsule))

    def _to_dlpack(self):
        return tf_dlpack.to_dlpack(self.data)


class MerlinCudfArray(MerlinArray, DlpackConvertible, CudaArrayConvertible):
    """
    Thin wrapper around a CuDF Series that implements conversion via DLPack.
    """

    @classmethod
    def _from_dlpack(cls, dlpack_capsule) -> "MerlinCudfArray":
        return cls(cudf.io.from_dlpack(dlpack_capsule))

    def _to_dlpack(self):
        return self.data.to_dlpack()

    @classmethod
    def _from_cuda_array(cls, cuda_array) -> "MerlinArray":
        return cls(cudf.Series(cuda_array))

    def _to_cuda_array(self):
        return self.data.values


class MerlinCupyArray(MerlinArray, CudaArrayConvertible, DlpackConvertible, NumbaConvertible):
    """
    Thin wrapper around a Cupy array that implements conversion
    via CUDA array interface, DLPack, and Numba.
    """

    @classmethod
    def _from_cuda_array(cls, cuda_array) -> "MerlinArray":
        return cls(cp.asarray(cuda_array))

    def _to_cuda_array(self):
        return self.data

    @classmethod
    def _from_dlpack(cls, dlpack_capsule) -> "MerlinCupyArray":
        return cls(cp.fromDlpack(dlpack_capsule))

    def _to_dlpack(self):
        return self.data.toDlpack()

    @classmethod
    def _from_numba(cls, numba_array) -> "MerlinCupyArray":
        return cls(cp.asarray(numba_array))

    def _to_numba(self) -> NumbaArray:
        return numba.cuda.to_device(self.data)


class MerlinNumpyArray(MerlinArray, NumbaConvertible):
    """
    Thin wrapper around a Numpy array that implements conversion via Numba.
    """

    @classmethod
    def _from_numba(cls, numba_array) -> "MerlinNumpyArray":
        return cls(numba_array.copy_to_host())

    def _to_numba(self) -> NumbaArray:
        return numba.cuda.to_device(self.data)
