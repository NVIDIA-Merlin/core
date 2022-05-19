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
from merlin.array.convertible import CudaArrayConvertible, DlpackConvertible, NumbaConvertible

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
