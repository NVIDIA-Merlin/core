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
import cupy as cp

from merlin.array.interfaces.base import MerlinArray


class MerlinCupyArray(MerlinArray):
    """
    _summary_

    Parameters
    ----------
    MerlinArray : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    @classmethod
    def build_from_cuda_array(cls, other):
        """
        _summary_

        Parameters
        ----------
        other : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return cls(cp.asarray(other))

    @classmethod
    def build_from_array(cls, other):
        """
        _summary_

        Parameters
        ----------
        other : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return cls(cp.asarray(other))

    @classmethod
    def build_from_dlpack(cls, other):
        """
        _summary_

        Parameters
        ----------
        other : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return cls(cp.fromDlpack(other))

    @classmethod
    def build_from_dlpack_capsule(cls, capsule):
        """
        _summary_

        Parameters
        ----------
        capsule : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return cls(cp.fromDlpack(capsule))
