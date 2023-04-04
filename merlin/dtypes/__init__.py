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

# flake8: noqa
from merlin.dtypes import mappings
from merlin.dtypes.aliases import *
from merlin.dtypes.base import DType
from merlin.dtypes.registry import _dtype_registry
from merlin.dtypes.shape import Dimension, Shape

# Convenience alias for registering dtypes
register = _dtype_registry.register


def dtype(external_dtype):
    # If the supplied dtype is None, then there's not a default dtype we can
    # universally translate to across frameworks, so raise an error and help
    # the downstream developer figure out how to handle that case explicitly
    if external_dtype is None:
        raise TypeError(
            "Merlin doesn't provide a default dtype mapping for `None`. "
            "This differs from the Numpy behavior you may be expecting, "
            "which treats `None` as an alias for `np.float64`. If you're "
            "expecting this dtype to be non-`None`, there may be an issue "
            "in upstream code. If you'd like to allow this dtype to be `None`, "
            "you can use a `try/except` to catch this error."
        )

    # If the supplied dtype is already a Merlin dtype, then there's
    # nothing for us to do and we can exit early
    if isinstance(external_dtype, DType):
        return external_dtype

    # If not, attempt to apply all the registered Merlin dtype mappings.
    # If we don't find a match with those, fall back on converting to
    # a numpy dtype and trying to match that instead.
    try:
        return _dtype_registry.to_merlin(external_dtype)
    except TypeError as base_exc:
        try:
            return _dtype_registry.to_merlin_via_numpy(external_dtype)
        except TypeError as exc:
            # If we fail to find a match even after we try converting to
            # numpy, re-raise the original exception because it has more
            # information about the original external dtype that's causing
            # the problem. (We want to highlight that dtype, not whatever
            # numpy dtype it was converted to in the interim.)
            raise base_exc from exc
