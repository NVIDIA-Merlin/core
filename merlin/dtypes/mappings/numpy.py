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
import numpy as np

import merlin.dtypes.aliases as mn
from merlin.dtypes.registry import _dtype_registry

numpy_dtypes = {
    # Unsigned Integer
    mn.uint8: [np.dtype("uint8"), np.uint8],
    mn.uint16: [np.dtype("uint16"), np.uint16],
    mn.uint32: [np.dtype("uint32"), np.uint32],
    mn.uint64: [np.dtype("uint64"), np.uint64],
    # Signed integer
    mn.int8: [np.dtype("int8"), np.int8],
    mn.int16: [np.dtype("int16"), np.int16],
    mn.int32: [np.dtype("int32"), np.int32],
    mn.int64: [np.dtype("int64"), np.int64],
    # Floating Point
    mn.float16: [np.dtype("float16"), np.float16],
    mn.float32: [np.dtype("float32"), np.float32],
    mn.float64: [np.dtype("float64"), np.float64],
    # Date/Time
    mn.datetime64: [np.dtype("datetime64"), np.datetime64],
    mn.datetime64Y: [np.dtype("datetime64[Y]")],
    mn.datetime64M: [np.dtype("datetime64[M]")],
    mn.datetime64D: [np.dtype("datetime64[D]")],
    mn.datetime64h: [np.dtype("datetime64[h]")],
    mn.datetime64m: [np.dtype("datetime64[m]")],
    mn.datetime64s: [np.dtype("datetime64[s]")],
    mn.datetime64ms: [np.dtype("datetime64[ms]")],
    mn.datetime64us: [np.dtype("datetime64[us]")],
    mn.datetime64ns: [np.dtype("datetime64[ns]")],
    # Miscellaneous
    mn.string: [np.dtype("str"), np.str],
    mn.object_: [np.dtype("O"), np.object],
    mn.boolean: [np.dtype("bool"), np.bool],
}
_dtype_registry.register("numpy", numpy_dtypes)
