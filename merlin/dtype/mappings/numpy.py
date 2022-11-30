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

from merlin.dtype import dtypes
from merlin.dtype.registry import _dtype_registry



def _numpy_dtype(raw_dtype):
    # Many Pandas dtypes have equivalent numpy dtypes
    if hasattr(raw_dtype, "numpy_dtype"):
        return np.dtype(raw_dtype.numpy_dtype)
    # cuDF categorical columns have varying element types
    elif hasattr(raw_dtype, "_categories"):
        return raw_dtype._categories.dtype
    # Rely on Numpy to do conversions from strings to dtypes (for now)
    elif isinstance(raw_dtype, str):
        return np.dtype(raw_dtype)
    # Tensorflow dtypes can convert themselves (in case we missed a mapping)
    elif hasattr(raw_dtype, "as_numpy_dtype"):
        return raw_dtype.as_numpy_dtype


numpy_dtypes = {
    # Unsigned Integer
    dtypes.uint8: [np.dtype("uint8"), np.uint8],
    dtypes.uint16: [np.dtype("uint16"), np.uint16],
    dtypes.uint32: [np.dtype("uint32"), np.uint32],
    dtypes.uint64: [np.dtype("uint64"), np.uint64],
    # Signed integer
    dtypes.int8: [np.dtype("int8"), np.int8],
    dtypes.int16: [np.dtype("int16"), np.int16],
    dtypes.int32: [np.dtype("int32"), np.int32],
    dtypes.int64: [np.dtype("int64"), np.int64],
    # Floating Point
    dtypes.float16: [np.dtype("float16"), np.float16],
    dtypes.float32: [np.dtype("float32"), np.float32],
    dtypes.float64: [np.dtype("float64"), np.float64],
    # Date/Time
    dtypes.datetime64: [np.dtype("datetime64"), np.datetime64],
    dtypes.datetime64Y: [np.dtype("datetime64[Y]")],
    dtypes.datetime64M: [np.dtype("datetime64[M]")],
    dtypes.datetime64D: [np.dtype("datetime64[D]")],
    dtypes.datetime64h: [np.dtype("datetime64[h]")],
    dtypes.datetime64m: [np.dtype("datetime64[m]")],
    dtypes.datetime64s: [np.dtype("datetime64[s]")],
    dtypes.datetime64ms: [np.dtype("datetime64[ms]")],
    dtypes.datetime64us: [np.dtype("datetime64[us]")],
    dtypes.datetime64ns: [np.dtype("datetime64[ns]")],
    # Miscellaneous
    dtypes.string: [np.dtype("str"), np.str],
    dtypes.object_: [np.dtype("O")],
    dtypes.boolean: [np.dtype("bool"), np.bool],
}
_dtype_registry.register("numpy", numpy_dtypes)
