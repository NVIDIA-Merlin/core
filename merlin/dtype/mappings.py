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

from merlin.core.compat import tritonclient
from merlin.dtype import dtypes
from merlin.dtype.registry import _dtype_registry


python_dtypes = {
    dtypes.boolean: bool,
    dtypes.int64: int,
    dtypes.float64: float,
    dtypes.string: str,
}
_dtype_registry.register("python", python_dtypes)


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


# Only define a Triton dtype mapping if `tritonclient` is available
try:
    import tritonclient.grpc.model_config_pb2 as model_config

    triton_dtypes = {
        # Unsigned Integer
        dtypes.uint8: [model_config.TYPE_UINT8],
        dtypes.uint16: [model_config.TYPE_UINT16],
        dtypes.uint32: [model_config.TYPE_UINT32],
        dtypes.uint64: [model_config.TYPE_UINT64],

        # Signed integer
        dtypes.int8: [model_config.TYPE_INT8],
        dtypes.int16: [model_config.TYPE_INT16],
        dtypes.int32: [model_config.TYPE_INT32],
        dtypes.int64: [model_config.TYPE_INT64],
        
        # Floating Point
        dtypes.float16: [model_config.TYPE_FP16],
        dtypes.float32: [model_config.TYPE_FP32,],
        dtypes.float64: [model_config.TYPE_FP64],
        
        # Miscellaneous
        dtypes.string: [model_config.TYPE_STRING],
        dtypes.boolean: [model_config.TYPE_BOOL],
    }
    _dtype_registry.register("triton", triton_dtypes)
except ImportError:
    pass
