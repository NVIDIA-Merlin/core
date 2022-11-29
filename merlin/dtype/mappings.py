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
from merlin.dtype.registry import DTypeMapping, _dtype_registry

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

try:
    import pandas as pd

    pandas_dtypes = DTypeMapping(
        {
            dtypes.string: [pd.StringDtype],
            dtypes.boolean: [pd.BooleanDtype],
        },
        preprocessing_lambda=lambda dtype: dtype.numpy_dtype,
    )
    _dtype_registry.register("pandas", pandas_dtypes)
except ImportError:
    pass


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
        dtypes.float32: [
            model_config.TYPE_FP32,
        ],
        dtypes.float64: [model_config.TYPE_FP64],
        # Miscellaneous
        dtypes.string: [model_config.TYPE_STRING],
        dtypes.boolean: [model_config.TYPE_BOOL],
    }
    _dtype_registry.register("triton", triton_dtypes)
except ImportError:
    pass



try:
    from torch import uint8, int8, int16, int32, int64
    from torch import float16, float32, float64
    from torch import bool as bool_

    torch_dtypes = {
        # Unsigned Integer
        dtypes.uint8: [uint8],
        # Signed integer
        dtypes.int8: [int8],
        dtypes.int16: [int16],
        dtypes.int32: [int32],
        dtypes.int64: [int64],
        # Floating Point
        dtypes.float16: [float16],
        dtypes.float32: [float32],
        dtypes.float64: [float64],
        # Miscellaneous
        dtypes.boolean: [bool_],
    }
    _dtype_registry.register("torch", torch_dtypes)
    _dtype_registry.register("pytorch", torch_dtypes)
except ImportError as exc:
    from warnings import warn

    warn(f"PyTorch dtype mappings did not load successfully due to this error: {exc.msg}")    

try:
    from tensorflow import dtypes as tf_dtypes

    tf_dtypes = DTypeMapping({
        # Unsigned Integer
        dtypes.uint8: [tf_dtypes.uint8],
        dtypes.uint16: [tf_dtypes.uint16],
        dtypes.uint32: [tf_dtypes.uint32],
        dtypes.uint64: [tf_dtypes.uint64],
        # Signed integer
        dtypes.int8: [tf_dtypes.int8],
        dtypes.int16: [tf_dtypes.int16],
        dtypes.int32: [tf_dtypes.int32],
        dtypes.int64: [tf_dtypes.int64],
        # Floating Point
        dtypes.float16: [tf_dtypes.float16],
        dtypes.float32: [tf_dtypes.float32],
        dtypes.float64: [tf_dtypes.float64],
        # Miscellaneous
        dtypes.boolean: [tf_dtypes.bool],
    }, tf_dtypes.DType)
    _dtype_registry.register("tf", tf_dtypes)
    _dtype_registry.register("tensorflow", tf_dtypes)
except ImportError:
    from warnings import warn

    warn(f"Tensorflow dtype mappings did not load successfully due to this error: {exc.msg}")    
