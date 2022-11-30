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
from merlin.dtype import dtypes
from merlin.dtype.registry import DTypeMapping, _dtype_registry

try:
    from tensorflow import dtypes as tf_dtypes

    tf_dtypes = DTypeMapping(
        {
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
        },
        base_class=tf_dtypes.DType,
    )
    _dtype_registry.register("tf", tf_dtypes)
    _dtype_registry.register("tensorflow", tf_dtypes)
except ImportError as exc:
    from warnings import warn

    warn(f"Tensorflow dtype mappings did not load successfully due to an error: {exc.msg}")
