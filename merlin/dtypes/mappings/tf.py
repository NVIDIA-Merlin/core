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
import merlin.dtypes.aliases as mn
from merlin.dtypes.mapping import DTypeMapping, NumpyPreprocessor
from merlin.dtypes.registry import _dtype_registry

try:
    from tensorflow import dtypes as tf_dtypes

    tf_dtypes = DTypeMapping(
        {
            # Unsigned Integer
            mn.uint8: [tf_dtypes.uint8],
            mn.uint16: [tf_dtypes.uint16],
            mn.uint32: [tf_dtypes.uint32],
            mn.uint64: [tf_dtypes.uint64],
            # Signed integer
            mn.int8: [tf_dtypes.int8],
            mn.int16: [tf_dtypes.int16],
            mn.int32: [tf_dtypes.int32],
            mn.int64: [tf_dtypes.int64],
            # Floating Point
            mn.float16: [tf_dtypes.float16],
            mn.float32: [tf_dtypes.float32],
            mn.float64: [tf_dtypes.float64],
            # Miscellaneous
            mn.boolean: [tf_dtypes.bool],
        },
        base_class=tf_dtypes.DType,
        translator=NumpyPreprocessor(
            "tf", lambda raw: raw.as_numpy_dtype, attrs=["as_numpy_dtype"]
        ),
    )
    _dtype_registry.register("tf", tf_dtypes)
    _dtype_registry.register("tensorflow", tf_dtypes)
except ImportError as exc:
    from warnings import warn

    warn(f"Tensorflow dtype mappings did not load successfully due to an error: {exc.msg}")
