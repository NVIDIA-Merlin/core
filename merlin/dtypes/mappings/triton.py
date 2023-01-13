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
from merlin.dtypes.registry import _dtype_registry

# Only define a Triton dtype mapping if `tritonclient` is available
try:
    import tritonclient.grpc.model_config_pb2 as model_config

    # The type constants on `model_config` are literally integers,
    # so this only works if we don't have any other dtypes that
    # either are or are equivalent to integers. We work around
    # this by checking base classes for dtypes that have them (e.g.
    # Tensorflow.)
    triton_dtypes = {
        # Unsigned Integer
        mn.uint8: [model_config.TYPE_UINT8],
        mn.uint16: [model_config.TYPE_UINT16],
        mn.uint32: [model_config.TYPE_UINT32],
        mn.uint64: [model_config.TYPE_UINT64],
        # Signed integer
        mn.int8: [model_config.TYPE_INT8],
        mn.int16: [model_config.TYPE_INT16],
        mn.int32: [model_config.TYPE_INT32],
        mn.int64: [model_config.TYPE_INT64],
        # Floating Point
        mn.float16: [model_config.TYPE_FP16],
        mn.float32: [
            model_config.TYPE_FP32,
        ],
        mn.float64: [model_config.TYPE_FP64],
        # Miscellaneous
        mn.string: [model_config.TYPE_STRING],
        mn.boolean: [model_config.TYPE_BOOL],
    }
    _dtype_registry.register("triton", triton_dtypes)
except ImportError as exc:
    from warnings import warn

    warn(f"Triton dtype mappings did not load successfully due to an error: {exc.msg}")
