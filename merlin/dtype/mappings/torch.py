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
from merlin.dtype.registry import _dtype_registry


try:
    from torch import bool as bool_
    from torch import float16, float32, float64, int8, int16, int32, int64, uint8

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

    warn(f"PyTorch dtype mappings did not load successfully due to an error: {exc.msg}")
