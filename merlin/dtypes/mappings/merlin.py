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

merlin_dtypes = {
    # Unsigned Integer
    mn.uint8: ["uint8"],
    mn.uint16: ["uint16"],
    mn.uint32: ["uint32"],
    mn.uint64: ["uint64"],
    # Signed integer
    mn.int8: ["int8"],
    mn.int16: ["int16"],
    mn.int32: ["int32"],
    mn.int64: ["int64"],
    # Floating Point
    mn.float16: ["float16"],
    mn.float32: ["float32"],
    mn.float64: ["float64"],
    # Date/Time
    mn.datetime64: ["datetime64"],
    mn.datetime64Y: ["datetime64[Y]"],
    mn.datetime64M: ["datetime64[M]"],
    mn.datetime64D: ["datetime64[D]"],
    mn.datetime64h: ["datetime64[h]"],
    mn.datetime64m: ["datetime64[m]"],
    mn.datetime64s: ["datetime64[s]"],
    mn.datetime64ms: ["datetime64[ms]"],
    mn.datetime64us: ["datetime64[us]"],
    mn.datetime64ns: ["datetime64[ns]"],
    # Miscellaneous
    mn.string: ["str", "string"],
    mn.object_: ["object"],
    mn.struct: ["struct"],
    mn.boolean: ["bool", "boolean"],
}
_dtype_registry.register("merlin", merlin_dtypes)
