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

from merlin.dtypes.base import DType, ElementType, ElementUnit

# Unsigned Integer
uint8 = DType("uint8", ElementType.UInt, 8)
uint16 = DType("uint16", ElementType.UInt, 16)
uint32 = DType("uint32", ElementType.UInt, 32)
uint64 = DType("uint64", ElementType.UInt, 64)

# Signed Integer
int8 = DType("int8", ElementType.Int, 8, signed=True)
int16 = DType("int16", ElementType.Int, 16, signed=True)
int32 = DType("int32", ElementType.Int, 32, signed=True)
int64 = DType("int64", ElementType.Int, 64, signed=True)

# Float
float16 = DType("float16", ElementType.Float, 16, signed=True)
float32 = DType("float32", ElementType.Float, 32, signed=True)
float64 = DType("float64", ElementType.Float, 64, signed=True)

# Date/Time
datetime64 = DType("datetime64", ElementType.DateTime, 64)
datetime64Y = DType("datetime64[Y]", ElementType.DateTime, 64, ElementUnit.Year)
datetime64M = DType("datetime64[M]", ElementType.DateTime, 64, ElementUnit.Month)
datetime64D = DType("datetime64[D]", ElementType.DateTime, 64, ElementUnit.Day)
datetime64h = DType("datetime64[h]", ElementType.DateTime, 64, ElementUnit.Hour)
datetime64m = DType("datetime64[m]", ElementType.DateTime, 64, ElementUnit.Minute)
datetime64s = DType("datetime64[s]", ElementType.DateTime, 64, ElementUnit.Second)
datetime64ms = DType("datetime64[ms]", ElementType.DateTime, 64, ElementUnit.Millisecond)
datetime64us = DType("datetime64[us]", ElementType.DateTime, 64, ElementUnit.Microsecond)
datetime64ns = DType("datetime64[ns]", ElementType.DateTime, 64, ElementUnit.Nanosecond)

# Miscellaneous
string = DType("str", ElementType.String)
boolean = DType("bool", ElementType.Bool)
object_ = DType("object", ElementType.Object)
struct = DType("struct", ElementType.Struct)
unknown = DType("unknown", ElementType.Unknown)
