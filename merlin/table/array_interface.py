#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
try:
    import tensorflow as tf
except:
    tf = None

try:
    import cupy as cp
except:
    cp = None


from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Type

import pytest

from merlin.dispatch.lazy import lazysingledispatch
from merlin.table import CupyColumn, TensorColumn, TensorflowColumn
from merlin.table.tensor_column import (
    ArrayColumn,
    CudaArrayColumn,
    Device,
    DlpackColumn,
    TensorColumn,
    TransferColumn,
)


@singledispatch
def convert_col(column: TensorColumn, target_type: Type):
    try:
        return from_cuda_array_col(to_cuda_array_col(column), target_type)
    except NotImplementedError:
        pass

    try:
        return from_dlpack_col(to_dlpack_col(column), target_type)
    except NotImplementedError:
        pass

    try:
        return from_array_col(to_array_col(column), target_type)
    except NotImplementedError:
        pass

    raise NotImplementedError("There's no translation")


@convert_col.register
def from_array_col(array_col: ArrayColumn, target_col_type: TensorColumn):
    target_array_type = target_col_type.array_type()
    if array_col.ref.device == Device.CPU:
        values = _from_array(target_array_type, array_col.values)
        offsets = _from_array(target_array_type, array_col.offsets)
    else:
        raise NotImplementedError
    return target_col_type(values, offsets, _ref=array_col.ref)


@convert_col.register
def from_cuda_array_col(array_col: CudaArrayColumn, target_col_type: TensorColumn):
    target_array_type = target_col_type.array_type()
    if array_col.ref.device == Device.GPU:
        values = _from_cuda_array(target_array_type, array_col.values)
        offsets = _from_cuda_array(target_array_type, array_col.offsets)
    else:
        raise NotImplementedError
    return target_col_type(values, offsets, _ref=array_col.ref)


@convert_col.register
def from_dlpack_col(dlpack_col: DlpackColumn, target_col_type: TensorColumn):
    target_array_type = target_col_type.array_type()
    if dlpack_col.ref.device == Device.GPU:
        values = _from_dlpack_gpu(target_array_type, dlpack_col.values)
        offsets = _from_dlpack_gpu(target_array_type, dlpack_col.offsets)
    else:
        values = _from_dlpack_cpu(target_array_type, dlpack_col.values)
        offsets = _from_dlpack_cpu(target_array_type, dlpack_col.offsets)
    return target_col_type(values, offsets, _ref=dlpack_col.ref)
