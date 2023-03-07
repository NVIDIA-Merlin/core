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
from typing import Type

from merlin.core.dispatch import (
    create_multihot_col,
    is_list_dtype,
    make_df,
    make_series,
    pull_apart_list,
)
from merlin.dispatch.lazy import lazy_singledispatch
from merlin.table.tensor_column import Device, TensorColumn, _DlpackColumn


@lazy_singledispatch
def _to_dlpack(tensor):
    raise NotImplementedError


@lazy_singledispatch
def _from_dlpack_cpu(to, capsule):
    raise NotImplementedError


@lazy_singledispatch
def _from_dlpack_gpu(to, capsule):
    raise NotImplementedError


def convert_col(column: TensorColumn, target_type: Type):
    if isinstance(column, target_type):
        return column

    try:
        return from_dlpack_col(to_dlpack_col(column), target_type)
    except NotImplementedError:
        pass

    raise NotImplementedError(
        f"Could not convert from type {type(column)} to "
        f"type {target_type} via a zero-copy cast mechanism."
    )


def to_dlpack_col(column: TensorColumn) -> _DlpackColumn:
    vals_cap = _to_dlpack(column.values)
    offs_cap = _to_dlpack(column.offsets) if column.offsets is not None else None
    return _DlpackColumn(vals_cap, offs_cap, column)


def from_dlpack_col(dlpack_col: _DlpackColumn, target_col_type: Type) -> TensorColumn:
    target_array_type = target_col_type.array_type()
    if dlpack_col.ref.device == Device.GPU:
        values = _from_dlpack_gpu(target_array_type, dlpack_col.values)
        offsets = (
            _from_dlpack_gpu(target_array_type, dlpack_col.offsets)
            if dlpack_col.offsets is not None
            else None
        )
    else:
        values = _from_dlpack_cpu(target_array_type, dlpack_col.values)
        offsets = (
            _from_dlpack_cpu(target_array_type, dlpack_col.offsets)
            if dlpack_col.offsets is not None
            else None
        )

    return target_col_type(values, offsets, _ref=dlpack_col.ref)


def df_from_tensor_table(table):
    from merlin.table.tensor_column import Device

    device = "cpu" if table.device == Device.CPU else None
    df_dict = {}
    for col_name, col_data in table.items():
        if col_data.offsets is not None:
            values = make_series(col_data.values, device=device)
            offsets = make_series(col_data.offsets, device=device)
            df_dict[col_name] = create_multihot_col(offsets, values)
        else:
            df_dict[col_name] = make_series(col_data.values, device=device)

    return make_df(df_dict, device=device)


tensor_table_from_df = lazy_singledispatch("tensor_table_from_df")


@tensor_table_from_df.register_lazy("pandas")
def _register_tensor_table_from_pandas_df():
    import pandas as pd

    from merlin.table import NumpyColumn

    @tensor_table_from_df.register(pd.DataFrame)
    def _tensor_table_from_pandas_df(df: pd.DataFrame):
        return _create_table_from_df(df, NumpyColumn, device="cpu")


@tensor_table_from_df.register_lazy("cudf")
def _register_tensor_table_from_cudf_df():
    import cudf

    from merlin.table import CupyColumn

    @tensor_table_from_df.register(cudf.DataFrame)
    def _tensor_table_from_cudf_df(df: cudf.DataFrame):
        return _create_table_from_df(df, CupyColumn)


def _create_table_from_df(df, column_type, device=None):
    from merlin.table import TensorTable

    array_cols = {}
    for col in df.columns:
        if is_list_dtype(df[col]):
            values_series, offsets_series = pull_apart_list(df[col], device=device)
            array_cols[col] = column_type(values_series.values, offsets_series.values)
        else:
            array_cols[col] = column_type(df[col].values)

    return TensorTable(array_cols)
