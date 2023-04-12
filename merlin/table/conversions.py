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
from typing import Callable, Type

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


def to_dlpack_col(column: TensorColumn, to_dlpack_fn: Callable) -> _DlpackColumn:
    """Creates  dlpack representation of the TensorColumn supplied.

    Parameters
    ----------
    column : TensorColumn
        Original data to be put in dlpack capsule(s)
    to_dlpack_fn : Callable
        The logic to use to create dlpack representation

    Returns
    -------
    _DlpackColumn
        A TensorColumn with values and offsets represented as dlpack capsules
    """
    vals_cap = to_dlpack_fn(column.values)
    offs_cap = to_dlpack_fn(column.offsets) if column.offsets is not None else None
    return _DlpackColumn(vals_cap, offs_cap, column)


def from_dlpack_col(
    dlpack_col: _DlpackColumn, from_dlpack_fn: Callable, target_col_type: Type, _unsafe: bool
) -> TensorColumn:
    """Unwraps a DLpack representation of a TensorColumn and creates a
    TensorColumn of the target_col_type. This function is used in conjunction
    with to_dlpack_col.

    Parameters
    ----------
    dlpack_col : _DlpackColumn
        The dlpack representation of the original TensorColum
    from_dlpack_fn : Callable
        Function containing logic to unwrap dlpack capsule
    target_col_type : Type
        Desired TensorColumn return type from unwrap of dlpack representation.

    Returns
    -------
    TensorColumn
        A TensorColumn of type target_col_type.
    """
    target_array_type = target_col_type.array_type()

    values = from_dlpack_fn(target_array_type, dlpack_col.values)
    offsets = (
        from_dlpack_fn(target_array_type, dlpack_col.offsets)
        if dlpack_col.offsets is not None
        else None
    )

    return target_col_type(values, offsets, _ref=dlpack_col.ref, _unsafe=_unsafe)


def _dispatch_dlpack_fns(column: TensorColumn, target_type: Type):
    from_dlpack_fn = _from_dlpack_gpu if column.device == Device.GPU else _from_dlpack_cpu
    to_dlpack_fn = _to_dlpack

    try:
        to_dlpack_fn = to_dlpack_fn.dispatch(column.values)
        from_dlpack_fn = from_dlpack_fn.dispatch(target_type.array_type())
        return (to_dlpack_fn, from_dlpack_fn)
    except NotImplementedError:
        pass

    raise NotImplementedError(
        f"Could not convert from type {type(column)} to "
        f"type {target_type} via a zero-copy cast mechanism."
    )


def convert_col(
    column: TensorColumn,
    target_type: Type,
    _to_dlpack_fn: Callable = None,
    _from_dlpack_fn: Callable = None,
    _unsafe: bool = False,
):
    """Convert a TensorColumn to a Different TensorColumn,
    uses DLPack (zero copy) to transfer between TensorColumns

    Parameters
    ----------
    column : TensorColumn
        The Column to be transformed
    target_type : Type
        The desired TensorColumn, to be produced
    _to_dlpack_fn : Callable, optional
        cached to_dlpack function, by default None
    _from_dlpack_fn : Callable, optional
        cached from_dlpack function, by default None

    Returns
    -------
    TensorColumn
        A TensorColumn of the type identified in target_type parameter.
    """
    # If there's nothing to do, take a shortcut
    if isinstance(column, target_type):
        return column

    # Decide how to convert
    if _to_dlpack_fn is None or _from_dlpack_fn is None:
        _to_dlpack_fn, _from_dlpack_fn = _dispatch_dlpack_fns(column, target_type)

    # Do the conversion
    dlpack_col = to_dlpack_col(column, _to_dlpack_fn)
    converted_col = from_dlpack_col(dlpack_col, _from_dlpack_fn, target_type, _unsafe=_unsafe)

    # Return the result
    return converted_col


def df_from_tensor_table(table):
    """
    Create a dataframe from a TensorTable
    """
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
    from merlin.core.compat import cudf
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
