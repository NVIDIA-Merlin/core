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
import enum
import functools
import itertools
from typing import Callable, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# unused HAS_GPU import is here for backwards compatibility
from merlin.core.compat import HAS_GPU  # pylint: disable=unused-import # noqa: F401
from merlin.core.compat import cudf
from merlin.core.compat import cupy as cp
from merlin.core.protocols import DataFrameLike, DictLike, SeriesLike

rmm = None

if cudf:
    try:
        import dask_cudf
        import rmm  # type: ignore[no-redef]
        from cudf.core.column import as_column, build_column

        try:
            # cudf >= 21.08
            from cudf.api.types import is_list_dtype as cudf_is_list_dtype
            from cudf.api.types import is_string_dtype as cudf_is_string_dtype
        except ImportError:
            # cudf < 21.08
            from cudf.utils.dtypes import is_list_dtype as cudf_is_list_dtype
            from cudf.utils.dtypes import is_string_dtype as cudf_is_string_dtype
    except ImportError:
        pass

try:
    # Dask >= 2021.5.1
    from dask.dataframe.core import hash_object_dispatch
except ImportError:
    # Dask < 2021.5.1
    from dask.dataframe.utils import hash_object_dispatch

try:
    import nvtx

    annotate = nvtx.annotate
except ImportError:
    # don't have nvtx installed - don't annotate our functions
    def annotate(*args, **kwargs):
        def inner1(func):
            @functools.wraps(func)
            def inner2(*args, **kwargs):
                return func(*args, **kwargs)

            return inner2

        return inner1


if cudf:
    DataFrameType = Union[pd.DataFrame, cudf.DataFrame]  # type: ignore
    SeriesType = Union[pd.Series, cudf.Series]  # type: ignore
else:
    DataFrameType = pd.DataFrame  # type: ignore
    SeriesType = pd.Series  # type: ignore


# Define mapping between non-nullable,
# and nullable types in Pandas
_PD_NULLABLE_MAP = {
    "float32": "Float32",
    "float64": "Float64",
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
    "uint8": "UInt8",
    "uint16": "UInt16",
    "uint32": "UInt32",
    "uint64": "UInt64",
}


class ExtData(enum.Enum):
    """Simple Enum to track external-data types"""

    DATASET = 0
    ARROW = 1
    CUDF = 2
    PANDAS = 3
    DASK_CUDF = 4
    DASK_PANDAS = 5
    PARQUET = 6
    CSV = 7


def create_merlin_dataset(df):
    """Create a merlin.ioDataset"""
    from merlin.io import Dataset

    if not isinstance(df, Dataset):
        # turn arrow format into readable for dispatch
        df_ext_format = detect_format(df)
        if df_ext_format == ExtData.ARROW:
            df = df.to_pandas() if not cudf else cudf.DataFrame.from_arrow(df)
            # run through make df to safely cast to df
        elif df_ext_format in [ExtData.DASK_CUDF, ExtData.DASK_PANDAS]:
            df = df.compute()
        return Dataset(df)
    return df


def read_parquet_metadata(path):
    """Read parquet metadata from path"""
    if cudf:
        return cudf.io.read_parquet_metadata(path)
    full_meta = pq.read_metadata(path)
    pf = pq.ParquetFile(path)
    return full_meta.num_rows, full_meta.num_row_groups, pf.schema.names


def get_lib():
    """Dispatch to the appropriate library (cudf or pandas) for the current environment"""
    return cudf if (cudf and HAS_GPU) else pd


def reinitialize(managed_memory=False):
    if rmm:
        rmm.reinitialize(managed_memory=managed_memory)
    return


def random_uniform(size):
    """Dispatch for numpy.random.RandomState"""
    if cp:
        return cp.random.uniform(size=size)
    else:
        return np.random.uniform(size=size)


def coo_matrix(data, row, col):
    """Dispatch for scipy.sparse.coo_matrix"""
    if cp:
        return cp.sparse.coo_matrix((data, row, col))
    else:
        import scipy

        return scipy.sparse.coo_matrix((data, row, col))


def is_dataframe_object(x):
    # Simple check if object is a cudf or pandas
    # DataFrame object
    if cudf:
        return isinstance(x, (cudf.DataFrame, pd.DataFrame))
    return isinstance(x, pd.DataFrame)


def nullable_series(data, like_df, dtype):
    # Return a Series containing the elements in `data`,
    # with a nullable version of `dtype`, using `like_df`
    # to infer the Series constructor
    if isinstance(like_df, pd.DataFrame):
        # Note that we cannot use "int32"/"int64" to
        # represent nullable data in pandas
        return like_df._constructor_sliced(data, dtype=_PD_NULLABLE_MAP.get(str(dtype), dtype))
    return like_df._constructor_sliced(data, dtype=dtype)


def is_series_object(x):
    # Simple check if object is a cudf or pandas
    # Series object
    if cudf:
        return isinstance(x, (cudf.Series, pd.Series))
    return isinstance(x, pd.Series)


def is_cpu_object(x):
    # Simple check if object is a cudf or pandas
    # DataFrame object
    return isinstance(x, (pd.DataFrame, pd.Series))


def is_series_or_dataframe_object(maybe_series_or_df):
    return is_series_object(maybe_series_or_df) or is_dataframe_object(maybe_series_or_df)


def hex_to_int(s, dtype=None):
    def pd_convert_hex(x):
        if pd.isnull(x):
            return pd.NA
        return int(x, 16)

    if isinstance(s, pd.Series):
        # Pandas Version
        if s.dtype == "object":
            s = s.apply(pd_convert_hex)
        return s.astype("Int64").astype(dtype or "Int32")
    else:
        # CuDF Version
        if s.dtype == "object":
            s = s.str.htoi()
        return s.astype(dtype or np.int32)


def random_state(seed, like_df=None):
    """Dispatch for numpy.random.RandomState"""
    if like_df is None:
        return cp.random.RandomState(seed) if cp else np.random.RandomState(seed)
    elif isinstance(like_df, (pd.DataFrame, pd.Series, pd.RangeIndex)):
        return np.random.RandomState(seed)
    elif cudf and isinstance(like_df, (cudf.DataFrame, cudf.Series, cudf.RangeIndex)):
        return cp.random.RandomState(seed)
    else:
        raise ValueError(
            "Unsupported dataframe type: "
            f"{type(like_df)}"
            " Supported types: a DataFrame, Series, or RangeIndex (cudf or pandas)."
        )


def arange(size, like_df=None, dtype=None):
    """Dispatch for numpy.arange"""
    if like_df is None:
        return cp.arange(size, dtype=dtype) if cp else np.arange(size, dtype=dtype)
    elif isinstance(like_df, (np.ndarray, pd.DataFrame, pd.Series, pd.RangeIndex)):
        return np.arange(size, dtype=dtype)
    elif cudf and isinstance(like_df, (cp.ndarray, cudf.DataFrame, cudf.Series, cudf.RangeIndex)):
        return cp.arange(size, dtype=dtype)
    else:
        raise ValueError(
            "Unsupported dataframe type: "
            f"{type(like_df)}"
            " Expected either a pandas or cudf DataFrame or Series."
        )


def array(x, like_df=None, dtype=None):
    """Dispatch for numpy.array"""
    if like_df is None:
        return cp.array(x, dtype=dtype) if cp else np.array(x, dtype=dtype)
    elif isinstance(like_df, (np.ndarray, pd.DataFrame, pd.Series, pd.RangeIndex)):
        return np.array(x, dtype=dtype)
    elif cudf and isinstance(like_df, (cp.ndarray, cudf.DataFrame, cudf.Series, cudf.RangeIndex)):
        return cp.array(x, dtype=dtype)
    else:
        raise ValueError(
            "Unsupported dataframe type: "
            f"{type(like_df)}"
            " Expected either a pandas or cudf DataFrame or Series."
        )


def zeros(size, like_df=None, dtype=None):
    """Dispatch for numpy.array"""
    if like_df is None:
        return cp.zeros(size, dtype=dtype) if cp else np.zeros(size, dtype=dtype)
    elif isinstance(like_df, (np.ndarray, pd.DataFrame, pd.Series, cudf.RangeIndex)):
        return np.zeros(size, dtype=dtype)
    elif cudf and isinstance(like_df, (cp.ndarray, cudf.DataFrame, cudf.Series, cudf.RangeIndex)):
        return cp.zeros(size, dtype=dtype)
    else:
        raise ValueError(
            "Unsupported dataframe type: "
            f"{type(like_df)}"
            " Expected either a pandas or cudf DataFrame or Series."
        )


def hash_series(ser):
    """Row-wise Series hash"""
    if isinstance(ser, pd.Series):
        # Using pandas hashing, which does not produce the
        # same result as cudf.Series.hash_values().  Do not
        # expect hash-based data transformations to be the
        # same on CPU and CPU.  TODO: Fix this (maybe use
        # murmurhash3 manually on CPU).
        return hash_object_dispatch(ser).values
    elif cudf and isinstance(ser, cudf.Series):
        if is_list_dtype(ser):
            return ser.list.leaves.hash_values()
        else:
            return ser.hash_values()
    else:
        raise ValueError(
            "Unsupported series type: " f"{type(ser)}" " Expected either a pandas or cudf Series."
        )


def series_has_nulls(s):
    """Check if Series contains any null values"""
    if isinstance(s, pd.Series):
        return s.isnull().values.any()
    else:
        return s.has_nulls


def list_val_dtype(ser: SeriesLike) -> np.dtype:
    """
    Return the dtype of the leaves from a list or nested list

    Parameters
    ----------
    ser : SeriesLike
        A series where the rows contain lists or nested lists

    Returns
    -------
    np.dtype
        The dtype of the innermost elements
    """
    if is_list_dtype(ser):
        if cudf is not None and isinstance(ser, cudf.Series):
            if is_list_dtype(ser):
                ser = ser.list.leaves
            return ser.dtype
        elif isinstance(ser, pd.Series):
            return pd.core.dtypes.cast.infer_dtype_from(next(iter(pd.core.common.flatten(ser))))[0]
    if isinstance(ser, np.ndarray):
        return ser.dtype
    # adds detection when in merlin column
    if hasattr(ser, "is_list"):
        return ser[0].dtype
    return None


def is_list_dtype(ser):
    """Check if Series, dtype, or array contains or represents list elements"""
    # adds detection for merlin column
    if hasattr(ser, "is_list"):
        return ser.is_list
    if isinstance(ser, pd.Series):
        if not len(ser):  # pylint: disable=len-as-condition
            return False
        return pd.api.types.is_list_like(ser.values[0])
    elif cudf and isinstance(ser, (cudf.Series, cudf.ListDtype)):
        return cudf_is_list_dtype(ser)
    elif isinstance(ser, np.ndarray) or (cp and isinstance(ser, cp.ndarray)):
        return len(ser.shape) > 1
    return pd.api.types.is_list_like(ser)


def is_string_dtype(dtype: np.dtype) -> bool:
    """Check if the dtype of obj is a string type

    Parameters
    ----------
    obj : np.dtype
        Potential string dtype to check

    Returns
    -------
    bool
        `True` if the dtype of `obj` is a string type
    """
    if cudf:
        return cudf_is_string_dtype(dtype)
    return pd.api.types.is_string_dtype(dtype)


def flatten_list_column_values(s):
    """returns a flattened list from a list column"""
    if isinstance(s, pd.Series):
        return pd.Series(itertools.chain(*s))
    elif cudf and isinstance(s, cudf.Series):
        return s.list.leaves
    elif cp and isinstance(s, cp.ndarray):
        return s.flatten()
    elif isinstance(s, np.ndarray):
        return s.flatten()
    else:
        raise ValueError(
            "Unsupported series type: "
            f"{type(s)} "
            "Expected either a pandas or cuDF Series. "
            "Or a NumPy or CuPy array"
        )


def flatten_list_column(s):
    """Flatten elements of a list-based column, and return as a DataFrame"""
    values = flatten_list_column_values(s)
    if isinstance(s, pd.Series) or not cudf:
        return pd.DataFrame({s.name: values})
    else:
        return cudf.DataFrame({s.name: values})


def concat_columns(args: list):
    """Dispatch function to concatenate DataFrames with axis=1"""
    if len(args) == 1:
        return args[0]
    elif cudf is not None and isinstance(args[0], cudf.DataFrame):
        return cudf.concat(
            [a.reset_index(drop=True) for a in args],
            axis=1,
        )
    elif isinstance(args[0], pd.DataFrame):
        return pd.concat(
            [a.reset_index(drop=True) for a in args],
            axis=1,
        )
    elif isinstance(args[0], DictLike):
        result = type(args[0])()
        for arg in args:
            result.update(arg)
        return result
    return None


def read_parquet_dispatch(df: DataFrameLike) -> Callable:
    """Dispatch function for reading parquet files"""
    return read_dispatch(df=df, fmt="parquet")


def read_dispatch(
    df: Union[DataFrameLike, str] = None, cpu=None, collection=False, fmt="parquet"
) -> Callable:
    """Return the necessary read_parquet function to generate
    data of a specified type.
    """
    if cpu or isinstance(df, pd.DataFrame):
        _mod = dd if collection else pd
    elif cudf and isinstance(df, cudf.DataFrame):
        _mod = dask_cudf if collection else cudf.io
    else:
        if collection:
            _mod = dask_cudf if cudf else dd
        else:
            _mod = cudf.io if cudf else pd
    _attr = "read_csv" if fmt == "csv" else "read_parquet"
    return getattr(_mod, _attr)


def parquet_writer_dispatch(df: DataFrameLike, path=None, **kwargs):
    """Return the necessary ParquetWriter class to write
    data of a specified type.

    If `path` is specified, an initialized `ParquetWriter`
    object will be returned.  To do this, the pyarrow schema
    will be inferred from df, and kwargs will be used for the
    ParquetWriter-initialization call.
    """
    _args = []
    if isinstance(df, pd.DataFrame):
        _cls = pq.ParquetWriter
        if path:
            _args.append(pa.Table.from_pandas(df, preserve_index=False).schema)
    elif cudf is not None:
        _cls = cudf.io.parquet.ParquetWriter
    else:
        raise ValueError(
            "Unable to load cudf. "
            "Please check that your environment has GPU(s) and cudf available."
        )

    if not path:
        return _cls

    ret = _cls(path, *_args, **kwargs)
    if isinstance(df, pd.DataFrame):
        ret.write_table = lambda df: _cls.write_table(
            ret, pa.Table.from_pandas(df, preserve_index=False)
        )
    return ret


def encode_list_column(original, encoded, dtype=None):
    """Convert `encoded` to be a list column with the
    same offsets as `original`
    """
    if isinstance(original, pd.Series):
        # Pandas version (not very efficient)
        offset = 0
        new_data = []
        for val in original.values:
            size = len(val)
            new_data.append(np.array(encoded[offset : offset + size], dtype=dtype))
            offset += size
        return pd.Series(new_data)
    else:
        # CuDF version
        encoded = as_column(encoded)
        if dtype:
            encoded = encoded.astype(dtype, copy=False)
        list_dtype = cudf.core.dtypes.ListDtype(encoded.dtype if dtype is None else dtype)
        return build_column(
            None,
            dtype=list_dtype,
            size=original.size,
            children=(original._column.offsets, encoded),
        )


def pull_apart_list(original, device=None):
    values = flatten_list_column_values(original)
    if isinstance(original, pd.Series):
        offsets = pd.concat([pd.Series([0]), original.map(len).cumsum()]).reset_index(drop=True)
        if isinstance(offsets[0], list):
            offsets = pd.Series(offsets.reshape().flatten()).reset_index(drop=True)
    else:
        offsets = original._column.offsets
        elements = original._column.elements
        if isinstance(elements, cudf.core.column.lists.ListColumn):
            offsets = make_series(elements.offsets)[offsets]
    return make_series(values, device), make_series(offsets, device)


def to_arrow(x):
    """Move data to arrow format"""
    if isinstance(x, pd.DataFrame):
        return pa.Table.from_pandas(x, preserve_index=False)
    else:
        return x.to_arrow()


def concat(objs, **kwargs):
    """dispatch function for concat"""
    if isinstance(objs[0], dd.DataFrame):
        return dd.multi.concat(objs)
    elif isinstance(objs[0], (pd.DataFrame, pd.Series)):
        return pd.concat(objs, **kwargs)
    elif cudf and isinstance(objs[0], (cudf.DataFrame, cudf.Series)):
        return cudf.core.reshape.concat(objs, **kwargs)
    else:
        raise ValueError(
            "Unsupported dataframe type: "
            f"{type(objs[0])}"
            " Expected a pandas, cudf, or dask DataFrame."
        )


def make_df(_like_df=None, device=None):
    """Return a DataFrame with the same dtype as `_like_df`"""
    if (
        not cudf
        or device == "cpu"
        or not HAS_GPU
        or isinstance(_like_df, (pd.DataFrame, pd.Series))
    ):
        # move to pandas need it on CPU (host memory)
        # can be a cudf, cupy or numpy Series
        if cudf and isinstance(_like_df, (cudf.DataFrame, cudf.Series)):
            # move to cpu
            return _like_df.to_pandas()
        if cp and isinstance(_like_df, cp.ndarray):
            return pd.DataFrame(_like_df.get())
        else:
            return pd.DataFrame(_like_df)
    else:
        if isinstance(_like_df, dict) and len(_like_df) > 0:
            if all(isinstance(v, pd.Series) for v in _like_df.values()):
                return pd.DataFrame(_like_df)
        return cudf.DataFrame(_like_df)


def make_series(_like_ser=None, device=None):
    """Return a Series with the same dtype as `_like_ser`"""
    if not cudf or device == "cpu":
        return pd.Series(_like_ser)
    return cudf.Series(_like_ser)


def add_to_series(series, to_add, prepend=True):
    if isinstance(series, pd.Series):
        series_to_add = pd.Series(to_add)
    elif isinstance(series, cudf.Series):
        series_to_add = cudf.Series(to_add)
    else:
        raise ValueError("Unrecognized series, please provide either a pandas a cudf series")

    series_to_concat = [series_to_add, series] if prepend else [series, series_to_add]

    return concat(series_to_concat)


def detect_format(data):
    """Utility to detect the format of `data`"""
    from merlin.io import Dataset

    if isinstance(data, Dataset):
        return ExtData.DATASET
    elif isinstance(data, dd.DataFrame):
        if isinstance(data._meta, pd.DataFrame):
            return ExtData.DASK_PANDAS
        return ExtData.DASK_CUDF
    elif isinstance(data, pd.DataFrame):
        return ExtData.PANDAS
    elif isinstance(data, pa.Table):
        return ExtData.ARROW
    elif isinstance(data, cudf.DataFrame):
        return ExtData.CUDF
    else:
        mapping = {
            "pq": ExtData.PARQUET,
            "parquet": ExtData.PARQUET,
            "csv": ExtData.CSV,
        }
        if isinstance(data, list) and data:
            file_type = mapping.get(str(data[0]).rsplit(".", maxsplit=1)[-1], None)
        else:
            file_type = mapping.get(str(data).rsplit(".", maxsplit=1)[-1], None)
        if file_type is None:
            raise ValueError("Data format not recognized.")
        return file_type


def convert_data(x, cpu=True, to_collection=None, npartitions=1):
    """Move data between cpu and gpu-backed data.

    Note that the input ``x`` may be an Arrow Table,
    but the output will only be a pandas or cudf DataFrame.
    Use `to_collection=True` to specify that the output should
    always be a Dask collection (otherwise, "serial" DataFrame
    objects will remain "serial").
    """
    if cpu:
        if isinstance(x, dd.DataFrame):
            # If input is a dask_cudf collection, convert
            # to a pandas-backed Dask collection
            if cudf is None or not isinstance(x, dask_cudf.DataFrame):
                # Already a Pandas-backed collection
                return x
            # Convert cudf-backed collection to pandas-backed collection
            return x.to_dask_dataframe()
        else:
            # Make sure _x is a pandas DataFrame
            _x = x if isinstance(x, pd.DataFrame) else x.to_pandas()
            # Output a collection if `to_collection=True`
            return dd.from_pandas(_x, sort=False, npartitions=npartitions) if to_collection else _x
    elif cudf and dask_cudf:
        if isinstance(x, dd.DataFrame):
            # If input is a Dask collection, convert to dask_cudf
            if isinstance(x, dask_cudf.DataFrame):
                # Already a cudf-backed Dask collection
                return x
            # Convert pandas-backed collection to cudf-backed collection
            return x.map_partitions(cudf.from_pandas)
        elif isinstance(x, pa.Table):
            return cudf.DataFrame.from_arrow(x)
        else:
            # Make sure _x is a cudf DataFrame
            _x = x
            if isinstance(x, pa.Table):
                _x = cudf.DataFrame.from_arrow(x)
            elif isinstance(x, pd.DataFrame):
                _x = cudf.DataFrame.from_pandas(x)
            # Output a collection if `to_collection=True`
            return (
                dask_cudf.from_cudf(_x, sort=False, npartitions=npartitions)
                if to_collection
                else _x
            )
    else:
        raise RuntimeError(
            "Unable to move data to GPU. "
            "cudf and dask_cudf are not available. "
            "Make sure these packages are installed and can be imported in this environment. "
        )


def to_host(x):
    """Move cudf.DataFrame to host memory for caching.

    All other data will pass through unchanged.
    """
    if cudf and isinstance(x, cudf.DataFrame):
        return x.to_arrow()
    return x


def from_host(x):
    if isinstance(x, pd.DataFrame):
        return cudf.DataFrame.from_pandas(x)
    elif isinstance(x, pa.Table):
        return cudf.DataFrame.from_arrow(x)
    return x


def build_cudf_list_column(new_elements, new_offsets):
    """Method creates a List series from the corresponding elements and
    row_lengths

    Parameters
    ----------
    elements : cudf.Series
        The elements of a pandas series
    row_lengths : cudf.Series
        The corresponding row lengths of the elements

    Returns
    -------
    cudf.Series
        The list column with corresponding elements and row_lengths as a series.
    """
    if cudf:
        return build_column(
            None,
            dtype=cudf.core.dtypes.ListDtype(new_elements.dtype),
            size=new_offsets.size - 1,
            children=(as_column(new_offsets), as_column(new_elements)),
        )
    return []


def build_pandas_list_column(elements, row_lengths):
    """Method creates a List series from the corresponding elements and
    row_lengths

    Parameters
    ----------
    elements : pd.Series
        The elements of a pandas series
    row_lengths : pd.Series
        The corresponding row lengths of the elements

    Returns
    -------
    pd.Series
        The list column with corresponding elements and row_lengths as a series.
    """
    offset = 0
    rows = []
    for row_length in row_lengths:
        row_length = int(row_length)
        row = elements[offset : offset + row_length]
        offset += row_length
        rows.append(row.values)
    return pd.Series(rows)


def create_multihot_col(offsets, elements):
    """
    offsets = cudf series with offset values for list data
    data = cudf series with the list data flattened to 1-d
    """
    if isinstance(elements, pd.Series):
        lh = pd.Series(offsets[1:]).reset_index(drop=True)
        rh = pd.Series(offsets[:-1]).reset_index(drop=True)
        row_lengths = lh - rh

        col = build_pandas_list_column(elements, row_lengths)
    else:
        offsets = as_column(offsets, dtype="int32")
        elements = as_column(elements)
        col = build_cudf_list_column(elements, offsets)
        col = cudf.Series(col)
    return col


def generate_local_seed(global_rank, global_size):
    random_state = get_random_state()
    if cp:
        seeds = random_state.tomaxint(size=global_size)
        cp.random.seed(seeds[global_rank].get())
    else:
        seeds = random_state.randint(0, 2**32, size=global_size)
    return seeds[global_rank]


def get_random_state():
    """get_random_state from either cupy or numpy."""
    if cp:
        return cp.random.get_random_state()
    return np.random.mtrand.RandomState()


def df_from_dict(col_dict):
    from merlin.table import TensorTable, df_from_tensor_table

    return df_from_tensor_table(TensorTable(col_dict))


def dict_from_df(df: DataFrameLike):
    from merlin.table import tensor_table_from_df

    return tensor_table_from_df(df).to_dict()
