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

import contextlib
import threading
from typing import Dict, Union

import pandas as pd
import pyarrow as pa
from dask.distributed import get_worker

from merlin.core.compat import cudf
from merlin.core.dispatch import DataFrameType

# Use global variable as the default
# cache when there are no distributed workers.
# Use a thread lock to "handle" multiple Dask
# worker threads.
_WORKER_CACHE = {}
_WORKER_CACHE_LOCK = threading.RLock()


@contextlib.contextmanager
def get_worker_cache(name):
    with _WORKER_CACHE_LOCK:
        yield _get_worker_cache(name)


def _get_worker_cache(name):
    """Utility to get the `name` element of the cache
    dictionary for the current worker.  If executed
    by anything other than a distributed Dask worker,
    we will use the global `_WORKER_CACHE` variable.
    """
    try:
        worker = get_worker()
    except ValueError:
        # There is no dask.distributed worker.
        # Assume client/worker are same process
        global _WORKER_CACHE  # pylint: disable=global-variable-not-assigned
        if name not in _WORKER_CACHE:
            _WORKER_CACHE[name] = {}
        return _WORKER_CACHE[name]
    if not hasattr(worker, "worker_cache"):
        worker.worker_cache = {}
    if name not in worker.worker_cache:
        worker.worker_cache[name] = {}
    return worker.worker_cache[name]


def fetch_table_data(
    table_cache: Dict[str, Union[pa.Table, DataFrameType]],
    path,
    *,
    cache="disk",
    cats_only=False,
    reader=None,
    columns=None,
    **reader_kwargs
) -> DataFrameType:
    """Utility to retrieve a cudf DataFrame from a cache (and adds the
    DataFrame to a cache if the element is missing).

    Parameters
    ----------
    table_cache : Dict[str, Union[pa.Table, DataFrameType]]
        Dataframe Cache
    path : str, path object or file-like object
        path to data representing DataFrame
    cache : str, optional
        Type of cache, by default "disk"
        Supported values, {"device", "disk", "host"}
    cats_only : bool, optional
        Return labels column with index value, by default False
    reader : Callable, optional
        DataFrame function to read parquet from path, by default None
    columns : List[str], optional
        Read subset of columns, by default None (read all columns)
    **reader_kwargs
        Optional keyword arguments to pass to reader function

    Returns
    -------
    DataFrameType
        DataFrame read from table cache or from path
    """

    _lib = cudf if cudf else pd
    reader = reader or _lib.read_parquet
    load_cudf = cudf and reader == cudf.read_parquet
    table_or_df: Union[pa.Table, DataFrameType] = table_cache.get(path, None)
    cache_df = cache == "device"
    if table_or_df is None:
        use_kwargs = {"columns": columns} if columns is not None else {}
        use_kwargs.update(reader_kwargs)
        df = reader(path, **use_kwargs)
        if columns is not None:
            df = df[columns]
        if cache == "host":
            if cudf and isinstance(df, cudf.DataFrame):
                table_cache[path] = df.to_arrow()
            elif isinstance(df, pd.DataFrame):
                table_cache[path] = pa.Table.from_pandas(df)
                cache_df = True
        if cats_only:
            df.index.name = "labels"
            df.reset_index(drop=False, inplace=True)
        if cache_df:
            table_cache[path] = df.copy(deep=False)
        return df
    elif isinstance(table_or_df, pa.Table):
        table = table_or_df
        if cudf and load_cudf:
            df = cudf.DataFrame.from_arrow(table)
        else:
            df = table.to_pandas()
        if columns is not None:
            df = df[columns]
        if cats_only:
            df.index.name = "labels"
            df.reset_index(drop=False, inplace=True)
        return df

    # DataFrame type
    return table_or_df


def clean_worker_cache(name=None):
    """Utility to clean the cache dictionary for the
    current worker.  If a `name` argument is passed,
    only that element of the dictionary will be removed.
    """
    with _WORKER_CACHE_LOCK:
        try:
            worker = get_worker()
        except ValueError:
            global _WORKER_CACHE  # pylint: disable=global-statement
            if _WORKER_CACHE:
                if name:
                    del _WORKER_CACHE[name]
                else:
                    del _WORKER_CACHE
                    _WORKER_CACHE = {}
            return
        if hasattr(worker, "worker_cache"):
            if name:
                del worker.worker_cache[name]
            else:
                del worker.worker_cache
        return
