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
import glob

import numpy as np
import pandas as pd
import pytest

from merlin.core.compat import HAS_GPU, cudf
from merlin.core.dispatch import dataframe_columnwise_explode, make_df
from merlin.core.utils import Distributed
from merlin.io import Dataset


class TestDatasetCpu:
    def test_true(self):
        dataset = Dataset(make_df({"a": [1, 2, 3]}), cpu=True)
        assert dataset.cpu is True
        assert isinstance(dataset.compute(), pd.DataFrame)

    @pytest.mark.skipif(not (cudf and HAS_GPU), reason="requires cuDF and GPU")
    def test_default_cudf(self):
        dataset = Dataset(make_df({"a": [1, 2, 3]}))
        assert dataset.cpu is False
        assert isinstance(dataset.compute(), cudf.DataFrame)

    @pytest.mark.skipif(cudf and HAS_GPU, reason="requires environment without cuDF and GPU")
    def test_default_pandas(self):
        dataset = Dataset(make_df({"a": [1, 2, 3]}))
        assert dataset.cpu is True
        assert isinstance(dataset.compute(), pd.DataFrame)

    @pytest.mark.skipif(not (cudf and HAS_GPU), reason="requires cuDF and GPU")
    def test_false_with_cudf_and_gpu(self):
        dataset = Dataset(make_df({"a": [1, 2, 3]}), cpu=False)
        assert dataset.cpu is False
        assert isinstance(dataset.compute(), cudf.DataFrame)

    @pytest.mark.skipif(cudf or HAS_GPU, reason="requires environment without cuDF or GPU")
    def test_false_missing_cudf_or_gpu(self):
        with pytest.raises(RuntimeError):
            Dataset(make_df({"a": [1, 2, 3]}), cpu=False)


def test_infer_list_dtype_unknown():
    df = pd.DataFrame({"col": [[], []]})
    dataset = Dataset(df, cpu=True)
    assert dataset.schema["col"].dtype.element_type.value == "unknown"


@pytest.mark.parametrize("engine", ["csv", "parquet"])
def test_dask_df_array_npy(tmpdir, datasets, engine):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    # cannot have any null/NA entries
    dataset = Dataset(Dataset(paths).to_ddf().compute().fillna(method="ffill"))
    path = str(tmpdir / "result.npy")
    dataset.to_npy(path)
    nparr = np.load(path, allow_pickle=True)
    numpy_arr = dataset.to_ddf().compute().to_numpy()
    assert (nparr == numpy_arr).all()


@pytest.mark.parametrize("append", [True, False])
@pytest.mark.parametrize("engine", ["csv", "parquet"])
def test_dask_df_array_npy_append(tmpdir, datasets, engine, append):
    df = make_df(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "embed_1": [1, 2, 3, 4, 5, 6],
            "embed_2": [1, 2, 3, 4, 5, 6],
            "embed_3": [1, 2, 3, 4, 5, 6],
        }
    )
    dataset = Dataset(df)
    path = str(tmpdir / "result.npy")
    dataset.to_npy(path, append=append)
    nparr = np.load(path, allow_pickle=True)
    numpy_arr = dataset.to_ddf().compute().to_numpy()
    assert (nparr == numpy_arr).all()


@pytest.mark.parametrize("append", [True, False])
@pytest.mark.parametrize("engine", ["csv", "parquet"])
def test_dask_df_array_npy_append_list(tmpdir, datasets, engine, append):
    df = make_df(
        {"id": [1, 2, 3, 4], "embedings": [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]}
    )
    dataset = Dataset(df, cpu=True)
    path = str(tmpdir / "result.npy")
    dataset.to_npy(path, append=append)
    nparr = np.load(path, allow_pickle=True)
    ddf = dataset.to_ddf().compute()
    numpy_arr = dataframe_columnwise_explode(ddf).to_numpy()
    assert (nparr == numpy_arr).all()


@pytest.mark.skipif(not cudf, reason="requires cuDF")
def test_to_ddf_incompatible_cluster():
    """Check that if we fail if the Dataset.to_ddf returns a dask_cudf.DataFrame
    in a context where the global dask client is a `LocalCluster`"""
    df = cudf.DataFrame({"col": [1, 2, 3]})
    dataset = Dataset(df)
    with Distributed(cluster_type="cpu"):
        with pytest.raises(RuntimeError) as exc_info:
            dataset.to_ddf()
    assert "`dask_cudf.DataFrame` is incompatible with `distributed.LocalCluster`." in str(
        exc_info.value
    )

def test_to_ddf_compatible_cluster():
    df = make_df({"col": [1, 2, 3]})
    dataset = Dataset(df)
    with Distributed():
        dataset.to_ddf()
