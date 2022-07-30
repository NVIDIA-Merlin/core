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
import pandas as pd
from dask.dataframe import assert_eq

import merlin.io


def test_dataset_to_parquet_from_parquet(tmpdir):
    input_path = tmpdir / "input"
    output_path = tmpdir / "output"

    input_path.mkdir()
    output_path.mkdir()

    # write out a parquet dataset composed of a couple of files
    input_parquet_files = [str(input_path / "0.parquet"), str(input_path / "1.parquet")]
    for filename in input_parquet_files:
        df = pd.DataFrame({"x": [1, 2, 3]})
        df.to_parquet(filename)

    # load up the input dataset, and write out again immediately
    ds = merlin.io.Dataset(input_parquet_files, engine="parquet")
    ds.to_parquet(output_path)

    reloaded_ds = merlin.io.Dataset(output_path, engine="parquet")
    assert_eq(reloaded_ds.compute(), ds.compute())
