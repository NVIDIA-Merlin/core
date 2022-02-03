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
import os

import numpy as np
import pandas as pd
import pytest
from dask.dataframe import assert_eq
from dask.dataframe.io.demo import names as name_list

import merlin

cudf = pytest.importorskip("cudf")
dask_cudf = pytest.importorskip("dask_cudf")

# Require uavro and fastavro library.
# Note that fastavro is only required to write
# avro files for testing, while uavro is actually
# used by AvroDatasetEngine.
fa = pytest.importorskip("fastavro")
pytest.importorskip("uavro")


@pytest.mark.parametrize("part_size", [None, "1KB"])
@pytest.mark.parametrize("size", [100, 5000])
@pytest.mark.parametrize("nfiles", [1, 2])
def test_avro_basic(tmpdir, part_size, size, nfiles):
    # Define avro schema
    schema = fa.parse_schema(
        {
            "name": "avro.example.User",
            "type": "record",
            "fields": [
                {"name": "name", "type": "string"},
                {"name": "age", "type": "int"},
            ],
        }
    )

    # Write avro dataset with two files.
    # Collect block and record (row) count while writing.
    nblocks = 0
    nrecords = 0
    paths = [os.path.join(str(tmpdir), f"test.{i}.avro") for i in range(nfiles)]
    records = []
    for path in paths:
        names = np.random.choice(name_list, size)
        ages = np.random.randint(18, 100, size)
        data = [{"name": names[i], "age": ages[i]} for i in range(size)]
        with open(path, "wb") as f:
            fa.writer(f, schema, data)
        with open(path, "rb") as fo:
            avro_reader = fa.block_reader(fo)
            for block in avro_reader:
                nrecords += block.num_records
                nblocks += 1
                records += list(block)
    if nfiles == 1:
        paths = paths[0]

    # Read back with dask.dataframe
    df = merlin.io.Dataset(paths, part_size=part_size, engine="avro").to_ddf()

    # Check basic length and partition count
    if part_size == "1KB":
        assert df.npartitions == nblocks
    assert len(df) == nrecords

    # Full comparison
    expect = pd.DataFrame.from_records(records)
    expect["age"] = expect["age"].astype("int32")
    assert_eq(df.compute().reset_index(drop=True), expect)
