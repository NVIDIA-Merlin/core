#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from dask.dataframe import assert_eq as assert_eq_dd

import merlin.dtypes as md
from merlin.dag import Graph
from merlin.dag.executors import DaskExecutor
from merlin.dag.ops import UDF, Rename
from merlin.dag.selector import ColumnSelector
from merlin.io import Dataset
from merlin.schema import Tags, TagSet

try:
    import cupy as cp

    _CPU = [True, False]
    _HAS_GPU = True
except ImportError:
    _CPU = [True]
    _HAS_GPU = False
    cp = None


@pytest.mark.parametrize("gpu_memory_frac", [0.1])
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("cpu", _CPU)
def test_udf(tmpdir, df, paths, gpu_memory_frac, engine, cpu):
    dataset = Dataset(paths, cpu=cpu)
    df_copy = df.copy()

    # Substring
    # Replacement
    substring = ColumnSelector(["name-cat", "name-string"]) >> UDF(lambda col: col.str.slice(1, 3))
    graph = Graph(substring)
    graph.construct_schema(dataset.schema)
    executor = DaskExecutor()
    executor.fit(dataset.to_ddf(), [graph.output_node])
    new_gdf = executor.transform(dataset.to_ddf(), graph).compute()

    assert_eq_dd(new_gdf["name-cat"], df_copy["name-cat"].str.slice(1, 3), check_index=False)
    assert_eq_dd(
        new_gdf["name-string"],
        df_copy["name-string"].str.slice(1, 3),
        check_index=False,
    )

    # No Replacement from old API (skipped for other examples)
    substring = (
        ColumnSelector(["name-cat", "name-string"])
        >> UDF(lambda col: col.str.slice(1, 3))
        >> Rename(postfix="_slice")
    )
    graph = Graph(substring + ["name-cat", "name-string"])
    graph.construct_schema(dataset.schema)
    executor = DaskExecutor()
    executor.fit(dataset, [graph.output_node])
    new_gdf = executor.transform(dataset.to_ddf(), [graph.output_node]).compute()

    assert_eq_dd(
        new_gdf["name-cat_slice"],
        df_copy["name-cat"].str.slice(1, 3),
        check_index=False,
        check_names=False,
    )
    assert_eq_dd(
        new_gdf["name-string_slice"],
        df_copy["name-string"].str.slice(1, 3),
        check_index=False,
        check_names=False,
    )
    assert_eq_dd(new_gdf["name-cat"], df_copy["name-cat"], check_index=False)
    assert_eq_dd(new_gdf["name-string"], df_copy["name-string"], check_index=False)

    # Replace
    # Replacement
    udf_op = ColumnSelector(["name-cat", "name-string"]) >> UDF(
        lambda col: col.str.replace("e", "XX")
    )
    graph = Graph(udf_op)
    graph.construct_schema(dataset.schema)
    executor = DaskExecutor()
    executor.fit(dataset, [graph.output_node])
    new_gdf = executor.transform(dataset.to_ddf(), graph).compute()

    assert_eq_dd(
        new_gdf["name-cat"],
        df_copy["name-cat"].str.replace("e", "XX"),
        check_index=False,
    )
    assert_eq_dd(
        new_gdf["name-string"],
        df_copy["name-string"].str.replace("e", "XX"),
        check_index=False,
    )

    # astype
    # Replacement
    udf_op = ColumnSelector(["id"]) >> UDF(lambda col: col.astype(float))
    graph = Graph(udf_op)
    graph.construct_schema(dataset.schema)
    executor = DaskExecutor()
    executor.fit(dataset, [graph.output_node])
    new_gdf = executor.transform(dataset.to_ddf(), graph).compute()

    assert new_gdf["id"].dtype == "float64"


@pytest.mark.parametrize("cpu", _CPU)
def test_udf_misalign(cpu):
    size = 12
    df0 = pd.DataFrame(
        {
            "a": np.arange(size),
            "b": np.random.choice(["apple", "banana", "orange"], size),
            "c": np.random.choice([0, 1], size),
        }
    )

    ddf0 = dd.from_pandas(df0, npartitions=4)

    cont_names = ColumnSelector(["a"])
    cat_names = ColumnSelector(["b"])
    label = ColumnSelector(["c"])
    if cpu:
        label_feature = label >> UDF(lambda col: np.where(col == 4, 0, 1))
    else:
        label_feature = label >> UDF(lambda col: cp.where(col == 4, 0, 1))

    dataset = Dataset(ddf0, cpu=cpu)

    graph = Graph(cat_names + cont_names + label_feature)
    graph.construct_schema(dataset.schema)

    executor = DaskExecutor()
    executor.fit(dataset, [graph.output_node])

    transformed = executor.transform(dataset.to_ddf(), graph)
    assert_eq_dd(
        df0[["a", "b"]],
        transformed.compute()[["a", "b"]],
        check_index=False,
    )


@pytest.mark.parametrize("cpu", _CPU)
def test_udf_schema_computation(cpu):
    size = 12
    df0 = pd.DataFrame(
        {
            "a": np.arange(size),
            "b": np.random.choice(["apple", "banana", "orange"], size),
            "c": np.random.choice([0, 1], size),
        }
    )
    ddf0 = dd.from_pandas(df0, npartitions=4)
    dataset = Dataset(ddf0, cpu=cpu)

    expected_dtype = np.float64
    expected_tags = [Tags.TARGET]
    expected_props = {"prop1": True}

    label = ColumnSelector(["c"])
    label_feature = label >> UDF(
        lambda col: col.astype(expected_dtype),
        dtype=expected_dtype,
        tags=expected_tags,
        properties=expected_props,
    )
    graph = Graph(label_feature)
    graph.construct_schema(dataset.schema)
    executor = DaskExecutor()
    executor.fit(dataset, [graph.output_node])

    output_schema = graph.output_node.output_schema

    assert output_schema["c"].dtype == md.dtype(expected_dtype)
    assert output_schema["c"].tags == TagSet(expected_tags)
    assert output_schema["c"].properties == expected_props


@pytest.mark.parametrize("cpu", _CPU)
def test_udf_dtype_propagation(cpu):
    size = 12
    df0 = pd.DataFrame(
        {
            "a": np.arange(size),
            "b": np.random.choice(["apple", "banana", "orange"], size),
            "c": np.random.choice([0, 1], size).astype(np.float32),
        }
    )
    ddf0 = dd.from_pandas(df0, npartitions=4)
    dataset = Dataset(ddf0, cpu=cpu)

    expected_dtype = np.float64

    label = ColumnSelector(["c"])
    label_feature = (
        label >> UDF(lambda col: col.astype(expected_dtype)) >> Rename(postfix="_renamed")
    )
    graph = Graph(label_feature)
    graph.construct_schema(dataset.schema)

    executor = DaskExecutor()
    executor.fit(dataset, [graph.output_node])

    output_schema = graph.output_node.output_schema

    assert output_schema["c_renamed"].dtype == md.dtype(expected_dtype)


@pytest.mark.parametrize("cpu", _CPU)
def test_udf_dtype_multi_op_propagation(cpu):
    size = 12
    df0 = pd.DataFrame(
        {
            "a": np.arange(size),
            "b": np.random.choice(["apple", "banana", "orange"], size),
            "c": np.random.choice([0, 1], size).astype(np.float16),
        }
    )
    ddf0 = dd.from_pandas(df0, npartitions=4)
    dataset = Dataset(ddf0, cpu=cpu)

    label = ColumnSelector(["a", "c"])

    label_feature = label >> UDF(lambda col: col.astype(np.float32)) >> Rename(postfix="_1st")
    b_labels = (
        label_feature["c_1st"] >> UDF(lambda col: col.astype(np.float64)) >> Rename(postfix="_2nd")
    )

    graph = Graph(b_labels)
    graph.construct_schema(dataset.schema)

    executor = DaskExecutor()
    executor.fit(dataset, [graph.output_node])

    output_schema = graph.output_node.output_schema

    assert output_schema["c_1st_2nd"].dtype == md.dtype(np.float64)
