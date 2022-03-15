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
import json

import numpy
import pytest

from merlin.schema import ColumnSchema, Schema
from merlin.schema.io.tensorflow_metadata import TensorflowMetadata
from merlin.schema.tags import Tags, TagSet


@pytest.mark.parametrize("d_types", [numpy.float32, numpy.float64, numpy.uint32, numpy.uint64])
def test_dtype_column_schema(d_types):
    column = ColumnSchema("name", tags=[], properties=[], dtype=d_types)
    assert column.dtype == d_types


def test_column_schema_meta():
    column = ColumnSchema("name", tags=["tag-1"], properties={"p1": "prop-1"})

    assert column.name == "name"
    assert "tag-1" in column.tags
    assert column.with_name("a").name == "a"
    assert set(column.with_tags("tag-2").tags) == set(["tag-1", "tag-2"])
    assert column.with_properties({"p2": "prop-2"}).properties == {
        "p1": "prop-1",
        "p2": "prop-2",
    }
    assert column.with_tags("tag-2").properties == {"p1": "prop-1"}
    assert set(column.with_properties({"p2": "prop-2"}).tags) == set(["tag-1"])

    assert column == ColumnSchema("name", tags=["tag-1"], properties={"p1": "prop-1"})
    # should not be the same no properties
    assert column != ColumnSchema("name", tags=["tag-1"])
    # should not be the same no tags
    assert column != ColumnSchema("name", properties={"p1": "prop-1"})


@pytest.mark.parametrize("props1", [{}, {"p1": "p1", "p2": "p2"}])
@pytest.mark.parametrize("props2", [{}, {"p3": "p3", "p4": "p4"}])
@pytest.mark.parametrize("tags1", [[], ["a", "b", "c"]])
@pytest.mark.parametrize("tags2", [[], ["c", "d", "e"]])
@pytest.mark.parametrize("d_type", [numpy.float32, numpy.float64, numpy.int32, numpy.int64])
@pytest.mark.parametrize("list_type", [True, False])
def test_column_schema_set_protobuf(tmpdir, props1, props2, tags1, tags2, d_type, list_type):
    # create a schema
    col_schema1 = ColumnSchema(
        "col1", tags=tags1, properties=props1, dtype=d_type, is_list=list_type
    )
    col_schema2 = ColumnSchema(
        "col2", tags=tags2, properties=props2, dtype=d_type, is_list=list_type
    )
    schema = Schema([col_schema1, col_schema2])

    # write schema out
    tf_metadata = TensorflowMetadata.from_merlin_schema(schema)
    tf_metadata.to_proto_text_file(str(tmpdir))

    # read schema back in
    tf_metadata = TensorflowMetadata.from_proto_text_file(str(tmpdir))
    loaded_schema = tf_metadata.to_merlin_schema()

    # compare read to origin
    assert schema == loaded_schema


@pytest.mark.parametrize("properties", [{}, {"domain": {"min": 0, "max": 10}}])
@pytest.mark.parametrize("tags", [[], ["a", "b", "c"]])
@pytest.mark.parametrize("dtype", [numpy.float, numpy.int])
@pytest.mark.parametrize("list_type", [True, False])
def test_schema_to_tensorflow_metadata(tmpdir, properties, tags, dtype, list_type):
    # make sure we can round trip a schema to TensorflowMetadata without going to disk
    schema = Schema(
        [ColumnSchema("col", tags=tags, properties=properties, dtype=dtype, is_list=list_type)]
    )
    loaded_schema = TensorflowMetadata.from_merlin_schema(schema).to_merlin_schema()
    assert schema == loaded_schema


@pytest.mark.parametrize("properties", [{}, {"domain": {"min": 0, "max": 10}}])
@pytest.mark.parametrize("tags", [[], ["a", "b", "c"]])
@pytest.mark.parametrize("dtype", [numpy.float, numpy.int])
@pytest.mark.parametrize("list_type", [True, False])
def test_schema_to_tensorflow_metadata_json(tmpdir, properties, tags, dtype, list_type):
    schema = Schema(
        [ColumnSchema("col", tags=tags, properties=properties, dtype=dtype, is_list=list_type)]
    )
    tf_metadata_json = TensorflowMetadata.from_merlin_schema(schema).to_json()
    loaded_schema = TensorflowMetadata.from_json(tf_metadata_json).to_merlin_schema()
    assert schema == loaded_schema


def test_tensorflow_metadata_from_json():
    # make sure we can load up tensorflowmetadata serialized json objects, like done by
    # merlin-models
    json_schema = """{"feature": [
    {
      "name": "categories",
      "valueCount": {
        "min": "1",
        "max": "4"
      },
      "type": "INT",
      "intDomain": {
        "name": "categories",
        "min": "1",
        "max": "331",
        "isCategorical": true
      },
      "annotation": {
        "tag": [
          "item"
        ]
      }
    }]}
    """

    schema = TensorflowMetadata.from_json(json_schema).to_merlin_schema()
    column_schema = schema.column_schemas["categories"]

    # make sure the value_count is set appropriately
    assert column_schema.properties["value_count"] == {"min": 1, "max": 4}
    assert column_schema.is_list
    assert column_schema.is_ragged

    # should have CATEGORICAL tag, even though not explicitly listed in annotation
    # (and instead should be inferred from the intDomain.isCategorical)
    assert Tags.CATEGORICAL in column_schema.tags

    assert column_schema.properties["domain"] == {"min": 1, "max": 331}

    # make sure the JSON formatted extra_metadata properties are human readable
    json_schema = json.loads(TensorflowMetadata.from_merlin_schema(schema).to_json())
    assert json_schema["feature"][0]["annotation"]["extraMetadata"] == [
        {"is_list": True, "is_ragged": True, "dtype_item_size": 64.0}
    ]


def test_column_schema_protobuf_domain_check(tmpdir):
    # create a schema
    schema1 = ColumnSchema(
        "col1",
        tags=[],
        properties={"domain": {"min": 0, "max": 10}},
        dtype=numpy.int,
        is_list=False,
    )
    schema2 = ColumnSchema(
        "col2",
        tags=[],
        properties={"domain": {"min": 0.0, "max": 10.0}},
        dtype=numpy.float,
        is_list=False,
    )
    saved_schema = Schema([schema1, schema2])

    # write schema out
    tf_metadata = TensorflowMetadata.from_merlin_schema(saved_schema)
    tf_metadata.to_proto_text_file(str(tmpdir))

    # read schema back in
    tf_metadata = TensorflowMetadata.from_proto_text_file(str(tmpdir))
    loaded_schema = tf_metadata.to_merlin_schema()

    assert saved_schema == loaded_schema


def test_column_schema_tags_normalize():
    schema1 = ColumnSchema("col1", tags=["categorical", "list", "item_id"])
    assert schema1.tags == TagSet([Tags.CATEGORICAL, Tags.LIST, Tags.ITEM_ID])


def test_dataset_schema_constructor():
    schema1 = ColumnSchema("col1", tags=["a", "b", "c"])
    schema2 = ColumnSchema("col2", tags=["c", "d", "e"])

    expected = {schema1.name: schema1, schema2.name: schema2}

    ds_schema_dict = Schema(expected)
    ds_schema_list = Schema([schema1, schema2])

    assert ds_schema_dict.column_schemas == expected
    assert ds_schema_list.column_schemas == expected

    with pytest.raises(TypeError) as exception_info:
        Schema(12345)

    assert "column_schemas" in str(exception_info.value)


def test_dataset_schema_select_by_tag():
    schema1 = ColumnSchema("col1", tags=["a", "b", "c"])
    schema2 = ColumnSchema("col2", tags=["b", "c", "d"])

    ds_schema = Schema([schema1, schema2])

    selected_schema1 = ds_schema.select_by_tag("a")
    selected_schema2 = ds_schema.select_by_tag("d")

    assert selected_schema1.column_schemas == {"col1": schema1}
    assert selected_schema2.column_schemas == {"col2": schema2}

    selected_schema_both = ds_schema.select_by_tag("c")
    selected_schema_neither = ds_schema.select_by_tag("e")
    selected_schema_multi = ds_schema.select_by_tag(["b", "c"])

    assert selected_schema_both.column_schemas == {"col1": schema1, "col2": schema2}
    assert selected_schema_neither.column_schemas == {}
    assert selected_schema_multi.column_schemas == {"col1": schema1, "col2": schema2}


def test_dataset_schema_select_by_name():
    schema1 = ColumnSchema("col1", tags=["a", "b", "c"])
    schema2 = ColumnSchema("col2", tags=["b", "c", "d"])

    ds_schema = Schema([schema1, schema2])

    selected_schema1 = ds_schema.select_by_name("col1")
    selected_schema2 = ds_schema.select_by_name("col2")

    assert selected_schema1.column_schemas == {"col1": schema1}
    assert selected_schema2.column_schemas == {"col2": schema2}

    selected_schema_multi = ds_schema.select_by_name(["col1", "col2"])

    assert selected_schema_multi.column_schemas == {"col1": schema1, "col2": schema2}

    assert ds_schema.select_by_name("col3") == Schema([])


def test_dataset_schemas_can_be_added():
    ds1_schema = Schema([ColumnSchema("col1"), ColumnSchema("col2")])
    ds2_schema = Schema([ColumnSchema("col3"), ColumnSchema("col4")])

    result = ds1_schema + ds2_schema

    expected = Schema(
        [
            ColumnSchema("col1"),
            ColumnSchema("col2"),
            ColumnSchema("col3"),
            ColumnSchema("col4"),
        ]
    )

    assert result == expected


def test_schema_can_be_added_to_none():
    schema_set = Schema(["a", "b", "c"])

    assert (schema_set + None) == schema_set
    assert (None + schema_set) == schema_set


def test_construct_schema_with_column_names():
    schema = Schema(["x", "y", "z"])
    expected = Schema([ColumnSchema("x"), ColumnSchema("y"), ColumnSchema("z")])

    assert schema == expected


def test_dataset_schema_column_names():
    ds_schema = Schema(["x", "y", "z"])

    assert ds_schema.column_names == ["x", "y", "z"]
