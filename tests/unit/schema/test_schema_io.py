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

import merlin.dtypes as md
from merlin.schema import ColumnSchema, Schema, Tags
from merlin.schema.io.tensorflow_metadata import TensorflowMetadata


def test_json_serialization_with_embedded_dicts():
    # there were a couple issues with exporting schemas with embedded dictionaries or lists
    # to json:
    # https://github.com/NVIDIA-Merlin/models/issues/231
    # Verify that this works like we expect

    schema = Schema(
        [
            ColumnSchema(
                "userid",
                dtype=md.int32,
                tags=[Tags.USER_ID, Tags.CATEGORICAL],
                properties={
                    "embedding_sizes": {"cardinality": 102987, "dimension": 2.0},
                    "list_value": [{"dict_val": 1.0}],
                },
            )
        ]
    )
    json_schema = TensorflowMetadata.from_merlin_schema(schema).to_json()
    output_schema = TensorflowMetadata.from_json(json_schema).to_merlin_schema()
    assert output_schema == schema


def test_merlin_to_proto_to_json_to_merlin():
    # we used to have an issue going from merlin -> tensorflowmetadata-> proto_text ->
    # tensorflowmetadata -> json schema would throw an error like
    # TypeError: memoryview: a bytes-like object is required, not 'dict'
    # verify that this works as expected
    schema = Schema(
        [
            ColumnSchema(
                "userid",
                dtype=md.int32,
                tags=[Tags.USER_ID, Tags.CATEGORICAL],
                properties={
                    "num_buckets": None,
                    "freq_threshold": 0.0,
                    "domain": {"min": 0, "max": 102987, "name": "userid"},
                },
            )
        ]
    )
    tfm_schema = TensorflowMetadata.from_merlin_schema(schema)
    proto_schema = tfm_schema.to_proto_text()

    tfm_schema2 = TensorflowMetadata.from_proto_text(proto_schema)
    json_schema = tfm_schema2.to_json()

    tfm_schema2 = TensorflowMetadata.from_json(json_schema)
    output_schema = tfm_schema2.to_merlin_schema()

    assert output_schema == schema


@pytest.mark.parametrize(
    ["value_count", "expected_is_list", "expected_is_ragged"],
    [
        [{"min": 1, "max": 1}, True, False],
        [{"min": 1, "max": 2}, True, True],
        [{"max": 5}, True, True],
    ],
)
def test_value_count(value_count, expected_is_list, expected_is_ragged):
    schema = Schema(
        [
            ColumnSchema(
                "example",
                dtype=md.int32,
                is_list=True,
                properties={
                    "value_count": value_count,
                },
            )
        ]
    )
    assert schema["example"].is_list == expected_is_list
    assert schema["example"].is_ragged == expected_is_ragged

    json_schema = TensorflowMetadata.from_merlin_schema(schema).to_json()
    output_schema = TensorflowMetadata.from_json(json_schema).to_merlin_schema()
    assert output_schema == schema


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

    assert column_schema.properties["domain"] == {"min": 1, "max": 331, "name": "categories"}

    # make sure the JSON formatted extra_metadata properties are human readable
    json_schema = json.loads(TensorflowMetadata.from_merlin_schema(schema).to_json())
    assert json_schema["feature"][0]["annotation"]["extraMetadata"] == [
        {"is_list": True, "is_ragged": True, "dtype_item_size": 64.0}
    ]
