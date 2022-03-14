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
                tags=[Tags.USER_ID, Tags.CATEGORICAL],
                properties={
                    "num_buckets": None,
                    "freq_threshold": 0.0,
                    "domain": {"min": 0, "max": 102987},
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
