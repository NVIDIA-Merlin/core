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
import numpy as np

from merlin.array.features import Feature, FeatureCollection
from merlin.schema import ColumnSchema, Schema, Tags


def test_select_features_by_name():
    schema = Schema(
        [
            ColumnSchema("a", tags=["a"]),
            ColumnSchema("b", tags=["a"]),
            ColumnSchema("c", tags=["c"]),
        ]
    )
    value = {"a": np.array([1, 2, 3]), "b": np.array([3, 4, 5]), "c": np.array([6, 7, 8])}
    fc = FeatureCollection(schema, value)
    fc_b = fc.select_by_name("b")
    fc_bc = fc.select_by_name(["b", "c"])

    assert fc_b["b"].schema.name in "b"
    assert (fc_b["b"].value.data == value["b"]).all()

    assert fc_bc["b"].schema.name in "b"
    assert (fc_bc["b"].value.data == value["b"]).all()

    assert fc_bc["c"].schema.name in "c"
    assert (fc_bc["c"].value.data == value["c"]).all()


def test_select_features_by_tag():
    schema = Schema(
        [
            ColumnSchema("a", tags=[Tags.CATEGORICAL]),
            ColumnSchema("b", tags=[Tags.CATEGORICAL]),
            ColumnSchema("c", tags=[Tags.CONTINUOUS]),
        ]
    )

    value = {"a": np.array([1, 2, 3]), "b": np.array([3, 4, 5]), "c": np.array([6, 7, 8])}

    features = FeatureCollection(schema, value)

    categorical = features.select_by_tag(Tags.CATEGORICAL)
    continuous = features.select_by_tag(Tags.CONTINUOUS)

    for feature_name in ["a", "b"]:
        assert categorical[feature_name].schema.name == feature_name
        assert (categorical[feature_name].value.data == value[feature_name]).all()

    assert continuous["c"].schema.name == "c"
    assert (continuous["c"].value.data == value["c"]).all()


def test_update_feature_schemas():
    schema = Schema(["a"])
    value = {"a": np.array([1.0, 2.0])}
    features = FeatureCollection(schema, value)

    new_schema = Schema([schema.column_schemas["a"].with_tags("updated")])
    updated_features = features.with_schema(new_schema)

    assert updated_features.schema == new_schema


def test_get_feature():
    schema = Schema(["a"])
    values = {"a": np.array([1.0, 2.0])}
    features = FeatureCollection(schema, values)

    feature = features["a"]

    assert isinstance(feature, Feature)
    assert feature.schema.name == "a"
    assert (feature.value.data == values["a"]).all()
