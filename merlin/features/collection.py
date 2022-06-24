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
from dataclasses import dataclass
from typing import Dict, Protocol, Sequence, Union, runtime_checkable

from merlin.features.array import MerlinArray
from merlin.features.df import VirtualDataframe
from merlin.schema import ColumnSchema, Schema, Tags


@dataclass(frozen=True)
class Feature:
    """
    A feature containing its schema and data.
    """

    schema: ColumnSchema
    values: MerlinArray


@runtime_checkable
class Features(Protocol):
    """
    This Protocol matches either real Pandas/cuDF dataframes
    or the VirtualDataframe class defined above when an object
    is checked with `is_instance(obj, Features)` which returns
    `True` for any object that defines all of the methods below
    """

    @property
    def columns(self):
        ...

    def __getitem__(self, key):
        ...

    def __setitem__(self, key, value):
        ...

    def __iter__(self):  # pylint: disable=non-iterator-returned
        ...


class FeatureCollection:
    """
    A collection of features containing their schemas and data.
    """

    def __init__(self, schema: Schema, values: Features):
        self.schema: Schema = schema
        self.values: Features = values

        if len(self.schema) != len(self.values.columns):
            raise ValueError("Schema and values must have the same number of columns")

        if set(self.schema.column_names) != set(self.values.columns):
            raise ValueError("Schema and Values have different column names.")

    @classmethod
    def from_values_dict(cls, schema: Schema, values: Dict):
        return FeatureCollection(schema, VirtualDataframe(values))

    def with_schema(self, schema: Schema) -> "FeatureCollection":
        """
        Create a new FeatureCollection with the same data and an updated Schema.

        Parameters
        ----------
        schema : Schema
            Schema to be applied to FeatureCollection

        Returns
        -------
        FeatureCollection
            New collection of features with updated Schema
        """
        return FeatureCollection(schema, self.values)

    def select_by_name(self, names: Union[str, Sequence[str]]) -> "FeatureCollection":
        """
        Create a new FeatureCollection with only the features that match the provided names.

        Parameters
        ----------
        names : string, [string]
            Names of the features to select.

        Returns
        -------
        FeatureCollection
            A collection of the features that match the provided names
        """
        sub_schema = self.schema.select_by_name(names)
        sub_values = self.values[sub_schema.column_names]

        return FeatureCollection(sub_schema, sub_values)

    def select_by_tag(
        self, tags: Union[str, Tags, Sequence[str], Sequence[Tags]]
    ) -> "FeatureCollection":
        """
        Create a new FeatureCollection with only the features that match the provided tags.

        Parameters
        ----------
        tags: Union[str, Tags, Sequence[str], Sequence[Tags]]
            Tags or tag strings of the features to select
        Returns
        -------
        FeatureCollection
            A collection of the features that match the provided tags
        """
        sub_schema = self.schema.select_by_tag(tags)
        sub_values = self.values[sub_schema.column_names]

        return FeatureCollection(sub_schema, sub_values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, feature_name: str) -> Feature:
        array = self.values[feature_name]

        if not isinstance(array, MerlinArray):
            array = MerlinArray.build(array)

        return Feature(self.schema.column_schemas[feature_name], array)

    def __iter__(self):
        for col_name in self.schema.column_names:
            yield self[col_name]
