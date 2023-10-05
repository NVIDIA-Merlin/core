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
import os
import pathlib
from typing import Union

import fsspec
import numpy

from merlin.schema.io import merlin_schema, proto_utils
from merlin.schema.io.merlin_schema import Dtype, Field, FloatDomain, IntDomain
from merlin.schema.io.merlin_schema import Schema as ProtoSchema
from merlin.schema.schema import ColumnSchema
from merlin.schema.schema import Schema as MerlinSchema
from merlin.schema.tags import Tags


class MerlinMetadata:
    """
    Reads and writes Merlin schemas as `merlin-metadata` Protobuf and JSON files
    """

    def __init__(self, schema: ProtoSchema = None):
        self.proto_schema = schema

    @classmethod
    def from_json(cls, json: Union[str, bytes]) -> "MerlinMetadata":
        """Create a MerlinMetadata schema object from a JSON string

        Parameters
        ----------
        json : Union[str, bytes]
            The JSON string to parse

        Returns
        -------
        MerlinMetadata
            Schema object parsed from JSON

        """
        schema = ProtoSchema().from_json(json)
        return MerlinMetadata(schema)

    @classmethod
    def from_json_file(cls, path: os.PathLike) -> "MerlinMetadata":
        """Create a MerlinMetadata schema object from a JSON file

        Parameters
        ----------
        path : str
            Path to the JSON file to parse

        Returns
        -------
        MerlinMetadata
            Schema object parsed from JSON file

        """
        return cls.from_json(_read_file(path))

    def copy(self, **kwargs) -> "MerlinMetadata":
        """Create a copy of this MerlinMetadata schema object

        Returns
        -------
        MerlinMetadata
            Copy of this MerlinMetadata schema object

        """
        schema_copy = proto_utils.copy_better_proto_message(self.proto_schema, **kwargs)
        return MerlinMetadata(schema_copy)

    @classmethod
    def from_merlin_schema(cls, schema: MerlinSchema) -> "MerlinMetadata":
        """Convert a MerlinSchema object to a MerlinMetadata schema object

        Parameters
        ----------
        schema : MerlinSchema
            Schema object to convert

        Returns
        -------
        MerlinMetadata
            Schema converted to a MerlinMetadata schema object

        """
        fields = []
        for col_name, col_schema in schema.column_schemas.items():
            fields.append(_pb_feature(col_schema))

        proto_schema = ProtoSchema(fields=fields)

        return MerlinMetadata(proto_schema)

    def to_merlin_schema(self) -> MerlinSchema:
        """Convert this MerlinMetadata schema object to a MerlinSchema object

        Returns
        -------
        MerlinSchema
            Schema converted to MerlinSchema object

        """
        merlin_schema = MerlinSchema()

        for field in self.proto_schema.fields:
            col_schema = _merlin_column(field)
            merlin_schema.column_schemas[col_schema.name] = col_schema

        return merlin_schema

    def to_json(self) -> str:
        """Convert this MerlinMetadata schema object to a JSON string

        Returns
        -------
        str
            Schema converted to a JSON string

        """
        return self.proto_schema.to_json()


def _pb_int_domain(column_schema):
    domain = column_schema.properties.get("domain")
    if domain is None:
        return None

    return IntDomain(
        name=domain.get("name", None),
        min=domain.get("min", None),
        max=domain.get("max", None),
        is_categorical=(
            Tags.CATEGORICAL in column_schema.tags or Tags.CATEGORICAL.value in column_schema.tags
        ),
    )


def _pb_float_domain(column_schema):
    domain = column_schema.properties.get("domain")
    if domain is None:
        return None
    return FloatDomain(
        name=domain.get("name", None),
        min=domain.get("min", None),
        max=domain.get("max", None),
    )


def _pb_extra_metadata(column_schema):
    properties = {
        k: v for k, v in column_schema.properties.items() if k not in ("domain", "value_count")
    }
    return merlin_schema.Any().from_dict(properties)


def _pb_tag(column_schema):
    return [tag.value if hasattr(tag, "value") else tag for tag in column_schema.tags]


def _pb_feature(column_schema):
    feature = Feature(name=column_schema.name)

    feature = _set_feature_domain(feature, column_schema)

    value_count = column_schema.properties.get("value_count", {})
    if value_count:
        min_length = value_count.get("min", 0)
        max_length = value_count.get("max", 0)
        feature.value_count = ValueCount(min=min_length, max=max_length)

    feature.annotation.tag = _pb_tag(column_schema)
    feature.annotation.extra_metadata.append(_pb_extra_metadata(column_schema))
    return feature


def _set_feature_domain(feature, column_schema):
    DOMAIN_CONSTRUCTORS = {
        FeatureType.INT: _pb_int_domain,
        FeatureType.FLOAT: _pb_float_domain,
    }

    pb_type = FEATURE_TYPES.get(_dtype_name(column_schema))
    if pb_type:
        feature.type = pb_type

        domain_attr = DOMAIN_ATTRS[pb_type]
        domain_fn = DOMAIN_CONSTRUCTORS[pb_type]
        domain = domain_fn(column_schema)
        if domain:
            setattr(feature, domain_attr, domain)

    return feature


def _merlin_domain(feature):
    domain = {}

    domain_attr = DOMAIN_ATTRS.get(feature.type)

    if domain_attr and proto_utils.has_field(feature, domain_attr):
        domain_value = getattr(feature, domain_attr)
        if hasattr(domain_value, "min") and hasattr(domain_value, "max"):
            domain["min"] = domain_value.min
            domain["max"] = domain_value.max

        if hasattr(domain_value, "is_categorical"):
            domain["is_categorical"] = domain_value.is_categorical

        if hasattr(domain_value, "name"):
            if domain_value.name:
                domain["name"] = domain_value.name

    return domain


def _merlin_value_count(feature):
    if proto_utils.has_field(feature, "value_count"):
        value_count = feature.value_count
        value_count_dict = {}
        if value_count.min > 0:
            value_count_dict["min"] = value_count.min
        if value_count.max > 0:
            value_count_dict["max"] = value_count.max
        return value_count_dict


def _merlin_properties(feature):
    extra_metadata = feature.annotation.extra_metadata
    if len(extra_metadata) > 1:
        raise ValueError(
            f"{feature.name}: extra_metadata should have 1 item, has \
            {len(feature.annotation.extra_metadata)}"
        )
    elif len(extra_metadata) == 1:
        properties = feature.annotation.extra_metadata[0].value

        if isinstance(properties, bytes):
            properties = merlin_schema.Any(value=properties).to_dict()

    else:
        properties = {}

    domain = _merlin_domain(feature)

    if domain:
        properties["domain"] = domain

    value_count = _merlin_value_count(feature)

    if value_count:
        properties["value_count"] = value_count
        properties["is_list"] = value_count.get("min", 0) > 0 or value_count.get("max", 0) > 0
        properties["is_ragged"] = value_count.get("min") != value_count.get("max")

    return properties


int_dtypes_map = {
    8: numpy.int8,
    16: numpy.int16,
    32: numpy.int32,
    64: numpy.int64,
}


float_dtypes_map = {
    16: numpy.float16,
    32: numpy.float32,
    64: numpy.float64,
}


def _merlin_dtype(feature, properties):
    dtype = None
    item_size = int(properties.get("dtype_item_size", 0)) or None
    if feature.type == FeatureType.INT:
        if item_size and item_size in int_dtypes_map:
            dtype = int_dtypes_map[item_size]
        else:
            dtype = numpy.int
    elif feature.type == FeatureType.FLOAT:
        if item_size and item_size in float_dtypes_map:
            dtype = float_dtypes_map[item_size]
        else:
            dtype = numpy.float
    return dtype


def _merlin_column(feature):
    name = feature.name
    tags = list(feature.annotation.tag) or []
    properties = _merlin_properties(feature)
    dtype = _merlin_dtype(feature, properties)

    is_list = properties.pop("is_list", False)
    is_ragged = properties.pop("is_ragged", False)
    properties.pop("dtype_item_size", False)

    domain = properties.get("domain")
    if domain and domain.pop("is_categorical", False):
        if Tags.CATEGORICAL not in tags:
            tags.append(Tags.CATEGORICAL)

    return ColumnSchema(name, tags, properties, dtype, is_list, is_ragged=is_ragged)


def _read_file(path: os.PathLike):
    # TODO: Should we be using fsspec here too?
    path = pathlib.Path(path)
    if path.is_file():
        with open(path, "r") as f:
            contents = f.read()
    else:
        raise ValueError("Path is not file")

    return contents


def _write_file(contents: str, path: str, filename: str):
    fs = fsspec.get_fs_token_paths(path)[0]

    try:
        with fs.open(fs.sep.join([str(path), filename]), "w") as f:
            f.write(contents)
    except Exception as e:
        if not fs.isdir(path):
            raise ValueError(f"The path provided is not a valid directory: {path}") from e
        raise
