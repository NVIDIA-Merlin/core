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
import pathlib
from typing import Union

import fsspec
import numpy

from ..schema import ColumnSchema
from ..schema import Schema as MerlinSchema
from ..tags import Tags
from . import proto_utils, schema_bp
from .schema_bp import Feature, FeatureType, FixedShape, FloatDomain, IntDomain
from .schema_bp import Schema as ProtoSchema
from .schema_bp import ValueCount

DOMAIN_ATTRS = {FeatureType.INT: "int_domain", FeatureType.FLOAT: "float_domain"}
FEATURE_TYPES = {
    "int": FeatureType.INT,
    "uint": FeatureType.INT,
    "float": FeatureType.FLOAT,
}


class TensorflowMetadata:
    def __init__(self, schema: ProtoSchema = None):
        self.proto_schema = schema

    @classmethod
    def from_json(cls, json: Union[str, bytes]) -> "TensorflowMetadata":
        schema = ProtoSchema().from_json(json)
        return TensorflowMetadata(schema)

    @classmethod
    def from_json_file(cls, path: str) -> "TensorflowMetadata":
        return cls.from_json(_read_file(path))

    @classmethod
    def from_proto_text(cls, path_or_proto_text: str) -> "TensorflowMetadata":
        from tensorflow_metadata.proto.v0 import schema_pb2

        schema = proto_utils.proto_text_to_better_proto(
            ProtoSchema(), path_or_proto_text, schema_pb2.Schema()
        )

        return TensorflowMetadata(schema)

    @classmethod
    def from_proto_text_file(cls, path: str, file_name="schema.pbtxt") -> "TensorflowMetadata":
        path = pathlib.Path(path) / file_name
        return cls.from_proto_text(_read_file(str(path)))

    def to_proto_text(self) -> str:
        from tensorflow_metadata.proto.v0 import schema_pb2

        return proto_utils.better_proto_to_proto_text(self.proto_schema, schema_pb2.Schema())

    def to_proto_text_file(self, path: str, file_name="schema.pbtxt") -> "TensorflowMetadata":
        return _write_file(self.to_proto_text(), path, file_name)

    def copy(self, **kwargs) -> "TensorflowMetadata":
        schema_copy = proto_utils.copy_better_proto_message(self.proto_schema, **kwargs)
        return TensorflowMetadata(schema_copy)

    @classmethod
    def from_merlin_schema(cls, schema: MerlinSchema):
        features = []
        for col_name, col_schema in schema.column_schemas.items():
            features.append(pb_feature(col_schema))

        proto_schema = ProtoSchema(feature=features)

        return TensorflowMetadata(proto_schema)

    def to_merlin_schema(self):
        merlin_schema = MerlinSchema()

        for feature in self.proto_schema.feature:
            col_schema = merlin_column(feature)
            merlin_schema.column_schemas[col_schema.name] = col_schema

        return merlin_schema

    def to_json(self):
        return self.proto_schema.to_json()


def pb_int_domain(column_schema):
    domain = column_schema.properties.get("domain")
    if domain is None:
        return None

    return IntDomain(
        name=domain.get("name", column_schema.name),
        min=domain.get("min", None),
        max=domain.get("max", None),
        is_categorical=(
            Tags.CATEGORICAL in column_schema.tags or Tags.CATEGORICAL.value in column_schema.tags
        ),
    )


def pb_float_domain(column_schema):
    domain = column_schema.properties.get("domain")
    if domain is None:
        return None
    return FloatDomain(
        name=column_schema.name,
        min=domain.get("min", None),
        max=domain.get("max", None),
    )


def _dtype_name(column_schema):
    # TODO: Decide if we need this since we've standardized on numpy types
    if hasattr(column_schema.dtype, "kind"):
        return numpy.core._dtype._kind_name(column_schema.dtype)
    elif hasattr(column_schema.dtype, "item"):
        return type(column_schema.dtype(1).item()).__name__
    elif isinstance(column_schema.dtype, str):
        return column_schema.dtype
    elif hasattr(column_schema.dtype, "__name__"):
        return column_schema.dtype.__name__
    else:
        raise TypeError(f"unsupported dtype for column schema: {column_schema.dtype}")


def pb_extra_metadata(column_schema):
    properties = {
        k: v for k, v in column_schema.properties.items() if k not in ("domain", "value_count")
    }
    properties["dtype_item_size"] = numpy.dtype(column_schema.dtype).itemsize * 8
    properties["is_list"] = column_schema.is_list
    properties["is_ragged"] = column_schema.is_ragged
    return schema_bp.Any().from_dict(properties)


def pb_tag(column_schema):
    return [tag.value if hasattr(tag, "value") else tag for tag in column_schema.tags]


def pb_feature(column_schema):
    feature = Feature(name=column_schema.name)

    feature = set_feature_domain(feature, column_schema)

    if column_schema.is_list:
        value_count = column_schema.properties.get("value_count", {})
        min_length = value_count.get("min")
        max_length = value_count.get("max")

        if min_length and max_length and min_length == max_length:
            feature.shape = FixedShape(min_length)
        elif min_length and max_length and min_length < max_length:
            feature.value_count = ValueCount(min=min_length, max=max_length)
        else:
            feature.value_count = ValueCount(min=0, max=0)

    feature.annotation.tag = pb_tag(column_schema)
    feature.annotation.extra_metadata.append(pb_extra_metadata(column_schema))
    return feature


def set_feature_domain(feature, column_schema):
    DOMAIN_CONSTRUCTORS = {
        FeatureType.INT: pb_int_domain,
        FeatureType.FLOAT: pb_float_domain,
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


def merlin_domain(feature):
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
            name = domain_value.name
            if name != feature.name:
                domain["name"] = domain_value.name

    return domain


def merlin_value_count(feature):
    if proto_utils.has_field(feature, "value_count"):
        value_count = feature.value_count
        if value_count.min != value_count.max != 0:
            return {"min": value_count.min, "max": value_count.max}


def merlin_properties(feature):
    extra_metadata = feature.annotation.extra_metadata
    if len(extra_metadata) > 1:
        raise ValueError(
            f"{feature.name}: extra_metadata should have 1 item, has \
            {len(feature.annotation.extra_metadata)}"
        )
    elif len(extra_metadata) == 1:
        properties = feature.annotation.extra_metadata[0].value

        if isinstance(properties, bytes):
            properties = schema_bp.Any(value=properties).to_dict()

    else:
        properties = {}

    domain = merlin_domain(feature)
    if domain:
        properties["domain"] = domain

    value_count = merlin_value_count(feature)
    if value_count:
        properties["value_count"] = value_count
        properties["is_list"] = True
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


def merlin_dtype(feature, properties):
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


def merlin_column(feature):
    name = feature.name
    tags = list(feature.annotation.tag) or []
    properties = merlin_properties(feature)
    dtype = merlin_dtype(feature, properties)

    is_list = properties.pop("is_list", False)
    is_ragged = properties.pop("is_ragged", False)
    properties.pop("dtype_item_size", False)

    domain = properties.get("domain")
    if domain and domain.pop("is_categorical", False):
        if Tags.CATEGORICAL not in tags:
            tags.append(Tags.CATEGORICAL)

    return ColumnSchema(name, tags, properties, dtype, is_list, is_ragged=is_ragged)


def _read_file(path: str):
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
