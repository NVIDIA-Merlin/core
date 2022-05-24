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
from typing import Dict, List, Protocol, Sequence, Union, runtime_checkable

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cudf as cf
except ImportError:
    cf = None

from merlin.features.array.base import MerlinArray
from merlin.features.array.cudf import MerlinCudfArray
from merlin.features.array.cupy import MerlinCupyArray
from merlin.features.array.numpy import MerlinNumpyArray
from merlin.features.array.tensorflow import MerlinTensorflowArray
from merlin.schema import ColumnSchema, Schema, Tags


@dataclass(frozen=True)
class Feature:
    """
    A feature containing its schema and data.
    """

    schema: ColumnSchema
    values: MerlinArray


class VirtualDataframe:
    def __init__(self, data=None):
        self._col_data = data or {}

    @property
    def columns(self) -> List[str]:
        return list(self._col_data.keys())

    def __getitem__(self, col_items):
        if isinstance(col_items, list):
            results = {name: self._col_data[name] for name in col_items}
            return VirtualDataframe(results)
        else:
            return self._col_data[col_items]

    def __len__(self):
        return len(self.columns)

    def __iter__(self):
        for name, tensor in self._col_data.items():
            yield name, tensor

    def __repr__(self):
        dict_rep = {}
        for k, v in self._col_data.items():
            dict_rep[k] = v
        return str(dict_rep)


@runtime_checkable
class Features(Protocol):
    @property
    def columns(self):
        ...

    def __getitem__(self, key):
        ...


class FeatureCollection:
    """
    A collection of features containing their schemas and data.
    """

    def __init__(self, schema: Schema, values: Union[Dict, Features]):
        self.schema: Schema = schema
        self.values: Features = None

        if isinstance(values, dict):
            self.values = VirtualDataframe(values)
        else:
            self.values = values

        if len(self.schema) != len(self.values.columns):
            raise ValueError("Schema and values must have the same number of columns")

        if set(self.schema.column_names) != set(self.values.columns):
            raise ValueError("Schema and Values have different column names.")

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
        sub_values = {name: self.values[name] for name in sub_schema.column_names}

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
        sub_values = {name: self.values[name] for name in sub_schema.column_names}

        return FeatureCollection(sub_schema, sub_values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, feature_name: str) -> Feature:
        array = self.values[feature_name]

        if not isinstance(array, MerlinArray):
            array = build_merlin_array(array)

        return Feature(self.schema.column_schemas[feature_name], array)

    def __iter__(self):
        for col_name in self.schema.column_names:
            yield self[col_name]


def build_merlin_array(values):
    """
    Construct a MerlinArray from a numpy array, cupy array, cudf array, or tensorflow tensor.

    Parameters
    ----------
    values : Union[np.ndarray, cp.ndarray, cf.DataFrame, tf.Tensor]
        The array to be converted to a MerlinArray

    Returns
    -------
    MerlinArray
        A MerlinArray object that wraps the provided array

    Raises
    ------
    TypeError
        If the provided array is not one of the supported types
    """
    if tf is not None and isinstance(values, tf.Tensor):
        return MerlinTensorflowArray(values)
    elif cp is not None and isinstance(values, cp.ndarray):
        return MerlinCupyArray(values)
    elif np is not None and isinstance(values, np.ndarray):
        return MerlinNumpyArray(values)
    elif np is not None and isinstance(values, list):
        return MerlinNumpyArray(np.array(values))
    elif cf is not None and isinstance(values, cf.Series):
        return MerlinCudfArray(values)
    else:
        raise TypeError(f"Unknown type of array: {type(values)}")
