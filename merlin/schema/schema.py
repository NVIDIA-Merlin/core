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

from dataclasses import InitVar, dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional, Text, Tuple, Union

import pandas as pd

import merlin.dtypes as md
from merlin.dtypes import DType
from merlin.dtypes.shape import Shape
from merlin.schema.tags import Tags, TagSet


class ColumnQuantity(Enum):
    """Describes the number of elements in each row of a column"""

    SCALAR = "scalar"
    FIXED_LIST = "fixed_list"
    RAGGED_LIST = "ragged_list"


@dataclass(frozen=True)
class Domain:
    """Describes an integer or float domain.

    Can be partially specified. With any of name, min, max.
    """

    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    name: Optional[str] = None

    @property
    def is_bounded(self):
        """Returns True of domain has both a lower and upper bound.

        Returns
        -------
        bool
            True if domain has both min and max defined.
        """
        return self.max and self.min


@dataclass(frozen=True)
class ColumnSchema:
    """A schema containing metadata of a dataframe column."""

    name: Text
    tags: Optional[Union[TagSet, List[Union[str, Tags]]]] = field(default_factory=TagSet)
    properties: Optional[Dict] = field(default_factory=dict)
    dtype: Optional[DType] = None
    is_list: Optional[bool] = None
    is_ragged: Optional[bool] = None
    dims: InitVar[Union[Tuple, Shape]] = None

    def __post_init__(self, dims):
        """Standardize tags and dtypes on initialization

        This method works around the inability to set attributes on frozen dataclass
        objects by using object.__setattr__, which bypasses the methods that frozen
        dataclasses lock down. That approach allows to do some normalization on the
        object's attribute values in the post init hook that we otherwise wouldn't
        have a way to implement.

        Raises:
            TypeError: If the provided dtype cannot be cast to a numpy dtype
            ValueError: If the provided shape, value counts, and/or flags are inconsistent
        """
        # Provide defaults and minor conversions for convenience
        object.__setattr__(self, "tags", TagSet(self.tags))

        dtype = md.dtype(self.dtype or md.unknown).without_shape
        object.__setattr__(self, "dtype", dtype)

        # Validate that everything provided is consistent
        value_counts = self.properties.get("value_count", {})
        if self.is_list and not self.is_ragged:
            if "max" in value_counts and "min" not in value_counts:
                value_counts["min"] = value_counts["max"]
            if "max" not in value_counts and "min" in value_counts:
                value_counts["max"] = value_counts["min"]

        self._validate_shape_info(self.shape, value_counts, self.is_list, self.is_ragged)

        # Pick which source to pull shape info from
        if dims:
            new_shape = Shape(dims)
        elif dtype.shape.dims:
            new_shape = dtype.shape
        elif value_counts:
            new_shape = self._shape_from_counts(Domain(**value_counts))
        elif self.is_list:
            new_shape = self._shape_from_flags(self.is_list)
        else:
            new_shape = Shape()

        # Update the shape and propagate out to flags and value counts
        dtype = dtype.with_shape(new_shape)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "is_list", dtype.shape.is_list)
        object.__setattr__(self, "is_ragged", dtype.shape.is_ragged)

        if new_shape.dims is not None and len(new_shape.dims) > 1:
            value_counts = {"min": new_shape.dims[1].min, "max": new_shape.dims[1].max}
            properties = {**self.properties, **{"value_count": value_counts}}
            object.__setattr__(self, "properties", properties)

    def _shape_from_flags(self, is_list):
        return Shape(((0, None), (0, None))) if is_list else None

    def _shape_from_counts(self, value_count):
        return Shape(((0, None), (value_count.min or 0, value_count.max)))

    @property
    def shape(self):
        return self.dtype.shape

    @property
    def quantity(self):
        """
        Describes the number of elements in each row of this column

        Returns
        -------
        ColumnQuantity
            SCALAR when one element per row
            FIXED_LIST when the same number of elements per row
            RAGGED_LIST when different numbers of elements per row
        """
        if self.is_list and self.is_ragged:
            return ColumnQuantity.RAGGED_LIST
        elif self.is_list:
            return ColumnQuantity.FIXED_LIST
        else:
            return ColumnQuantity.SCALAR

    def with_name(self, name: str) -> "ColumnSchema":
        """Create a copy of this ColumnSchema object with a different column name

        Parameters
        ----------
        name : str
            New column name

        Returns
        -------
        ColumnSchema
            Copied object with new column name

        """
        return replace(self, name=name)

    def with_tags(self, tags: Union[str, Tags]) -> "ColumnSchema":
        """Create a copy of this ColumnSchema object with different column tags

        Parameters
        ----------
        tags : Union[str, Tags]
            New column tags

        Returns
        -------
        ColumnSchema
            Copied object with new column tags

        """
        return replace(self, tags=self.tags.override(tags))  # type: ignore

    def with_properties(self, properties: dict) -> "ColumnSchema":
        """Create a copy of this ColumnSchema object with different column properties

        Parameters
        ----------
        properties : dict
            New column properties

        Returns
        -------
        ColumnSchema
            Copied object with new column properties

        Raises
        ------
        TypeError
            If properties are not a dict

        """
        if not isinstance(properties, dict):
            raise TypeError("ColumnSchema properties must be a dictionary")

        # Using new dictionary to avoid passing old ref to new schema
        new_properties = {**self.properties, **properties}

        value_counts = properties.get("value_count", {})

        if value_counts:
            return replace(
                self,
                properties=new_properties,
                dtype=self.dtype.without_shape,
                is_list=None,
                is_ragged=None,
            )
        else:
            return replace(
                self,
                properties=new_properties,
            )

    def with_dtype(self, dtype, is_list: bool = None, is_ragged: bool = None) -> "ColumnSchema":
        """Create a copy of this ColumnSchema object with different column dtype

        Parameters
        ----------
        dtype : np.dtype
            New column dtype
        is_list: bool :
            Whether rows in this column contain lists.
             (Default value = None)
        is_ragged: bool :
            Whether lists in this column have varying lengths.
             (Default value = None)

        Returns
        -------
        ColumnSchema
            Copied object with new column dtype

        """
        new_dtype = md.dtype(dtype).with_shape(self.shape)

        properties = self.properties.copy()
        if is_list is not None or is_ragged is not None:
            properties.pop("value_count", None)
            new_dtype = new_dtype.without_shape

        return replace(
            self, dtype=new_dtype, properties=properties, is_list=is_list, is_ragged=is_ragged
        )

    def with_shape(self, shape: Union[Tuple, Shape]) -> "ColumnSchema":
        """
        Create a copy of this object with a new shape

        Parameters
        ----------
        shape : Union[Tuple, Shape]
            Object to set as shape, must be either a tuple or Shape.

        Returns
        -------
        ColumnSchema
            A copy of this object containing the provided shape value

        Raises
        ------
        TypeError
            If value is not either a tuple or a Shape
        """
        dims = Shape(shape).as_tuple
        properties = self.properties.copy()
        properties.pop("value_count", None)
        return replace(
            self,
            dims=dims,
            properties=properties,
            is_list=None,
            is_ragged=None,
        )

    @property
    def int_domain(self) -> Optional[Domain]:
        return self._domain() if self.dtype.is_integer else None

    @property
    def float_domain(self) -> Optional[Domain]:
        return self._domain() if self.dtype.is_float else None

    @property
    def value_count(self) -> Optional[Domain]:
        value_count = self.properties.get("value_count")
        return Domain(**value_count) if value_count else None

    def __merge__(self, other):
        col_schema = (
            self.with_name(other.name)
            .with_dtype(other.dtype)
            .with_tags(other.tags)
            .with_properties(other.properties)
            .with_shape(other.shape)
        )
        return col_schema

    def __str__(self) -> str:
        return self.name

    def _domain(self) -> Optional[Domain]:
        """ """
        domain = self.properties.get("domain")
        return Domain(**domain) if domain else None

    def _validate_shape_info(self, shape, value_counts, is_list, is_ragged):
        value_counts = value_counts or {}

        min_count = value_counts.get("min", None)
        max_count = value_counts.get("max", None)
        ragged_counts = min_count != max_count

        if shape and shape.dims is not None:
            if is_ragged is not None and shape.is_ragged != is_ragged:
                raise ValueError(
                    f"Provided value of `is_ragged={is_ragged}` "
                    f"is inconsistent with shape `{shape}`."
                )
            elif is_list is not None and shape.is_list != is_list:
                raise ValueError(
                    f"Provided value of `is_list={is_list}` "
                    f"is inconsistent with shape `{shape}`."
                )

        if value_counts and shape and shape.dims is not None:
            if (min_count and min_count != shape.dims[1].min) or (
                max_count and max_count != shape.dims[1].max
            ):
                raise ValueError(
                    f"Provided value counts `{value_counts}` "
                    f"are inconsistent with shape `{shape}`."
                )

        if is_list is False and is_ragged is True:
            raise ValueError(
                "Columns with `is_list=False` can't set `is_ragged=True`, "
                "since non-list columns can't be ragged."
            )

        if value_counts and is_ragged is not None and is_ragged != ragged_counts:
            raise ValueError(
                f"Provided value of `is_ragged={is_ragged}` "
                f"is inconsistent with value counts `{value_counts}`."
            )

        # TODO: Enable this validation once we've removed these cases
        #       from downstream Merlin libraries
        # if (
        #     not value_counts
        #     and not (shape and shape.dims)
        #     and is_list is True
        #     and is_ragged is False
        # ):
        #     raise ValueError(
        #         "Can't determine a shape for this column from "
        #         "`is_list=True` and `is_ragged=False` without value counts. "
        #     )


class Schema:
    """A collection of column schemas for a dataset."""

    def __init__(self, column_schemas=None):
        column_schemas = column_schemas or {}

        if isinstance(column_schemas, dict):
            self.column_schemas = column_schemas
        elif isinstance(column_schemas, (list, tuple)):
            self.column_schemas = {}
            for column_schema in column_schemas:
                if isinstance(column_schema, str):
                    column_schema = ColumnSchema(column_schema)
                self.column_schemas[column_schema.name] = column_schema
        else:
            raise TypeError("The `column_schemas` parameter must be a list or dict.")

    @property
    def column_names(self):
        return list(self.column_schemas.keys())

    def select(self, selector) -> "Schema":
        """Select matching columns from this Schema object using a ColumnSelector

        Parameters
        ----------
        selector : ColumnSelector
            Selector that describes which columns match

        Returns
        -------
        Schema
            New object containing only the ColumnSchemas of selected columns

        """
        if selector is not None:
            if selector.all:
                return self

            schema = Schema()
            if selector.names:
                schema += self.select_by_name(selector.names)
            if selector.tags:
                schema += self.select_by_tag(selector.tags)
            return schema
        return self

    def apply(self, selector) -> "Schema":
        return self.select(selector)

    def excluding(self, selector) -> "Schema":
        """Select non-matching columns from this Schema object using a ColumnSelector

        Parameters
        ----------
        selector : ColumnSelector
            Selector that describes which columns match

        Returns
        -------
        Schema
            New object containing only the ColumnSchemas of selected columns

        """
        schema = self
        if selector is not None:
            if selector.all:
                return Schema()
            if selector.names:
                schema = schema.excluding_by_name(selector.names)
            if selector.tags:
                schema = schema.excluding_by_tag(selector.tags)

        return schema

    def apply_inverse(self, selector) -> "Schema":
        return self.excluding(selector)

    def select_by_tag(self, tags: Union[Union[str, Tags], List[Union[str, Tags]]]) -> "Schema":
        """Select matching columns from this Schema object using a list of tags

        Parameters
        ----------
        tags : List[Union[str, Tags]] :
            List of tags that describes which columns match

        Returns
        -------
        Schema
            New object containing only the ColumnSchemas of selected columns

        """
        if not isinstance(tags, (list, tuple)):
            tags = [tags]

        selected_schemas = {}

        for _, column_schema in self.column_schemas.items():
            if any(x in column_schema.tags for x in tags):
                selected_schemas[column_schema.name] = column_schema

        return Schema(selected_schemas)

    def excluding_by_tag(self, tags) -> "Schema":
        if not isinstance(tags, (list, tuple)):
            tags = [tags]

        selected_schemas = {}

        for column_schema in self.column_schemas.values():
            if not any(x in column_schema.tags for x in tags):
                selected_schemas[column_schema.name] = column_schema

        return Schema(selected_schemas)

    def remove_by_tag(self, tags) -> "Schema":
        return self.excluding_by_tag(tags)

    def select_by_name(self, names: List[str]) -> "Schema":
        """Select matching columns from this Schema object using a list of column names

        Parameters
        ----------
        names: List[str] :
            List of column names that describes which columns match

        Returns
        -------
        Schema
            New object containing only the ColumnSchemas of selected columns

        """
        if isinstance(names, str):
            names = [names]

        selected_schemas = {
            key: self.column_schemas[key] for key in names if self.column_schemas.get(key, None)
        }
        return Schema(selected_schemas)

    def excluding_by_name(self, col_names: List[str]):
        """Remove columns from this Schema object by name

        Parameters
        ----------
        col_names : List[str]
            Names of the column to remove

        Returns
        -------
        Schema
            New Schema object after the columns are removed

        """
        return Schema(
            [
                col_schema
                for col_name, col_schema in self.column_schemas.items()
                if col_name not in col_names
            ]
        )

    def remove_col(self, col_name: str) -> "Schema":
        """Remove a column from this Schema object by name

        Parameters
        ----------
        col_name : str
            Name of the column to remove

        Returns
        -------
        Schema
            This Schema object after the column is removed

        """
        return self.excluding_by_name([col_name])

    def without(self, col_names: List[str]) -> "Schema":
        return self.excluding_by_name(col_names)

    def get(self, col_name: str, default: ColumnSchema = None) -> ColumnSchema:
        """Get a ColumnSchema by name

        Parameters
        ----------
        col_name : str
            Name of the column to get
        default: ColumnSchema :
            Default value to return if column is not found.
             (Default value = None)

        Returns
        -------
        ColumnSchema
            Retrieved column schema (or default value, if not found)

        """
        return self.column_schemas.get(col_name, default)

    @property
    def first(self) -> ColumnSchema:
        """
        Returns the first ColumnSchema in the Schema. Useful for cases where you select down
        to a single column via select_by_name or select_by_tag, and just want the value

        Returns
        -------
        ColumnSchema
            The first column schema present in this Schema object

        Raises
        ------
        ValueError
            If this Schema object contains no column schemas
        """
        if not self.column_schemas:
            raise ValueError("There are no columns in this schema to call .first on")

        return next(iter(self.column_schemas.values()))

    def __getitem__(self, column_name):
        if isinstance(column_name, str):
            return self.column_schemas[column_name]
        elif isinstance(column_name, (list, tuple)):
            return Schema([self.column_schemas[col_name] for col_name in column_name])

    def __setitem__(self, column_name, column_schema):
        self.column_schemas[column_name] = column_schema

    def __iter__(self):
        return iter(self.column_schemas.values())

    def __len__(self):
        return len(self.column_schemas)

    def __repr__(self):
        return str([col_schema.__dict__ for col_schema in self.column_schemas.values()])

    def _repr_html_(self):
        # Repr for Jupyter Notebook
        return self.to_pandas()._repr_html_()

    def to_pandas(self) -> pd.DataFrame:
        """Convert this Schema object to a pandas DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame containing the column schemas in this Schema object

        """
        props = [c.__dict__ for c in self.column_schemas.values()]

        return pd.json_normalize(props)

    def __eq__(self, other):
        if not isinstance(other, Schema) or len(self.column_schemas) != len(other.column_schemas):
            return False
        return self.column_schemas == other.column_schemas

    def __add__(self, other):
        if other is None:
            return self
        if not isinstance(other, Schema):
            raise TypeError(f"unsupported operand type(s) for +: 'Schema' and {type(other)}")

        col_schemas = []

        # must account for same columns in both schemas,
        # use the one with more information for each field
        keys_self_not_other = [
            col_name for col_name in self.column_names if col_name not in other.column_names
        ]

        for key in keys_self_not_other:
            col_schemas.append(self.column_schemas[key])

        for col_name, other_schema in other.column_schemas.items():
            if col_name in self.column_schemas:
                # check which one
                self_schema = self.column_schemas[col_name]
                col_schemas.append(self_schema.__merge__(other_schema))
            else:
                col_schemas.append(other_schema)

        return Schema(col_schemas)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if other is None:
            return self

        if not isinstance(other, Schema):
            raise TypeError(f"unsupported operand type(s) for -: 'Schema' and {type(other)}")

        result = Schema({**self.column_schemas})

        for key in other.column_schemas.keys():
            if key in self.column_schemas.keys():
                result.column_schemas.pop(key, None)

        return result
