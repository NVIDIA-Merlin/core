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

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Text, Union

import numpy as np
import pandas as pd

from merlin.schema.tags import Tags, TagSet


class ColumnQuantity(Enum):
    """Describes the number of elements in each row of a column"""

    SCALAR = "scalar"
    FIXED_LIST = "fixed_list"
    RAGGED_LIST = "ragged_list"


@dataclass(frozen=True)
class Domain:
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
    tags: Optional[TagSet] = field(default_factory=TagSet)
    properties: Optional[Dict] = field(default_factory=dict)
    dtype: Optional[object] = None
    is_list: Optional[bool] = None
    is_ragged: Optional[bool] = None

    def __post_init__(self):
        """Standardize tags and dtypes on initialization

        Raises:
            TypeError: If the provided dtype cannot be cast to a numpy dtype
        """
        tags = TagSet(self.tags)
        object.__setattr__(self, "tags", tags)

        try:
            if hasattr(self.dtype, "numpy_dtype"):
                dtype = np.dtype(self.dtype.numpy_dtype)
            elif hasattr(self.dtype, "_categories"):
                dtype = self.dtype._categories.dtype
            elif isinstance(self.dtype, pd.StringDtype):
                dtype = np.dtype("O")
            else:
                dtype = np.dtype(self.dtype)
        except TypeError as err:
            raise TypeError(
                f"Unsupported dtype {self.dtype}, unable to cast {self.dtype} to a numpy dtype."
            ) from err

        object.__setattr__(self, "dtype", dtype)

        value_count = Domain(**self.properties.get("value_count", {}))

        if value_count.min == 0 or value_count.max == 0:
            raise ValueError(
                "`value_count` min and max must be greater than zero. "
                f"Provided min: {value_count.min} max: {value_count.max}"
            )

        if self.is_list is None:
            if value_count.max and value_count.max > 0:
                object.__setattr__(self, "is_list", True)
            else:
                object.__setattr__(self, "is_list", False)

        if self.is_ragged is None:
            if value_count.is_bounded and value_count.max > value_count.min:
                object.__setattr__(self, "is_ragged", True)
            elif value_count.is_bounded and value_count.max == value_count.min:
                object.__setattr__(self, "is_ragged", False)
            else:
                object.__setattr__(self, "is_ragged", self.is_list)

        if self.is_ragged and not self.is_list:
            raise ValueError(
                "`is_ragged` is set to `True` but `is_list` is not. "
                "Only list columns can set the `is_ragged` flag."
            )

        if self.is_ragged and value_count.is_bounded and value_count.min == value_count.max:
            raise ValueError(
                "`is_ragged` is set to `True` but `value_count.min` == `value_count.max`. "
                "If value_count min/max are equal. "
                "This is a fixed size list and `is_ragged` should be set to False. "
            )

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
        return ColumnSchema(
            name,
            tags=self.tags,
            properties=self.properties,
            dtype=self.dtype,
            is_list=self.is_list,
            is_ragged=self.is_ragged,
        )

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
        return ColumnSchema(
            self.name,
            tags=self.tags.override(tags),
            properties=self.properties,
            dtype=self.dtype,
            is_list=self.is_list,
            is_ragged=self.is_ragged,
        )

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
            raise TypeError("properties must be in dict format, key: value")

        # Using new dictionary to avoid passing old ref to new schema
        new_properties = {**self.properties, **properties}

        is_ragged = self.is_ragged
        value_count = Domain(**new_properties.get("value_count", {}))
        if value_count.is_bounded and value_count.max == value_count.min:
            is_ragged = False

        return ColumnSchema(
            self.name,
            tags=self.tags,
            properties=new_properties,
            dtype=self.dtype,
            is_list=self.is_list,
            is_ragged=is_ragged,
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
        is_list = is_list if is_list is not None else self.is_list

        if is_list:
            is_ragged = is_ragged if is_ragged is not None else self.is_ragged
        else:
            is_ragged = False

        return ColumnSchema(
            self.name,
            tags=self.tags,
            properties=self.properties,
            dtype=dtype,
            is_list=is_list,
            is_ragged=is_ragged,
        )

    @property
    def int_domain(self) -> Optional[Domain]:
        return self._domain() if np.issubdtype(self.dtype, np.integer) else None

    @property
    def float_domain(self) -> Optional[Domain]:
        return self._domain() if np.issubdtype(self.dtype, np.floating) else None

    @property
    def value_count(self) -> Optional[Domain]:
        value_count = self.properties.get("value_count")
        return Domain(**value_count) if value_count else None

    def __merge__(self, other):
        col_schema = self.with_tags(other.tags)
        col_schema = col_schema.with_properties(other.properties)
        col_schema = col_schema.with_dtype(
            other.dtype, is_list=other.is_list, is_ragged=other.is_ragged
        )
        col_schema = col_schema.with_name(other.name)
        return col_schema

    def __str__(self) -> str:
        return self.name

    def _domain(self) -> Optional[Domain]:
        """ """
        domain = self.properties.get("domain")
        return Domain(**domain) if domain else None


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
