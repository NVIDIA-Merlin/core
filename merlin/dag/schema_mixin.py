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
from __future__ import annotations

from enum import Flag, auto

from merlin.dag.selector import ColumnSelector
from merlin.schema import ColumnSchema, Schema


class Supports(Flag):
    """Indicates what type of data representation this operator supports for transformations"""

    # cudf dataframe
    CPU_DATAFRAME = auto()
    # pandas dataframe
    GPU_DATAFRAME = auto()
    # dict of column name to numpy array
    CPU_DICT_ARRAY = auto()
    # dict of column name to cupy array
    GPU_DICT_ARRAY = auto()


class ComputeSchemaMixin:
    def compute_input_schema(
        self,
        parents_schema: Schema,
        selector: ColumnSelector = None,
    ) -> Schema:
        """Given the schemas coming from upstream sources and a column selector for the
        input columns, returns a set of schemas for the input columns this operator will use
        Parameters
        -----------
        root_schema: Schema
            Base schema of the dataset before running any operators.
        parents_schema: Schema
            The combined schemas of the upstream parents feeding into this operator
        deps_schema: Schema
            The combined schemas of the upstream dependencies feeding into this operator
        col_selector: ColumnSelector
            The column selector to apply to the input schema
        Returns
        -------
        Schema
            The schemas of the columns used by this operator
        """
        selector = selector or ColumnSelector("*")

        self._validate_matching_cols(parents_schema, selector, self.compute_input_schema.__name__)

        return parents_schema

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector = None,
    ) -> Schema:
        """Given a set of schemas and a column selector for the input columns,
        returns a set of schemas for the transformed columns this operator will produce
        Parameters
        -----------
        input_schema: Schema
            The schemas of the columns to apply this operator to
        col_selector: ColumnSelector
            The column selector to apply to the input schema
        Returns
        -------
        Schema
            The schemas of the columns produced by this operator
        """
        col_selector = col_selector or ColumnSelector("*")

        if not col_selector:
            col_selector = ColumnSelector(input_schema.column_names)

        if col_selector.tags:
            tags_col_selector = ColumnSelector(tags=col_selector.tags)
            filtered_schema = input_schema.apply(tags_col_selector)
            col_selector += ColumnSelector(filtered_schema.column_names)

            # zero tags because already filtered
            col_selector._tags = []

        self._validate_matching_cols(
            input_schema, col_selector, self.compute_output_schema.__name__
        )

        output_schema = Schema()
        for output_col_name, input_col_names in self.column_mapping(col_selector).items():
            col_schema = ColumnSchema(output_col_name)
            col_schema = self._compute_dtype(col_schema, input_schema[input_col_names])
            col_schema = self._compute_tags(col_schema, input_schema[input_col_names])
            col_schema = self._compute_properties(col_schema, input_schema[input_col_names])
            output_schema += Schema([col_schema])

        return output_schema

    def _compute_dtype(self, col_schema: Schema, input_schema: Schema):
        dtype = col_schema.dtype
        is_list = col_schema.is_list
        is_ragged = col_schema.is_ragged

        if input_schema.column_schemas:
            source_col_name = input_schema.column_names[0]
            dtype = input_schema[source_col_name].dtype
            is_list = input_schema[source_col_name].is_list
            is_ragged = input_schema[source_col_name].is_ragged

        if self.output_dtype is not None:
            dtype = self.output_dtype
            is_list = any(cs.is_list for _, cs in input_schema.column_schemas.items())
            is_ragged = any(cs.is_ragged for _, cs in input_schema.column_schemas.items())

        return col_schema.with_dtype(dtype, is_list=is_list, is_ragged=is_ragged)

    def _compute_tags(self, col_schema: Schema, input_schema: Schema):
        tags = []
        if input_schema.column_schemas:
            source_col_name = input_schema.column_names[0]
            tags = input_schema[source_col_name].tags

        # Override empty tags with tags from the input schema
        # Override input schema tags with the output tags of this operator
        return col_schema.with_tags(tags).with_tags(self.output_tags)

    def _compute_properties(self, col_schema: Schema, input_schema: Schema):
        properties = {}

        if input_schema.column_schemas:
            source_col_name = input_schema.column_names[0]
            properties.update(input_schema.column_schemas[source_col_name].properties)

        properties.update(self.output_properties)

        return col_schema.with_properties(properties)

    def column_mapping(self, col_selector: ColumnSelector):
        column_mapping = {}
        for col_name in col_selector.names:
            column_mapping[col_name] = [col_name]
        return column_mapping

    def compute_column_schema(self, col_name: str, input_schema: Schema):
        methods = [self._compute_dtype, self._compute_tags, self._compute_properties]
        return self._compute_column_schema(col_name, input_schema, methods=methods)

    def _compute_column_schema(self, col_name: str, input_schema: Schema, methods=None):
        col_schema = ColumnSchema(col_name)

        for method in methods:
            col_schema = method(col_schema, input_schema)

        return col_schema

    def _validate_matching_cols(self, schema: Schema, selector: ColumnSelector, method_name: str):
        selector = selector or ColumnSelector()
        missing_cols = [name for name in selector.names if name not in schema.column_names]
        if missing_cols:
            raise ValueError(
                f"Missing columns {missing_cols} found in operator"
                f"{self.__class__.__name__} during {method_name}."
            )

    @property
    def output_dtype(self):
        return None

    @property
    def output_tags(self):
        return []

    @property
    def output_properties(self):
        return {}

    @property
    def supports(self) -> Supports:
        """Returns what kind of data representation this operator supports"""
        return ()
