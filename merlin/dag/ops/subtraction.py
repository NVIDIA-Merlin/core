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

from merlin.core.dispatch import DataFrameType
from merlin.dag.base_operator import BaseOperator
from merlin.dag.selector import ColumnSelector
from merlin.schema import Schema


class SubtractionOp(BaseOperator):
    """
    This operator class provides an implementation for the `-` operator used in constructing graphs.
    """

    def __init__(self, selector=None):
        self.selector = selector
        super().__init__()

    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: ColumnSelector,
        dependencies_selector: ColumnSelector,
    ) -> ColumnSelector:
        """
        Creates selector of all columns from the input schema

        Parameters
        ----------
        input_schema : Schema
            Combined schema of the columns coming from upstream nodes
        selector : ColumnSelector
            Existing column selector for this node in the graph (often None)
        parents_selector : ColumnSelector
            Combined column selectors of parent nodes
        dependencies_selector : ColumnSelector
            Combined column selectors of dependency nodes

        Returns
        -------
        ColumnSelector
            Selector of all columns from the input schema
        """
        return ColumnSelector(input_schema.column_names)

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        """
        Return remaining schemas of columns after removing dependencies

        Parameters
        ----------
        root_schema : Schema
            Schema of the columns from the input dataset
        parents_schema : Schema
            Schema of the columns from the parent nodes
        deps_schema : Schema
            Schema of the columns from the dependency nodes
        selector : ColumnSelector
            Existing column selector for this node in the graph (often None)

        Returns
        -------
        Schema
            Remaining schema of columns from parents after removing dependencies
        """
        result = None
        if deps_schema.column_schemas:
            result = parents_schema - deps_schema
        else:
            subtraction_selector = self.selector or selector
            result = parents_schema.apply_inverse(subtraction_selector)
        return result

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        """Simply returns the selected output columns from the input dataframe

        The main functionality of this operator has to do with computing the schemas
        for `-` nodes in the Workflow graph, so very little has to happen in the
        `transform` method.

        Parameters
        -----------
        columns: list of str or list of list of str
            The columns to apply this operator to
        df: Dataframe
            A pandas or cudf dataframe that this operator will work on

        Returns
        -------
        DataFrame
            Returns a transformed dataframe for this operator
        """
        selector = self.selector or col_selector
        return super()._get_columns(df, selector)
