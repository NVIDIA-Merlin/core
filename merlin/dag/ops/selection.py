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

import logging

from merlin.core.dispatch import DataFrameType
from merlin.dag.base_operator import BaseOperator
from merlin.dag.selector import ColumnSelector
from merlin.schema import Schema

LOG = logging.getLogger("SelectionOp")


class SelectionOp(BaseOperator):
    """
    This operator class provides an implementation of the behavior of selection (e.g. input) nodes.
    """

    def __init__(self, selector=None):
        self.selector = selector
        super().__init__()

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        """Simply returns the selected output columns from the input dataframe

        The main functionality of this operator has to do with computing the schemas
        for selection nodes in the Workflow graph, so very little has to happen in the
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
        selector = col_selector or self.selector
        return super()._get_columns(df, selector)

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        """
        Return the schemas of columns

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
            Schema of selected columns from input, parents, and dependencies
        """
        upstream_schema = root_schema + parents_schema + deps_schema
        return upstream_schema.apply(self.selector)

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
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
        selector = col_selector or self.selector
        return super().compute_output_schema(input_schema, selector, prev_output_schema)
