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

from typing import Any, List, Union

import merlin.dag
from merlin.dag.schema_mixin import ComputeSchemaMixin
from merlin.dag.selector import ColumnSelector
from merlin.schema import Schema


class BaseOperator(ComputeSchemaMixin):
    """
    Base class for all operator classes.
    """

    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: ColumnSelector,
        dependencies_selector: ColumnSelector,
    ) -> ColumnSelector:
        """
        Compute the input selector based on the parents' and dependencies' selectors

        Parameters
        ----------
        input_schema : Schema
            Schema of the full set of input columns
        selector : ColumnSelector
            Selector from this operator's graph node
        parents_selector : ColumnSelector
            Combined selector from all of this operator's graph node's parents
        dependencies_selector : ColumnSelector
            Combined selector from all of this operator's graph node's upstream dependencies

        Returns
        -------
        ColumnSelector
            The computed selector for this operator's graph node
        """
        self._validate_matching_cols(input_schema, selector, self.compute_selector.__name__)

        return selector

    # TODO: Update instructions for how to define custom
    # operators to reflect constructing the column mapping
    # (They should no longer override this method)
    def output_column_names(self, col_selector: ColumnSelector) -> ColumnSelector:
        """Given a set of columns names returns the names of the transformed columns this
        operator will produce
        Parameters
        -----------
        columns: list of str, or list of list of str
            The columns to apply this operator to
        Returns
        -------
        list of str, or list of list of str
            The names of columns produced by this operator
        """
        return ColumnSelector(list(self.column_mapping(col_selector).keys()))

    @property
    def dependencies(self) -> List[Union[str, Any]]:
        """Defines an optional list of column dependencies for this operator. This lets you consume columns
        that aren't part of the main transformation workflow.
        Returns
        -------
        str, list of str or ColumnSelector, optional
            Extra dependencies of this operator. Defaults to None
        """
        return []

    def __rrshift__(self, other):
        return ColumnSelector(other) >> self

    @property
    def label(self) -> str:
        return self.__class__.__name__

    def create_node(self, selector):
        return merlin.dag.Node(selector)

    def _get_columns(self, df, selector):
        if isinstance(df, dict):
            return {col_name: df[col_name] for col_name in selector.names}
        else:
            return df[selector.names]
