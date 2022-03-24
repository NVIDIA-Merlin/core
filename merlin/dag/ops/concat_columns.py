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

from merlin.core.dispatch import DataFrameType
from merlin.dag.base_operator import BaseOperator
from merlin.dag.selector import ColumnSelector
from merlin.schema import Schema


class ConcatColumns(BaseOperator):
    """
    This operator class provides an implementation for the `+` operator used in constructing graphs.
    """

    def __init__(self, label=None):
        self._label = label or self.__class__.__name__
        super().__init__()

    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: ColumnSelector,
        dependencies_selector: ColumnSelector,
    ) -> ColumnSelector:
        """
        Combine selectors from the nodes being added

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
            Combined column selectors of parent and dependency nodes
        """
        self._validate_matching_cols(
            input_schema,
            parents_selector + dependencies_selector,
            self.compute_selector.__name__,
        )

        return parents_selector + dependencies_selector

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        """
        Combine schemas from the nodes being added

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
            Combined schema of columns from parents and dependencies
        """
        return parents_schema + deps_schema

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        """Simply returns the selected output columns from the input dataframe

        The main functionality of this operator has to do with computing the schemas
        for `+` nodes in the Workflow graph, so very little has to happen in the
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
        return super()._get_columns(df, col_selector)

    @property
    def label(self) -> str:
        """
        Display name of this operator
        """
        return self._label
