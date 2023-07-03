#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
from typing import Optional

from merlin.core.protocols import Transformable
from merlin.dag import BaseOperator, DataFormats
from merlin.dag.selector import ColumnSelector
from merlin.schema import Schema
from merlin.telemetry import get_telemetry_provider


class Logging(BaseOperator):
    def __init__(self, columns=None):
        self.selector = ColumnSelector(columns) if columns else ColumnSelector("*")

    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: Optional[ColumnSelector] = None,
        dependencies_selector: Optional[ColumnSelector] = None,
    ) -> ColumnSelector:
        # Since we might have a wildcard selector, we need to resolve
        # abstract selectors to concrete column names here
        self.selector = self.selector.resolve(input_schema)

        return super().compute_selector(
            input_schema, selector, parents_selector, dependencies_selector
        )

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        # We want to log a subset of the columns provided, but we don't
        # want to change which columns are returned, so we ignore the
        # selector provided as arg, use our own selector, and return the
        # original data unmodified
        loggable = transformable[self.selector.names]
        provider = get_telemetry_provider()
        provider.log(loggable.to_dict(orient="list"))
        return transformable

    @property
    def supported_formats(self) -> DataFormats:
        # Since we'd like to have string support for logging, this op
        # only supports dataframes so that the executors don't try to
        # convert any string-containing input to `TensorTable`
        # (which intentionally doesn't have string support)
        return DataFormats.PANDAS_DATAFRAME | DataFormats.CUDF_DATAFRAME
