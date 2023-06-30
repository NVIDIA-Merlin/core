from typing import Optional

from merlin.dag.ops.selection import SelectionOp
from merlin.dag.selector import ColumnSelector
from merlin.schema import Schema


class GroupingOp(SelectionOp):
    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: Optional[ColumnSelector] = None,
        dependencies_selector: Optional[ColumnSelector] = None,
    ) -> ColumnSelector:
        upstream_selector = parents_selector + dependencies_selector
        new_selector = ColumnSelector(subgroups=upstream_selector)
        selector = super().compute_selector(
            input_schema,
            new_selector,
        )
        return selector
