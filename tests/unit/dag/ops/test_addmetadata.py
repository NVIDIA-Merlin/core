import numpy as np
import pytest

import merlin.dag.ops.add_metadata as ops
from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema


@pytest.mark.parametrize("properties", [{}, {"p1": "1"}])
@pytest.mark.parametrize("tags", [[], ["TAG1", "TAG2"]])
@pytest.mark.parametrize(
    "op",
    [
        ops.AddMetadata(tags=["excellent"], properties={"domain": {"min": 0, "max": 20}}),
        ops.AddTags(tags=["excellent"]),
        ops.AddProperties(properties={"domain": {"min": 0, "max": 20}}),
        ops.TagAsUserID(),
        ops.TagAsItemID(),
        ops.TagAsUserFeatures(),
        ops.TagAsItemFeatures(),
    ],
)
@pytest.mark.parametrize("selection", [["1"], ["2", "3"], ["1", "2", "3", "4"]])
def test_schema_out(tags, properties, selection, op):
    # Create columnSchemas
    column_schemas = []
    all_cols = []
    for x in range(5):
        all_cols.append(str(x))
        column_schemas.append(
            ColumnSchema(str(x), dtype=np.int32, tags=tags, properties=properties)
        )

    # Turn to Schema
    input_schema = Schema(column_schemas)

    # run schema through op
    selector = ColumnSelector(selection)
    output_schema = op.compute_output_schema(input_schema, selector)

    # should have dtype float
    for input_col_name in selector.names:
        output_col_names = [name for name in output_schema.column_schemas if input_col_name in name]
        if output_col_names:
            for output_col_name in output_col_names:
                result_schema = output_schema.column_schemas[output_col_name]

                expected_dtype = op._compute_dtype(
                    ColumnSchema(output_col_name),
                    Schema([input_schema.column_schemas[input_col_name]]),
                ).dtype

                expected_tags = op._compute_tags(
                    ColumnSchema(output_col_name),
                    Schema([input_schema.column_schemas[input_col_name]]),
                ).tags

                expected_properties = op._compute_properties(
                    ColumnSchema(output_col_name),
                    Schema([input_schema.column_schemas[input_col_name]]),
                ).properties

                assert result_schema.dtype == expected_dtype
                if output_col_name in selector.names:
                    assert result_schema.properties == expected_properties

                    assert len(result_schema.tags) == len(expected_tags)
                else:
                    assert set(expected_tags).issubset(result_schema.tags)

    not_used = [col for col in all_cols if col not in selector.names]
    for input_col_name in not_used:
        assert input_col_name not in output_schema.column_schemas
