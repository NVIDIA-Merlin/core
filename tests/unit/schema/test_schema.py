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
import dataclasses

import pytest

from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema, Tags


def test_select_by_name():
    col1_schema = ColumnSchema("col1", tags=["a", "b", "c"])
    col2_schema = ColumnSchema("col2", tags=["b", "c", "d"])

    schema = Schema([col1_schema, col2_schema])

    col1_selection = schema.select_by_name("col1")
    col2_selection = schema.select_by_name("col2")

    assert col1_selection == Schema([col1_schema])
    assert col2_selection == Schema([col2_schema])

    select_multi = schema.select_by_name(["col1", "col2"])
    select_missing = schema.select_by_name("col3")

    assert select_multi == Schema([col1_schema, col2_schema])
    assert select_missing == Schema([])


def test_select_by_tag():
    col1_schema = ColumnSchema("col1", tags=["a", "b", "c"])
    col2_schema = ColumnSchema("col2", tags=["b", "c", "d"])

    schema = Schema([col1_schema, col2_schema])

    col1_selection = schema.select_by_tag("a")
    col2_selection = schema.select_by_tag("d")

    assert col1_selection == Schema([col1_schema])
    assert col2_selection == Schema([col2_schema])

    select_both = schema.select_by_tag("c")
    select_multi = schema.select_by_tag(["b", "c"])
    select_neither = schema.select_by_tag("e")

    assert select_both == Schema([col1_schema, col2_schema])
    assert select_multi == Schema([col1_schema, col2_schema])
    assert select_neither == Schema([])


def test_select_by_tag_string():
    col1_schema = ColumnSchema("col1", tags=[Tags.CATEGORICAL, Tags.ITEM])
    col2_schema = ColumnSchema("col2", tags=[Tags.ITEM_ID])

    schema = Schema([col1_schema, col2_schema])

    col1_selection = schema.select_by_tag("categorical")
    col2_selection = schema.select_by_tag("item_id")

    assert col1_selection == Schema([col1_schema])
    assert col2_selection == Schema([col2_schema])


@pytest.mark.parametrize(
    "item_id_col_tags",
    [[Tags.ITEM, Tags.ID], [Tags.ITEM_ID], ["item_id"], ["ITEM_ID"], ["ID", "item"]],
)
@pytest.mark.parametrize(
    "select_by_tags",
    [[Tags.ITEM, Tags.ID], [Tags.ITEM_ID], ["item_id"], ["ITEM_ID"], ["ITEM", "id"]],
)
def test_select_by_compound_tag(item_id_col_tags, select_by_tags):
    item_id_col_schema = ColumnSchema("item_id", tags=item_id_col_tags)
    other_col_schema = ColumnSchema("feature")
    schema = Schema([item_id_col_schema, other_col_schema])

    selection = schema.select_by_tag(select_by_tags, all)
    assert selection == Schema([item_id_col_schema])


@pytest.mark.parametrize(
    "item_id_col_tags",
    [[Tags.ITEM, Tags.ID], [Tags.ITEM_ID], ["item_id"], ["ITEM_ID"], ["ID", "item"]],
)
@pytest.mark.parametrize(
    "remove_by_tags",
    [[Tags.ITEM, Tags.ID], [Tags.ITEM_ID], ["item_id"], ["ITEM_ID"], ["ITEM", "id"]],
)
def test_remove_by_compound_tag(item_id_col_tags, remove_by_tags):
    item_id_col_schema = ColumnSchema("item_id", tags=item_id_col_tags)
    other_col_schema = ColumnSchema("feature")
    schema = Schema([item_id_col_schema, other_col_schema])

    selection = schema.remove_by_tag(remove_by_tags, all)
    assert selection == Schema([other_col_schema])


def test_select_by_any_tags():
    col1_schema = ColumnSchema("col1", tags=["a", "b", "c"])
    col2_schema = ColumnSchema("col2", tags=["b", "c", "d"])
    col3_schema = ColumnSchema("col3", tags=["b", "e", "f"])

    schema = Schema([col1_schema, col2_schema, col3_schema])

    col1_selection = schema.select_by_tag("a", any)
    col2_selection = schema.select_by_tag("d", any)

    assert col1_selection == Schema([col1_schema])
    assert col2_selection == Schema([col2_schema])

    select_both = schema.select_by_tag("c", any)
    select_multi = schema.select_by_tag(["b", "c"], any)
    select_neither = schema.select_by_tag("unknown", any)

    assert select_both == Schema([col1_schema, col2_schema])
    assert select_multi == Schema([col1_schema, col2_schema, col3_schema])
    assert select_neither == Schema([])


def test_select_by_all_tags():
    col1_schema = ColumnSchema("col1", tags=["a", "b", "c"])
    col2_schema = ColumnSchema("col2", tags=["b", "c", "d"])
    col3_schema = ColumnSchema("col3", tags=["b", "e", "f"])

    schema = Schema([col1_schema, col2_schema, col3_schema])

    select_multi_a = schema.select_by_tag(["a", "b"], all)
    select_multi_b = schema.select_by_tag(["c", "d"], all)
    select_multi_c = schema.select_by_tag(["a", "e"], all)

    assert select_multi_a == Schema([col1_schema])
    assert select_multi_b == Schema([col2_schema])
    assert select_multi_c == Schema([])


def test_select():
    col1_schema = ColumnSchema("col1", tags=["a", "b", "c"])
    col2_schema = ColumnSchema("col2", tags=["b", "c", "d"])
    schema = Schema([col1_schema, col2_schema])

    col1_selection = schema.select(ColumnSelector(["col1"]))
    col2_selection = schema.select(ColumnSelector(["col2"]))

    assert col1_selection == Schema([col1_schema])
    assert col2_selection == Schema([col2_schema])

    col1_selection = schema.select(ColumnSelector(tags=["a"]))
    col2_selection = schema.select(ColumnSelector(tags=["d"]))

    assert col1_selection == Schema([col1_schema])
    assert col2_selection == Schema([col2_schema])


def test_select_all():
    col1_schema = ColumnSchema("col1", tags=["a", "b", "c"])
    col2_schema = ColumnSchema("col2", tags=["b", "c", "d"])
    schema = Schema([col1_schema, col2_schema])

    selector = ColumnSelector("*")
    selection = schema.select(selector)
    assert selection == schema


def test_excluding_by_name():
    col1_schema = ColumnSchema("col1", tags=["a", "b", "c"])
    col2_schema = ColumnSchema("col2", tags=["b", "c", "d"])
    schema = Schema([col1_schema, col2_schema])

    col1_exclusion = schema.excluding_by_name(["col1"])
    col2_exclusion = schema.excluding_by_name(["col2"])

    assert col1_exclusion == Schema([col2_schema])
    assert col2_exclusion == Schema([col1_schema])

    multi_exclusion = schema.excluding_by_name(["col1", "col2"])
    missing_exclusion = schema.excluding_by_name(["col3"])

    assert multi_exclusion == Schema([])
    assert missing_exclusion == Schema([col1_schema, col2_schema])


def test_excluding_by_tag():
    col1_schema = ColumnSchema("col1", tags=["a", "b", "c"])
    col2_schema = ColumnSchema("col2", tags=["b", "c", "d"])

    schema = Schema([col1_schema, col2_schema])

    col1_exclusion = schema.excluding_by_tag("d")
    col2_exclusion = schema.excluding_by_tag("a")

    assert col1_exclusion == Schema([col1_schema])
    assert col2_exclusion == Schema([col2_schema])

    excluding_both = schema.excluding_by_tag("c")
    excluding_multi = schema.excluding_by_tag(["b", "c"])
    excluding_neither = schema.excluding_by_tag("e")

    assert excluding_both == Schema([])
    assert excluding_multi == Schema([])
    assert excluding_neither == Schema([col1_schema, col2_schema])


def test_excluding():
    col1_schema = ColumnSchema("col1", tags=["a", "b", "c"])
    col2_schema = ColumnSchema("col2", tags=["b", "c", "d"])
    schema = Schema([col1_schema, col2_schema])

    col1_exclusion = schema.excluding(ColumnSelector(["col1"]))
    col2_exclusion = schema.excluding(ColumnSelector(["col2"]))

    assert col1_exclusion == Schema([col2_schema])
    assert col2_exclusion == Schema([col1_schema])

    col1_exclusion = schema.excluding(ColumnSelector(tags=["d"]))
    col2_exclusion = schema.excluding(ColumnSelector(tags=["a"]))

    assert col1_exclusion == Schema([col1_schema])
    assert col2_exclusion == Schema([col2_schema])


def test_schema_can_be_added_to_none():
    schema_set = Schema(["a", "b", "c"])

    assert (schema_set + None) == schema_set
    assert (None + schema_set) == schema_set


def test_schema_to_pandas():
    import pandas as pd

    schema_set = Schema(["a", "b", "c"])
    df = schema_set.to_pandas()

    expected_columns = [field.name for field in dataclasses.fields(ColumnSchema)]
    expected_columns.remove("properties")

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == expected_columns


def test_construct_schema_with_column_names():
    schema = Schema(["x", "y", "z"])
    expected = Schema([ColumnSchema("x"), ColumnSchema("y"), ColumnSchema("z")])

    assert schema == expected


def test_dataset_schema_column_names():
    ds_schema = Schema(["x", "y", "z"])

    assert ds_schema.column_names == ["x", "y", "z"]


def test_dataset_schema_constructor():
    schema1 = ColumnSchema("col1", tags=["a", "b", "c"])
    schema2 = ColumnSchema("col2", tags=["c", "d", "e"])

    expected = {schema1.name: schema1, schema2.name: schema2}

    ds_schema_dict = Schema(expected)
    ds_schema_list = Schema([schema1, schema2])

    assert ds_schema_dict.column_schemas == expected
    assert ds_schema_list.column_schemas == expected

    with pytest.raises(TypeError) as exception_info:
        Schema(12345)

    assert "column_schemas" in str(exception_info.value)


def test_dataset_schemas_can_be_added():
    ds1_schema = Schema([ColumnSchema("col1"), ColumnSchema("col2")])
    ds2_schema = Schema([ColumnSchema("col3"), ColumnSchema("col4")])

    result = ds1_schema + ds2_schema

    expected = Schema(
        [
            ColumnSchema("col1"),
            ColumnSchema("col2"),
            ColumnSchema("col3"),
            ColumnSchema("col4"),
        ]
    )

    assert result == expected
