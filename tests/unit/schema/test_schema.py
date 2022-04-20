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

from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema

# from merlin.schema.tags import Tags, TagSet


def test_select():
    col1_schema = ColumnSchema("col1", tags=["a", "b", "c"])
    col2_schema = ColumnSchema("col2", tags=["b", "c", "d"])
    schema = Schema([col1_schema, col2_schema])

    col1_selection = schema.select(ColumnSelector(["col1"]))
    col2_selection = schema.select(ColumnSelector(["col2"]))

    assert col1_selection.column_schemas == {"col1": col1_schema}
    assert col2_selection.column_schemas == {"col2": col2_schema}

    multi_selection = schema.select(ColumnSelector(["col1", "col2"]))
    empty_selection = schema.select(ColumnSelector(["col3"]))

    assert multi_selection.column_schemas == {"col1": col1_schema, "col2": col2_schema}
    assert empty_selection == Schema([])


def test_dataset_schema_select_by_name():
    # Shrink this down, so it only tests passing the names and creating a selector
    schema1 = ColumnSchema("col1", tags=["a", "b", "c"])
    schema2 = ColumnSchema("col2", tags=["b", "c", "d"])

    ds_schema = Schema([schema1, schema2])

    selected_schema1 = ds_schema.select_by_name("col1")
    selected_schema2 = ds_schema.select_by_name("col2")

    assert selected_schema1.column_schemas == {"col1": schema1}
    assert selected_schema2.column_schemas == {"col2": schema2}

    selected_schema_multi = ds_schema.select_by_name(["col1", "col2"])

    assert selected_schema_multi.column_schemas == {"col1": schema1, "col2": schema2}

    assert ds_schema.select_by_name("col3") == Schema([])


def test_dataset_schema_select_by_tag():
    schema1 = ColumnSchema("col1", tags=["a", "b", "c"])
    schema2 = ColumnSchema("col2", tags=["b", "c", "d"])

    ds_schema = Schema([schema1, schema2])

    selected_schema1 = ds_schema.select_by_tag("a")
    selected_schema2 = ds_schema.select_by_tag("d")

    assert selected_schema1.column_schemas == {"col1": schema1}
    assert selected_schema2.column_schemas == {"col2": schema2}

    selected_schema_both = ds_schema.select_by_tag("c")
    selected_schema_neither = ds_schema.select_by_tag("e")
    selected_schema_multi = ds_schema.select_by_tag(["b", "c"])

    assert selected_schema_both.column_schemas == {"col1": schema1, "col2": schema2}
    assert selected_schema_neither.column_schemas == {}
    assert selected_schema_multi.column_schemas == {"col1": schema1, "col2": schema2}


def test_excluding():
    pass


# def test_excluding_by_name()
# def test_excluding_by_tag()
# def test_remove_col()
# def without()
