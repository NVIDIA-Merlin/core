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
import numpy
import pytest

from merlin.schema import ColumnSchema
from merlin.schema.schema import ColumnQuantity
from merlin.schema.tags import Tags, TagSet


@pytest.mark.parametrize("d_types", [numpy.float32, numpy.float64, numpy.uint32, numpy.uint64])
def test_dtype_column_schema(d_types):
    column = ColumnSchema("name", tags=[], properties=[], dtype=d_types)
    assert column.dtype == d_types


def test_column_schema_meta():
    column = ColumnSchema("name", tags=["tag-1"], properties={"p1": "prop-1"})

    assert column.name == "name"
    assert "tag-1" in column.tags
    assert column.with_name("a").name == "a"
    assert set(column.with_tags("tag-2").tags) == set(["tag-1", "tag-2"])
    assert column.with_properties({"p2": "prop-2"}).properties == {
        "p1": "prop-1",
        "p2": "prop-2",
    }
    assert column.with_tags("tag-2").properties == {"p1": "prop-1"}
    assert set(column.with_properties({"p2": "prop-2"}).tags) == set(["tag-1"])

    assert column == ColumnSchema("name", tags=["tag-1"], properties={"p1": "prop-1"})
    # should not be the same no properties
    assert column != ColumnSchema("name", tags=["tag-1"])
    # should not be the same no tags
    assert column != ColumnSchema("name", properties={"p1": "prop-1"})


def test_column_schema_tags_normalize():
    schema1 = ColumnSchema("col1", tags=["categorical", "list", "item_id"])
    assert schema1.tags == TagSet([Tags.CATEGORICAL, Tags.LIST, Tags.ITEM_ID])


def test_list_column_attributes():
    col0_schema = ColumnSchema("col0")

    assert not col0_schema.is_list
    assert not col0_schema.is_ragged
    assert col0_schema.quantity == ColumnQuantity.SCALAR

    col1_schema = ColumnSchema("col1", is_list=False, is_ragged=False)

    assert not col1_schema.is_list
    assert not col1_schema.is_ragged
    assert col1_schema.quantity == ColumnQuantity.SCALAR

    col2_schema = ColumnSchema("col2", is_list=True)

    assert col2_schema.is_list
    assert col2_schema.is_ragged
    assert col2_schema.quantity == ColumnQuantity.RAGGED_LIST

    col3_schema = ColumnSchema("col3", is_list=True, is_ragged=True)

    assert col3_schema.is_list
    assert col3_schema.is_ragged
    assert col3_schema.quantity == ColumnQuantity.RAGGED_LIST

    col4_schema = ColumnSchema("col4", is_list=True, is_ragged=False)

    assert col4_schema.is_list
    assert not col4_schema.is_ragged
    assert col4_schema.quantity == ColumnQuantity.FIXED_LIST

    with pytest.raises(ValueError):
        ColumnSchema("col5", is_list=False, is_ragged=True)
