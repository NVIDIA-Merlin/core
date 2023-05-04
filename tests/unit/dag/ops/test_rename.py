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
import numpy as np
import pandas as pd
import pytest

from merlin.core.compat import cudf
from merlin.dag import ColumnSelector
from merlin.dag.ops.rename import Rename
from merlin.table import TensorTable
from tests.conftest import assert_eq

transformables = [pd.DataFrame, TensorTable]
if cudf:
    transformables.append(cudf.DataFrame)


@pytest.mark.parametrize("transformable", transformables)
def test_rename(transformable):
    df = transformable({"x": np.array([1, 2, 3, 4, 5]), "y": np.array([6, 7, 8, 9, 10])})

    selector = ColumnSelector(["x", "y"])

    op = Rename(f=lambda name: name.upper())
    transformed = op.transform(selector, df)
    expected = transformable({"X": np.array([1, 2, 3, 4, 5]), "Y": np.array([6, 7, 8, 9, 10])})
    assert_eq(transformed, expected)

    op = Rename(postfix="_lower")
    transformed = op.transform(selector, df)
    expected = transformable(
        {"x_lower": np.array([1, 2, 3, 4, 5]), "y_lower": np.array([6, 7, 8, 9, 10])}
    )
    assert_eq(transformed, expected)

    selector = ColumnSelector(["x"])

    op = Rename(name="z")
    transformed = op.transform(selector, df)
    expected = transformable({"z": np.array([1, 2, 3, 4, 5])})
    assert_eq(transformed, expected)

    op = Rename(f=lambda name: name.upper())
    transformed = op.transform(selector, df)
    expected = transformable({"X": np.array([1, 2, 3, 4, 5])})
    assert_eq(transformed, expected)
