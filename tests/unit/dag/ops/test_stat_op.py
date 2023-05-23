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
from typing import Dict

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from merlin.core.compat import cudf
from merlin.dag import ColumnSelector, Graph
from merlin.dag.executors import DaskExecutor
from merlin.dag.ops.stat_operator import StatOperator
from merlin.io.dataset import Dataset

transformables = [pd.DataFrame]
if cudf:
    transformables.append(cudf.DataFrame)


class FitOp(StatOperator):
    def fit(self, col_selector: ColumnSelector, ddf: dd.DataFrame):
        fit_exactly_once = "fit_exactly_once" not in self.stats
        self.stats: Dict[str, bool] = {"fit_exactly_once": fit_exactly_once}

    def fit_finalize(self, dask_stats):
        return dask_stats

    def clear(self):
        self.stats = {}


@pytest.mark.parametrize("transformable", transformables)
@pytest.mark.parametrize("engine", ["parquet"])
def test_fitted_stat_op(transformable, engine):
    df = transformable({"x": np.array([1, 2, 3, 4, 5]), "y": np.array([6, 7, 8, 9, 10])})

    op = FitOp()
    graph = ["x", "y"] >> op
    graph = Graph(graph)
    executor = DaskExecutor()

    executor.fit(Dataset(df), graph)
    assert op.stats == {"fit_exactly_once": True}

    executor.fit(Dataset(df), graph, refit=False)
    assert op.stats == {"fit_exactly_once": True}


@pytest.mark.parametrize("transformable", transformables)
@pytest.mark.parametrize("engine", ["parquet"])
def test_fit_op_before_transfrom(transformable, engine):
    df = transformable({"x": np.array([1, 2, 3, 4, 5]), "y": np.array([6, 7, 8, 9, 10])})

    op = FitOp()
    graph = ["x", "y"] >> op
    graph = Graph(graph)
    executor = DaskExecutor()
    graph.construct_schema(Dataset(df).schema)
    with pytest.raises(RuntimeError) as exc:
        executor.transform(Dataset(df).to_ddf(), graph)
    assert "attempting to use them to transform data" in str(exc.value)

    executor.fit(Dataset(df), graph)
    executor.transform(Dataset(df).to_ddf(), graph)
    assert op.stats == {"fit_exactly_once": True}
