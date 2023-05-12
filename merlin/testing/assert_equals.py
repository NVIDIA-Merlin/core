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

from merlin.core.compat import pandas as pd
from merlin.dispatch.lazy import lazy_singledispatch
from merlin.table import TensorTable


def assert_table_equal(left: TensorTable, right: TensorTable):
    pd.testing.assert_frame_equal(left.cpu().to_df(), right.cpu().to_df())


@lazy_singledispatch
def assert_transformable_equal(left, right):
    raise NotImplementedError


@assert_transformable_equal.register(TensorTable)
def _assert_equal_table(left, right):
    assert_table_equal(left, right)


@assert_transformable_equal.register_lazy("cudf")
def _register_assert_equal_df_cudf():
    import cudf

    @assert_transformable_equal.register(cudf.DataFrame)
    def _assert_equal_df_cudf(left, right):
        cudf.testing.assert_frame_equal(left, right)


@assert_transformable_equal.register_lazy("pandas")
def _register_assert_equal_pandas():
    import pandas

    @assert_transformable_equal.register(pandas.DataFrame)
    def _assert_equal_pandas(left, right):
        pandas.testing.assert_frame_equal(left, right)
