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
from merlin.features.array.base import MerlinArray
from merlin.features.compat import numpy
from merlin.features.df import VirtualDataFrame


def test_virtual_dataframe_set_item():
    df = VirtualDataFrame()

    values = numpy.random.randint(0, 9, 5)
    df["col_name"] = values
    assert (df["col_name"] == values).all()

    merlin_array = MerlinArray.build(values)
    df["col_name"] = merlin_array
    assert (df["col_name"] == merlin_array.array).all()
    assert df._col_data["col_name"] == merlin_array
