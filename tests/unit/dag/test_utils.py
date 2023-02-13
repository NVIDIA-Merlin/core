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

from merlin.dag import group_values_offsets, ungroup_values_offsets


def test_flat_dict_to_tuple_dict():
    col1 = np.array([1, 2, 3, 4, 5])
    col2_values = np.array([6, 7, 8, 9, 10])
    col2_offsets = np.array([0, 2, 5])

    flat_dict = {"col1": col1, "col2__values": col2_values, "col2__offsets": col2_offsets}

    tuple_dict = {"col1": col1, "col2": (col2_values, col2_offsets)}

    assert ungroup_values_offsets(tuple_dict) == flat_dict
    assert group_values_offsets(flat_dict) == tuple_dict
