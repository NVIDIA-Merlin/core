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
import pytest

from merlin.core.compat import cupy as cp
from merlin.core.compat import numpy as np
from merlin.core.compat import tensorflow as tf
from merlin.core.compat import torch as th
from merlin.table import CupyColumn, NumpyColumn, TensorflowColumn
from merlin.table.conversions import convert_col

source_cols = []
output_col_types = []

if cp:
    cp_array = cp.asarray([1, 2, 3, 4])
    source_cols.append(CupyColumn(values=cp_array, offsets=cp_array))
    output_col_types.append(CupyColumn)

if np:
    np_array = np.array([1, 2, 3, 4])
    source_cols.append(NumpyColumn(values=np_array, offsets=np_array))
    output_col_types.append(NumpyColumn)

if tf:
    with tf.device("/CPU"):
        tf_tensor = tf.random.uniform((10,))
        cpu_tf_column = TensorflowColumn(values=tf_tensor, offsets=tf_tensor)
    with tf.device("/GPU:0"):
        tf_tensor = tf.random.uniform((10,))
        gpu_tf_column = TensorflowColumn(values=tf_tensor, offsets=tf_tensor)
    source_cols.extend([cpu_tf_column, gpu_tf_column])
    output_col_types.append(TensorflowColumn)


@pytest.mark.parametrize("source_cols", source_cols)
@pytest.mark.parametrize("output_col", output_col_types)
def test_convert_col(source_cols, output_col):
    if source_cols.device not in output_col.supported_devices():
        with pytest.raises(NotImplementedError) as exc:
            converted_col = convert_col(source_cols, output_col)
        assert "Could not convert from type" in str(exc.value)
    else:
        converted_col = convert_col(source_cols, output_col)
        assert isinstance(converted_col, output_col)
