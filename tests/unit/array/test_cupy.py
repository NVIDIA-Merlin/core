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

import cupy as cp
import numpy as np
import tensorflow as tf

from merlin.array.cupy import MerlinCupyArray
from merlin.array.numpy import MerlinNumpyArray
from merlin.array.tensorflow import MerlinTensorflowArray


def test_cp_merlinarray_to_cp_merlinarray():
    cp_tensor = cp.array([1, 2, 3, 4])

    cp_array = MerlinCupyArray(cp_tensor)
    cp_array2 = cp_array.to(MerlinCupyArray)
    assert isinstance(cp_array2.data, cp.ndarray)
    assert (cp.asnumpy(cp_array.data) == cp.asnumpy(cp_array2.data)).all()


def test_cupy_merlinarray_to_tf_merlinarray():
    cp_tensor = cp.array([1, 2, 3, 4])

    cp_array = MerlinCupyArray(cp_tensor)
    tf_array = cp_array.to(MerlinTensorflowArray)
    assert isinstance(tf_array.data, tf.Tensor)
    assert (tf_array.data.numpy() == cp.asnumpy(cp_array.data)).all()


def test_cp_merlinarray_to_np_merlinarray():
    cp_tensor = cp.array([1, 2, 3, 4])

    cp_array = MerlinCupyArray(cp_tensor)
    np_array = cp_array.to(MerlinNumpyArray)
    assert isinstance(np_array.data, np.ndarray)
    assert (np_array.data == cp.asnumpy(cp_array.data)).all()
