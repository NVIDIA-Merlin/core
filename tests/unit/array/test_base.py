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

import cupy as cp
import numpy as np

# tensorflow/cupy
import tensorflow as tf

from merlin.array.base import MerlinCupyArray, MerlinNumpyArray, MerlinTensorflowArray


def test_cp_merlinarray_to_cp_merlinarray():
    cp_tensor = cp.array([1, 2, 3, 4])

    cp_array = MerlinCupyArray(cp_tensor)
    cp_array2 = cp_array.to(MerlinCupyArray)
    assert isinstance(cp_array2.data, cp.ndarray)


def test_tf_merlinarray_to_tf_merlinarray():
    tf_tensor = tf.random.uniform((10,))

    tf_array = MerlinTensorflowArray(tf_tensor)
    tf_array2 = tf_array.to(MerlinTensorflowArray)
    assert isinstance(tf_array2.data, tf.Tensor)


def test_tf_merlinarray_to_cupy_merlinarray():
    tf_tensor = tf.random.uniform((10,))

    tf_array = MerlinTensorflowArray(tf_tensor)
    cp_array = tf_array.to(MerlinCupyArray)
    assert isinstance(cp_array.data, cp.ndarray)


def test_cupy_merlinarray_to_tf_merlinarray():
    cp_tensor = cp.array([1, 2, 3, 4])

    cp_array = MerlinCupyArray(cp_tensor)
    tf_array = cp_array.to(MerlinTensorflowArray)
    assert isinstance(tf_array.data, tf.Tensor)


def test_np_merlinarray_to_cp_merlinarray():
    np_tensor = np.array([1, 2, 3, 4])

    np_array = MerlinNumpyArray(np_tensor)
    cp_array = np_array.to(MerlinCupyArray)
    assert isinstance(cp_array.data, cp.ndarray)


def test_cp_merlinarray_to_np_merlinarray():
    cp_tensor = cp.array([1, 2, 3, 4])

    cp_array = MerlinCupyArray(cp_tensor)
    np_array = cp_array.to(MerlinNumpyArray)
    assert isinstance(np_array.data, np.ndarray)
