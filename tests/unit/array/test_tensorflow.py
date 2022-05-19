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
import tensorflow as tf

from merlin.array.cupy import MerlinCupyArray
from merlin.array.tensorflow import MerlinTensorflowArray


def test_tf_merlinarray_to_tf_merlinarray():
    tf_tensor = tf.random.uniform((10,))

    tf_array = MerlinTensorflowArray(tf_tensor)
    tf_array2 = tf_array.to(MerlinTensorflowArray)
    assert isinstance(tf_array2.data, tf.Tensor)
    assert (tf_array.data.numpy() == tf_array2.data.numpy()).all()


def test_tf_merlinarray_to_cupy_merlinarray():
    tf_tensor = tf.random.uniform((10,))

    tf_array = MerlinTensorflowArray(tf_tensor)
    cp_array = tf_array.to(MerlinCupyArray)
    assert isinstance(cp_array.data, cp.ndarray)
    assert (tf_array.data.numpy() == cp.asnumpy(cp_array.data)).all()
