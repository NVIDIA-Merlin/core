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
import cupy as cp
import torch as th

from merlin.dispatch.lazy import lazysingledispatch


def to_array(to):
    return _to_array(to)


@lazysingledispatch
def _to_array(data):
    raise NotImplementedError


@_to_array.register_lazy("cupy")
def register_to_array_from_cupy():
    import cupy as cp  # pylint:disable=reimported

    @_to_array.register(cp.ndarray)
    def _to_array_from_cupy(data):
        return cp.asnumpy(data)


@_to_array.register_lazy("torch")
def register_to_array_from_torch():
    import torch as th  # pylint:disable=reimported

    @_to_array.register(th.Tensor)
    def _to_array_from_torch(data):
        return data.cpu().numpy()


@_to_array.register_lazy("tensorflow")
def register_to_array_from_tf():
    import tensorflow as tf

    @_to_array.register(tf.Tensor)
    def _to_array_from_tensorflow(data):
        return data.numpy()


def from_array(capsule, to):
    return _from_array(to, capsule)


@lazysingledispatch
def _from_array(dest_type, data):
    raise NotImplementedError


@_from_array.register_lazy("cupy")
def register_from_array_to_cupy():
    import cupy as cp  # pylint:disable=reimported

    @_from_array.register(cp.ndarray)
    def _from_array_to_cupy(dest_type, data):
        return cp.asarray(data)


@_from_array.register_lazy("torch")
def register_from_array_to_torch():
    import torch as th  # pylint:disable=reimported

    @_from_array.register(th.Tensor)
    def _from_array_to_torch(dest_type, data):
        return th.as_tensor(data)


@_from_array.register_lazy("tensorflow")
def register_from_array_to_tf():
    import tensorflow as tf

    @_from_array.register(tf.Tensor)
    def _from_array_to_tensorflow(dest_type, data):
        return tf.convert_to_tensor(data)


# from cupy capsule
def test_cupy_from_cupy_pycapsule():
    import cupy as cp

    cp_arr = cp.asarray([1, 2, 3, 4])
    new_cp_arr = from_array(to_array(cp_arr), cp.ndarray)
    assert all(new_cp_arr == cp_arr)


def test_torch_from_cupy_pycapsule():
    import torch as th

    cp_arr = cp.asarray([1, 2, 3, 4])
    new_th_tensor = from_array(to_array(cp_arr), th.Tensor)
    assert all(new_th_tensor.cpu().numpy() == cp.asnumpy(cp_arr))


def test_tensorflow_from_cupy_pycapsule():
    import tensorflow as tf

    cp_arr = cp.asarray([1, 2, 3, 4])
    new_tf_tensor = from_array(to_array(cp_arr), tf.Tensor)
    assert all(new_tf_tensor.numpy() == cp.asnumpy(cp_arr))


import tensorflow as tf


# from tensorflow capsule
def test_cupy_from_tf_pycapsule():
    import cupy as cp

    with tf.device("/GPU:0"):
        # cannot use tf.constant here to create the representation
        tf_arr = tf.random.uniform((10,))
        new_cp_arr = from_array(to_array(tf_arr), cp.ndarray)
    assert all(cp.asnumpy(new_cp_arr) == tf_arr.numpy())


def test_torch_from_tf_pycapsule():
    import torch as th

    tf_arr = tf.constant([1, 2, 3, 4])
    new_th_tensor = from_array(to_array(tf_arr), th.Tensor)
    assert all(new_th_tensor.cpu().numpy() == tf_arr.numpy())


def test_tensorflow_from_tf_pycapsule():
    import tensorflow as tf

    tf_arr = tf.constant([1, 2, 3, 4])
    new_tf_tensor = from_array(to_array(tf_arr), tf.Tensor)
    assert all(new_tf_tensor.numpy() == tf_arr.numpy())


import torch


# from torch capsule
def test_cupy_from_torch_pycapsule():
    import cupy as cp

    torch_arr = torch.Tensor([12, 3, 4, 5]).cuda()
    new_cp_arr = from_array(to_array(torch_arr), cp.ndarray)
    assert all(cp.asnumpy(new_cp_arr) == torch_arr.cpu().numpy())


def test_torch_from_torch_pycapsule():
    import torch as th

    torch_arr = torch.Tensor([12, 3, 4, 5]).cuda()
    new_th_tensor = from_array(to_array(torch_arr), th.Tensor)
    assert all(new_th_tensor.cpu().numpy() == torch_arr.cpu().numpy())


def test_tensorflow_from_tf_pycapsule():
    import tensorflow as tf

    torch_arr = torch.Tensor([12, 3, 4, 5]).cuda()
    new_tf_tensor = from_array(to_array(torch_arr), tf.Tensor)
    assert all(new_tf_tensor.numpy() == torch_arr.cpu().numpy())
