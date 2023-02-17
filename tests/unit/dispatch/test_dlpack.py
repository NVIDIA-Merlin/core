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


def to_dlpack(to):
    return _to_dlpack(to)


@lazysingledispatch
def _to_dlpack(data):
    raise NotImplementedError


@_to_dlpack.register_lazy("cupy")
def register_to_dlpack_from_cupy():
    import cupy as cp  # pylint:disable=reimported

    @_to_dlpack.register(cp.ndarray)
    def _to_dlpack_from_cupy(data):
        return data.toDlpack()


@_to_dlpack.register_lazy("torch")
def register_to_dlpack_from_torch():
    import torch as th  # pylint:disable=reimported

    @_to_dlpack.register(th.Tensor)
    def _to_dlpack_from_torch(data):
        return th.utils.dlpack.to_dlpack(data)


@_to_dlpack.register_lazy("tensorflow")
def register_to_dlpack_from_tf():
    import tensorflow as tf

    @_to_dlpack.register(tf.Tensor)
    def _to_dlpack_from_tensorflow(data):
        return tf.experimental.dlpack.to_dlpack(data)


def from_dlpack(capsule, to):
    return _from_dlpack(to, capsule)


@lazysingledispatch
def _from_dlpack(dest_type, capsule):
    raise NotImplementedError


@_from_dlpack.register_lazy("cupy")
def register_from_dlpack_to_cupy():
    import cupy as cp  # pylint:disable=reimported

    @_from_dlpack.register(cp.ndarray)
    def _from_dlpack_to_cupy(dest_type, capsule):
        return cp.fromDlpack(capsule)


@_from_dlpack.register_lazy("torch")
def register_from_dlpack_to_torch():
    import torch as th  # pylint:disable=reimported

    @_from_dlpack.register(th.Tensor)
    def _from_dlpack_to_torch(dest_type, capsule):
        return th.utils.dlpack.from_dlpack(capsule)


@_from_dlpack.register_lazy("tensorflow")
def register_from_dlpack_to_tf():
    import tensorflow as tf

    @_from_dlpack.register(tf.Tensor)
    def _from_dlpack_to_tensorflow(dest_type, capsule):
        return tf.experimental.dlpack.from_dlpack(capsule)


# from cupy capsule
def test_cupy_from_cupy_pycapsule():
    import cupy as cp

    cp_arr = cp.asarray([1, 2, 3, 4])
    new_cp_arr = from_dlpack(to_dlpack(cp_arr), cp.ndarray)
    assert all(new_cp_arr == cp_arr)


def test_torch_from_cupy_pycapsule():
    import torch as th

    cp_arr = cp.asarray([1, 2, 3, 4])
    new_th_tensor = from_dlpack(to_dlpack(cp_arr), th.Tensor)
    assert all(new_th_tensor.cpu().numpy() == cp.asnumpy(cp_arr))


def test_tensorflow_from_cupy_pycapsule():
    import tensorflow as tf

    cp_arr = cp.asarray([1, 2, 3, 4])
    new_tf_tensor = from_dlpack(to_dlpack(cp_arr), tf.Tensor)
    assert all(new_tf_tensor.numpy() == cp.asnumpy(cp_arr))


import tensorflow as tf


# from tensorflow capsule
def test_cupy_from_tf_pycapsule():
    import cupy as cp

    with tf.device("/GPU:0"):
        # cannot use tf.constant here to create the representation
        tf_arr = tf.random.uniform((10,))
        new_cp_arr = from_dlpack(to_dlpack(tf_arr), cp.ndarray)
    assert all(cp.asnumpy(new_cp_arr) == tf_arr.numpy())


def test_torch_from_tf_pycapsule():
    import torch as th

    tf_arr = tf.constant([1, 2, 3, 4])
    new_th_tensor = from_dlpack(to_dlpack(tf_arr), th.Tensor)
    assert all(new_th_tensor.cpu().numpy() == tf_arr.numpy())


def test_tensorflow_from_tf_pycapsule():
    import tensorflow as tf

    tf_arr = tf.constant([1, 2, 3, 4])
    new_tf_tensor = from_dlpack(to_dlpack(tf_arr), tf.Tensor)
    assert all(new_tf_tensor.numpy() == tf_arr.numpy())


import torch


# from torch capsule
def test_cupy_from_torch_pycapsule():
    import cupy as cp

    torch_arr = torch.Tensor([12, 3, 4, 5]).cuda()
    new_cp_arr = from_dlpack(to_dlpack(torch_arr), cp.ndarray)
    assert all(cp.asnumpy(new_cp_arr) == torch_arr.cpu().numpy())


def test_torch_from_torch_pycapsule():
    import torch as th

    torch_arr = torch.Tensor([12, 3, 4, 5]).cuda()
    new_th_tensor = from_dlpack(to_dlpack(torch_arr), th.Tensor)
    assert all(new_th_tensor.cpu().numpy() == torch_arr.cpu().numpy())


def test_tensorflow_from_tf_pycapsule():
    import tensorflow as tf

    torch_arr = torch.Tensor([12, 3, 4, 5]).cuda()
    new_tf_tensor = from_dlpack(to_dlpack(torch_arr), tf.Tensor)
    assert all(new_tf_tensor.numpy() == torch_arr.cpu().numpy())
