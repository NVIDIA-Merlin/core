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


def to_cuda_array(to):
    return _to_cuda_array(to)


@lazysingledispatch
def _to_cuda_array(data):
    raise NotImplementedError


@_to_cuda_array.register_lazy("cupy")
def register_to_cuda_array_from_cupy():
    import cupy as cp  # pylint:disable=reimported

    @_to_cuda_array.register(cp.ndarray)
    def _to_cuda_array_from_cupy(data):
        return data


@_to_cuda_array.register_lazy("torch")
def register_to_cuda_array_from__torch():
    import torch as th  # pylint:disable=reimported

    @_to_cuda_array.register(th.Tensor)
    def _to_cuda_array_from_torch(data):
        return data


def from_cuda_array(capsule, to):
    return _from_cuda_array(to, capsule)


@lazysingledispatch
def _from_cuda_array(dest_type, data):
    raise NotImplementedError


@_from_cuda_array.register_lazy("cupy")
def register_from_dlpack_to_cupy():
    import cupy as cp  # pylint:disable=reimported

    @_from_cuda_array.register(cp.ndarray)
    def _from_cuda_array_to_cupy(dest_type, data):
        return cp.asarray(data)


@_from_cuda_array.register_lazy("torch")
def register_from_cuda_array_to_torch():
    import torch as th  # pylint:disable=reimported

    @_from_cuda_array.register(th.Tensor)
    def _from_cuda_array_to_torch(dest_type, data):
        return th.as_tensor(data)


# from cupy capsule
def test_cupy_from_cupy_cuda_array():
    import cupy as cp

    cp_arr = cp.asarray([1, 2, 3, 4])
    new_cp_arr = from_cuda_array(to_cuda_array(cp_arr), cp.ndarray)
    assert all(new_cp_arr == cp_arr)


def test_torch_from_cupy_cuda_array():
    import torch as th

    cp_arr = cp.asarray([1, 2, 3, 4])
    new_th_tensor = from_cuda_array(to_cuda_array(cp_arr), th.Tensor)
    assert all(new_th_tensor.cpu().numpy() == cp.asnumpy(cp_arr))


import torch


# from torch capsule
def test_cupy_from_torch_cuda_array():
    import cupy as cp

    torch_arr = torch.Tensor([12, 3, 4, 5]).cuda()
    new_cp_arr = from_cuda_array(to_cuda_array(torch_arr), cp.ndarray)
    assert all(cp.asnumpy(new_cp_arr) == torch_arr.cpu().numpy())


def test_torch_from_torch_cuda_array():
    import torch as th

    torch_arr = torch.Tensor([12, 3, 4, 5]).cuda()
    new_th_tensor = from_cuda_array(to_cuda_array(torch_arr), th.Tensor)
    assert all(new_th_tensor.cpu().numpy() == torch_arr.cpu().numpy())
