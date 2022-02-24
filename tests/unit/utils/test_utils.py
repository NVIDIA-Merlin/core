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

import pytest

from merlin.core.utils import Distributed, Serial, global_dask_client, set_dask_client

try:
    import cudf

    _CPU = [True, False]
except ImportError:
    _CPU = [True]
    cudf = None
_HAS_GPU = cudf is not None


@pytest.mark.parametrize("cpu", _CPU)
def test_serial_context(client, cpu):

    # Set distributed client
    set_dask_client(client=client)
    assert global_dask_client() == client

    # Check that the global dask client
    # becomes None in a `with Serial()` block
    with Serial():
        assert global_dask_client() is None

    # Global client should revert outside
    # the `with Serial()` block
    assert global_dask_client() == client


@pytest.mark.parametrize("cpu", [True, False])
@pytest.mark.parametrize("nested_serial", _CPU)
def test_nvt_distributed(cpu, nested_serial):

    if cpu:
        distributed = pytest.importorskip("distributed")
        cluster_type = "cpu"
        cluster_cls = distributed.LocalCluster
    else:
        dask_cuda = pytest.importorskip("dask_cuda")
        cluster_type = "cuda"
        cluster_cls = dask_cuda.LocalCUDACluster

    # Set the global client to None
    set_dask_client(client=None)
    assert global_dask_client() is None

    # Check that a new local cluster is deployed within
    # a `with Distributed()` block
    with Distributed(cluster_type=cluster_type, n_workers=1, force_new=True) as dist:
        assert dist.client is not None
        assert global_dask_client() == dist.client
        assert len(dist.cluster.workers) == 1
        assert isinstance(dist.cluster, cluster_cls)

        # Check that we can nest a `with Serial()` block
        # inside a `with Distributed()` block
        if nested_serial:
            with Serial():
                assert global_dask_client() is None
            assert global_dask_client() == dist.client

    # Global client should revert to None outside
    # the `with Distributed()` block
    assert global_dask_client() is None


@pytest.mark.parametrize("cpu", _CPU)
def test_nvt_distributed_force(client, cpu):

    if cpu:
        distributed = pytest.importorskip("distributed")
        cluster_type = "cpu"
        cluster_cls = distributed.LocalCluster
    else:
        dask_cuda = pytest.importorskip("dask_cuda")
        cluster_type = "cuda"
        cluster_cls = dask_cuda.LocalCUDACluster

    # Set distributed client
    set_dask_client(client=client)
    assert global_dask_client() == client

    # Check that a new local cluster is deployed within
    # a `with Distributed()` block. Since we are using
    # `force_new=True`, the new cluster should NOT be
    # the same as the original `client`.
    with Distributed(cluster_type=cluster_type, force_new=True, n_workers=1) as dist:
        assert dist.client != client
        assert global_dask_client() == dist.client
        assert len(dist.cluster.workers) == 1
        assert isinstance(dist.cluster, cluster_cls)

    # We should revert to the original client
    # outside the `with Distributed()` block
    assert global_dask_client() == client

    # Check that the default behavior is to avoid
    # deploying a new cluster (and warning the user)
    # if an existing client is detected
    with pytest.warns(UserWarning):
        with Distributed(cluster_type=cluster_type, n_workers=1) as dist:
            assert dist.client == client
            assert global_dask_client() == dist.client

    # We should revert to the original client
    # outside the `with Distributed()` block
    assert global_dask_client() == client
