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

# pylint: disable=unused-import
import warnings

from numba import cuda

from merlin.core.has_gpu import HAS_GPU  # noqa pylint: disable=unused-import

if not cuda.is_available():
    cuda = None

try:
    import psutil
except ImportError:
    psutil = None


def pynvml_mem_size(kind="total", index=0):
    """Get Memory Info for device.

    Parameters
    ----------
    kind : str, optional
        Either "free" or "total", by default "total"
    index : int, optional
        Device Index, by default 0

    Returns
    -------
    int
        Either free or total memory on device depending on the kind parameter.

    Raises
    ------
    ValueError
        When kind is not one of {"free", "total"}
    """
    import pynvml

    pynvml.nvmlInit()
    size = None
    if kind == "free":
        size = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index)).free)
    elif kind == "total":
        size = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index)).total)
    else:
        raise ValueError(f"{kind} not a supported option for device_mem_size.")
    pynvml.nvmlShutdown()
    return size


def device_mem_size(kind="total", cpu=False):
    """Get Memory Info for either CPU or GPU.

    Parameters
    ----------
    kind : str, optional
        Either "total" or "free", by default "total"
    cpu : bool, optional
        Specifies whether to check memory for CPU or GPU, by default False

    Returns
    -------
    int
        Free or total memory on device

    Raises
    ------
    ValueError
        When kind is provided with an unsupported value.
    """
    # Use psutil (if available) for cpu mode
    cpu = cpu or not cuda
    if cpu and psutil:
        if kind == "total":
            return psutil.virtual_memory().total
        elif kind == "free":
            return psutil.virtual_memory().free
    elif cpu:
        warnings.warn("Please install psutil for full cpu=True support.")
        # Assume 1GB of memory
        return int(1e9)

    if kind not in ["free", "total"]:
        raise ValueError(f"{kind} not a supported option for device_mem_size.")
    try:
        if kind == "free":
            return int(cuda.current_context().get_memory_info()[0])
        else:
            return int(cuda.current_context().get_memory_info()[1])
    except NotImplementedError:
        if kind == "free":
            # Not using NVML "free" memory, because it will not include RMM-managed memory
            warnings.warn("get_memory_info is not supported. Using total device memory from NVML.")
        size = pynvml_mem_size(kind="total", index=0)
        return size


try:
    import numpy
except ImportError:
    numpy = None

try:
    import pandas
except ImportError:
    pandas = None

if HAS_GPU:
    try:
        import cupy
    except ImportError:
        cupy = None

    try:
        import cudf
    except ImportError:
        cudf = None

    try:
        import dask_cudf
    except ImportError:
        dask_cudf = None
else:
    # Without a GPU available none of these packages should be used
    cupy = None
    cudf = None
    dask_cudf = None
