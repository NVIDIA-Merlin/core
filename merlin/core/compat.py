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
import os
import warnings

from packaging import version

try:
    import psutil
except ImportError:
    psutil = None

try:
    from numba import cuda
except ImportError:
    cuda = None

from merlin.core.has_gpu import HAS_GPU


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
    import cupy
except ImportError:
    cupy = None

try:
    import cudf
except ImportError:
    cudf = None

try:
    import tensorflow

    def configure_tensorflow(memory_allocation=None, device=None):
        """Utility to help configure tensorflow to not use 100% of gpu memory as buffer"""
        tf = tensorflow
        total_gpu_mem_mb = device_mem_size(kind="total", cpu=(not HAS_GPU)) / (1024**2)

        if memory_allocation is None:
            memory_allocation = os.environ.get("TF_MEMORY_ALLOCATION", 0.5)

        if float(memory_allocation) < 1:
            memory_allocation = total_gpu_mem_mb * float(memory_allocation)
        memory_allocation = int(memory_allocation)
        assert memory_allocation < total_gpu_mem_mb

        if HAS_GPU:
            tf_devices = tf.config.list_physical_devices("GPU")

            if len(tf_devices) == 0:
                raise ImportError("TensorFlow is not configured for GPU")

            for tf_device in tf_devices:
                try:
                    tf.config.set_logical_device_configuration(
                        tf_device,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_allocation)],
                    )
                except RuntimeError:
                    warnings.warn(
                        "TensorFlow runtime already initialized, may not be enough memory for cudf"
                    )
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        tf_device,
                        [
                            tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=memory_allocation
                            )
                        ],
                    )
                except RuntimeError as e:
                    # Virtual devices must be set before GPUs have been initialized
                    warnings.warn(str(e))

        # versions using TF earlier than 2.3.0 need to use extension
        # library for dlpack support to avoid memory leak issue
        __TF_DLPACK_STABLE_VERSION = "2.3.0"
        if version.parse(tf.__version__) < version.parse(__TF_DLPACK_STABLE_VERSION):
            try:
                from tfdlpack import from_dlpack
            except ModuleNotFoundError as e:
                message = (
                    "If using TensorFlow < 2.3.0, you must install tfdlpack-gpu extension library"
                )
                raise ModuleNotFoundError(message) from e

        else:
            from tensorflow.experimental.dlpack import from_dlpack

        return from_dlpack

    configure_tensorflow()

    from tensorflow.python.framework import ops as tf_ops
except ImportError:
    tensorflow = None
    tf_ops = None

try:
    import torch
except ImportError:
    torch = None
