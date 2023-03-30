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
    from numba import cuda
except ImportError:
    cuda = None

from merlin.core.has_gpu import HAS_GPU
from merlin.core.utils import device_mem_size

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

        # TODO: what will this look like in any sort
        # of distributed set up?
        if device is None:
            device = int(os.environ.get("TF_VISIBLE_DEVICE", 0))
        tf_devices = tf.config.list_physical_devices("GPU")
        if HAS_GPU and len(tf_devices) == 0:
            raise ImportError("TensorFlow is not configured for GPU")
        if HAS_GPU:
            try:
                tf.config.set_logical_device_configuration(
                    tf_devices[device],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_allocation)],
                )
            except RuntimeError:
                warnings.warn(
                    "TensorFlow runtime already initialized, may not be enough memory for cudf"
                )
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    tf_devices[device],
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
