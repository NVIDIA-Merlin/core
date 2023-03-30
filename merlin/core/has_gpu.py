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

# pylint: disable=unused-import
import os

from dask.distributed.diagnostics import nvml


def _get_gpu_count():
    """Get Number of GPU devices accounting for CUDA_VISIBLE_DEVICES environment variable"""
    # Using the `dask.distributed.diagnostics.nvml.device_get_count`
    # helper function from dask to check device counts with NVML
    # since this handles some complexity of checking NVML state for us.

    # Note: We can't use `numba.cuda.gpus`, since this has some side effects
    # that are incompatible with Dask-CUDA. If CUDA runtime functions are
    # called before Dask-CUDA can spawn worker processes
    # then Dask-CUDA it will not work correctly (raises an exception)
    nvml_device_count = nvml.device_get_count()
    if nvml_device_count == 0:
        return 0
    try:
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        if cuda_visible_devices:
            return len(cuda_visible_devices.split(","))
        else:
            return 0
    except KeyError:
        return nvml_device_count


HAS_GPU = _get_gpu_count() > 0
