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
try:
    from numba import cuda
except ImportError:
    cuda = None

from dask.distributed.diagnostics import nvml

# Using the `dask.distributed.diagnostics.nvml.device_get_count`
# helper function from dask to check device counts with NVML
# since this handles some complexity of checking NVML state for us.

# Note: We can't use `numba.cuda.gpus`, since this has some side effects
# that are incompatible with Dask-CUDA. If CUDA runtime functions are
# called before Dask-CUDA can spawn worker processes
# then Dask-CUDA it will not work correctly (raises an exception)

HAS_GPU = nvml.device_get_count() > 0
