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

HAS_GPU = False
try:
    import dask_cuda

    HAS_GPU = dask_cuda.utils.get_gpu_count()
except ImportError:
    # Don't let numba.cuda set the context
    # unless dask_cuda is not installed
    if cuda is not None:
        try:
            HAS_GPU = len(cuda.gpus.lst) > 0
        except cuda.cudadrv.error.CudaSupportError:
            pass
