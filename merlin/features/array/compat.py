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

try:
    import cudf  # noqa
except ImportError:
    cudf = None

try:
    import cupy  # noqa
except ImportError:
    cupy = None

try:
    import numpy  # noqa
except ImportError:
    numpy = None

try:
    import pandas  # noqa
except ImportError:
    pandas = None

try:
    import tensorflow  # noqa

    if not hasattr(tensorflow, "Tensor"):
        tensorflow = None
except ImportError:
    tensorflow = None
