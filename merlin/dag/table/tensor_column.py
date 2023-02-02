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
from functools import singledispatch
from typing import Any, Union

import merlin.dtypes as md
from merlin.core.compat import cupy as cp
from merlin.core.compat import numpy as np
from merlin.core.compat import tensorflow as tf
from merlin.core.compat import tf_ops
from merlin.core.compat import torch as th
from merlin.core.protocols import SeriesLike


# This should always contains arrays or tensors, not series
class TensorColumn(SeriesLike):
    """
    A simple wrapper around an array of values
    """

    def __init__(self, values: Any, offsets: Any = None, dtype=None):
        super().__init__()

        self.values = values
        self.offsets = offsets
        self.dtype = md.dtype(dtype or values.dtype)
