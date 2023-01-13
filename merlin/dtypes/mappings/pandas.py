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
import numpy as np

import merlin.dtypes.aliases as mn
from merlin.dtypes.mapping import DTypeMapping, NumpyPreprocessor
from merlin.dtypes.registry import _dtype_registry

try:
    import pandas as pd

    pandas_dtypes = DTypeMapping(
        {
            mn.string: [pd.StringDtype(), pd.StringDtype],
            mn.boolean: [pd.BooleanDtype(), pd.BooleanDtype],
        },
        translator=NumpyPreprocessor(
            "pandas", lambda raw: np.dtype(raw.numpy_dtype), attrs=["numpy_dtype"]
        ),
    )
    _dtype_registry.register("pandas", pandas_dtypes)
except ImportError as exc:
    from warnings import warn

    warn(f"Pandas dtype mappings did not load successfully due to an error: {exc.msg}")
