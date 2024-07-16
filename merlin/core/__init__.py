#
# Copyright (c) 20222024, NVIDIA CORPORATION.
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

from merlin.core import _version

__version__ = _version.get_versions()["version"]


# If dask is installed, make sure query-planning is disabled
_DASK_QUERY_PLANNING_ENABLED = False
try:
    import dask

    dask.config.set({"dataframe.query-planning": False})
except ImportError:
    pass
else:
    import sys

    import dask.dataframe as dd
    from packaging.version import parse

    if parse(dask.__version__) > parse("2024.6.0"):
        _DASK_QUERY_PLANNING_ENABLED = dd.DASK_EXPR_ENABLED
    else:
        _DASK_QUERY_PLANNING_ENABLED = "dask_expr" in sys.modules


if _DASK_QUERY_PLANNING_ENABLED:
    raise NotImplementedError(
        "Merlin does not support the query-planning API in Dask "
        "Dataframe yet. Please make sure query-planning is "
        "disabled before dask.dataframe is imported. E.g.:\n"
        "dask.config.set({'dataframe.query-planning': False})"
    )
