#
# Copyright (c) 2024, NVIDIA CORPORATION.
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

_DASK_QUERY_PLANNING_ENABLED = False
try:
    # Disable query-planning and string conversion
    import dask

    dask.config.set(
        {
            "dataframe.query-planning": False,
            "dataframe.convert-string": False,
        }
    )
except ImportError:
    pass
else:
    import sys

    import dask.dataframe as dd
    from packaging.version import parse

    if parse(dask.__version__) > parse("2024.6.0"):
        # For newer versions of dask, we can just check
        # the official DASK_EXPR_ENABLED constant
        _DASK_QUERY_PLANNING_ENABLED = dd.DASK_EXPR_ENABLED
    else:
        # For older versions of dask, we must assume query
        # planning is enabled if dask_expr was imported
        # (because we can't know for sure)
        _DASK_QUERY_PLANNING_ENABLED = "dask_expr" in sys.modules


if _DASK_QUERY_PLANNING_ENABLED:
    raise NotImplementedError(
        "Merlin does not support the query-planning API in Dask "
        "Dataframe yet. Please make sure query-planning is "
        "disabled before dask.dataframe is imported.\n\n"
        "e.g. dask.config.set({'dataframe.query-planning': False})"
        "\n\nOr set the environment variable: "
        "export DASK_DATAFRAME__QUERY_PLANNING=False"
    )
