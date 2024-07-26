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
    dask = None
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


def validate_dask_configs():
    """Central check for problematic config options in Dask"""
    if _DASK_QUERY_PLANNING_ENABLED:
        raise NotImplementedError(
            "Merlin does not support the query-planning API in "
            "Dask Dataframe yet. Please make sure query-planning is "
            "disabled before dask.dataframe is imported.\n\ne.g."
            "dask.config.set({'dataframe.query-planning': False})"
            "\n\nOr set the environment variable: "
            "export DASK_DATAFRAME__QUERY_PLANNING=False"
        )

    if dask is not None and dask.config.get("dataframe.convert-string"):
        raise NotImplementedError(
            "Merlin does not support automatic string conversion in "
            "Dask Dataframe yet. Please make sure this option is "
            "disabled.\n\ne.g."
            "dask.config.set({'dataframe.convert-string': False})"
            "\n\nOr set the environment variable: "
            "export DASK_DATAFRAME__CONVERT_STRING=False"
        )
