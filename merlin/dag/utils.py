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


def ungroup_values_offsets(grouped_cols: dict) -> dict:
    """
    Flatten columns with values/offsets tuples in a dictionary to separate keys

    Parameters
    ----------
    grouped_cols : dict
        A dictionary of column arrays including values/offsets tuples

    Returns
    -------
    dict
        A dictionary of column arrays with separate keys for values and offsets
    """
    flat_cols = {}

    for key, value in grouped_cols.items():
        if isinstance(value, tuple):
            flat_cols[f"{key}__values"] = value[0]
            flat_cols[f"{key}__offsets"] = value[1]
        else:
            flat_cols[key] = value

    return flat_cols


def group_values_offsets(flat_cols: dict) -> dict:
    """
    Convert separate values/offsets keys for columns into tuples w/ a single key

    Parameters
    ----------
    flat_cols : dict
        A dictionary of column arrays with separate keys for values and offsets

    Returns
    -------
    dict
        A dictionary of column arrays including values/offsets tuples
    """
    grouped_cols = {}

    for key, value in flat_cols.items():
        if key.endswith("__values"):
            col_name = key.replace("__values", "")
            grouped_cols[col_name] = (flat_cols[key], flat_cols[f"{col_name}__offsets"])
        elif key.endswith("__offsets"):
            pass
        else:
            grouped_cols[key] = value

    return grouped_cols
