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

from abc import ABC, abstractmethod

import requests


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


class ModelRegistry(ABC):
    """
    The ModelRegistry class is used to find model paths that will be imported into an
    InferenceOperator.

    To implement your own ModelRegistry subclass, the only method that must be implemented is
    `get_artifact_uri`, which must return a string indicating the model's export path.

    ```python
    PredictTensorflow.from_model_registry(
        MyModelRegistry("model_name", "model_version")
    )
    ```
    """

    @abstractmethod
    def get_artifact_uri(self) -> str:
        """
        This returns the URI of the model artifact.
        """


class MLFlowModelRegistry(ModelRegistry):
    def __init__(self, name: str, version: str, tracking_uri: str):
        """
        Fetches the model path from an mlflow model registry.

        Note that this will return a relative path if you did not configure your mlflow
        experiment's `artifact_location` to be an absolute path.

        Parameters
        ----------
        name : str
            Name of the model in the mlflow registry.
        version : str
            Version of the model to use.
        tracking_uri : str
            Base URI of the mlflow tracking server. If running locally, this would likely be
            http://localhost:5000
        """
        self.name = name
        self.version = version
        self.tracking_uri = tracking_uri.rstrip("/")

    def get_artifact_uri(self) -> str:
        mv = requests.get(
            f"{self.tracking_uri}/ajax-api/2.0/preview/mlflow/model-versions/get-download-uri",
            params={"name": self.name, "version": self.version},
        )

        if mv.status_code != 200:
            raise ValueError(
                f"Could not find a Model Version for model {self.name} with version {self.version}."
            )
        model_path = mv.json()["artifact_uri"]
        return model_path
