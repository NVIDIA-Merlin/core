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
from typing import Protocol, runtime_checkable


@runtime_checkable
class DictLike(Protocol):
    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    def __getitem__(self, key):
        ...

    def __setitem__(self, key, value):
        ...

    def __delitem__(self, key):
        ...

    def keys(self):
        ...

    def items(self):
        ...

    def values(self):
        ...

    def update(self, other):
        ...

    def copy(self):
        ...


@runtime_checkable
class SeriesLike(Protocol):
    def values(self):
        ...

    def dtype(self):
        ...

    def __getitem__(self, index):
        ...

    def __eq__(self, other):
        ...


@runtime_checkable
class DataFrameLike(Protocol):
    def columns(self):
        ...

    def dtypes(self):
        ...

    def __getitem__(self, index):
        ...


@runtime_checkable
class Transformable(DictLike, DataFrameLike, Protocol):
    ...
