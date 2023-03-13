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

# pylint:disable=too-many-public-methods
from typing import Protocol, runtime_checkable


@runtime_checkable
class DictLike(Protocol):
    """
    These methods are present on plain Python dictionaries and also on DataFrames, which
    are conceptually a dictionary of columns/series. Both Python dictionaries and DataFrames
    therefore implement this Protocol, although neither sub-classes it. That means that
    `isinstance(obj, DictLike)` will return `True` at runtime if obj is a dictionary, a DataFrame,
    or any other type that implements the following methods.
    """

    def __iter__(self):
        return iter({})

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
    """
    These methods are defined by Pandas and cuDF series, and also by the array-wrapping
    `Column` class defined in `merlin.dag`. If we want to provide column-level transformations
    on data (e.g. to zero-copy share it across frameworks), the `Column` class would provide
    a potential place to do that, and this Protocol would allow us to build abstractions that
    make working with arrays and Series interchangeably possible.
    """

    def values(self):
        ...

    def dtype(self):
        ...

    def __getitem__(self, index):
        ...

    def __eq__(self, other):
        ...


@runtime_checkable
class Transformable(DictLike, Protocol):
    """
    In addition to the dictionary methods that are shared by dataframes, there are a few
    methods from dataframes that we use so frequently that it's easier to wrap a dictionary
    in a class and add them to the wrapper class than it would be to refactor the whole code
    base to do without them.
    """

    def columns(self):
        ...

    def dtypes(self):
        ...

    def __getitem__(self, index):
        ...


@runtime_checkable
class DataFrameLike(Transformable, Protocol):
    """
    This is the maximal set of methods shared by both Pandas dataframes and cuDF dataframes
    that aren't already part of the Transformable protocol. In theory, if there were another
    dataframe library that implemented the methods in this Protocol (e.g. Polars), we could
    use its dataframes in any place where we use the DataFrameLike type, but right now this
    protocol is only intended to match Pandas and cuDF dataframes.
    """

    def apply(self):
        ...

    def describe(self):
        ...

    def drop(self):
        ...

    def explode(self):
        ...

    def groupby(self):
        ...

    def head(self):
        ...

    def interpolate(self):
        ...

    def join(self):
        ...

    def max(self):
        ...

    def mean(self):
        ...

    def median(self):
        ...

    def pipe(self):
        ...

    def pivot(self):
        ...

    def product(self):
        ...

    def quantile(self):
        ...

    def rename(self):
        ...

    def replace(self):
        ...

    def sample(self):
        ...

    def shape(self):
        ...

    def shift(self):
        ...

    def std(self):
        ...

    def sum(self):
        ...

    def tail(self):
        ...

    def to_dict(self):
        ...

    def to_numpy(self):
        ...

    def transpose(self):
        ...

    def unstack(self):
        ...

    def var(self):
        ...
