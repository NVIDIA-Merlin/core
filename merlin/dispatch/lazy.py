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
from functools import singledispatch
from inspect import isclass


class LazyDispatcher:
    """
    A wrapper around `functools.singledispatch` that allows lazy registration
    if the top-level module containing the types to be registered becomes
    available later.
    """

    def __init__(self, func):
        self.dispatcher = singledispatch(func)
        self.func = func
        self._lazy = {}

    def register(self, cls, func=None):
        """
        Registers a new implementation for the given *cls* on a *generic_func*.

        generic_func.register(cls, func) -> func

        Parameters
        ----------
        func : function, optional
            The registration function to be lazily executed if the class
            becomes available, by default None

        Returns
        -------
        function
            Version of func wrapped by `functools.singledispatch`
        """
        return self.dispatcher.register(cls, func=func)

    def register_lazy(self, module_name: str, func=None):
        """
        Registers a new lazy registration function for the given *module* on a *generic_func*.

        To use lazy registration, the default implementation of the generic function *must*
        raise NotImplementedError.

        Parameters
        ----------
        module_name : str
            Name of the top level module that will be imported in the registration func.
        func : function, optional
            The registration function to be lazily executed if the top level module
            becomes available, by default None

        Returns
        -------
        function
            Wrapped version of func
        """

        def wrapper(func):
            self._lazy[module_name] = func
            return func

        return wrapper(func) if func is not None else wrapper

    def __call__(self, *args_, **kwargs_):
        try:
            dispatch_type = self._dispatch_type(args_[0])
            return self._singledispatch(dispatch_type, *args_, **kwargs_)
        except NotImplementedError:
            try:
                module_name = dispatch_type.__module__.partition(".")[0]
                self._lazy[module_name]()
                self._lazy.pop(module_name, None)
            except (AttributeError, KeyError):
                self._raise_not_impl(self.func, args_[0])
            else:
                return self._singledispatch(dispatch_type, *args_, **kwargs_)

    def _singledispatch(self, obj_type, *args, **kwargs):
        return self.dispatcher.dispatch(obj_type)(*args, **kwargs)

    def _dispatch_type(self, arg):
        return arg if isclass(arg) else type(arg)

    def _raise_not_impl(self, func, arg):
        funcname = getattr(func, "__name__", "lazysingledispatch function")
        raise NotImplementedError(
            f"{funcname} doesn't have a registered implementation "
            f"for type of {arg} ({type(arg)})"
        )


def lazysingledispatch(func):
    """
    Single-dispatch generic function decorator with lazy registration.

    Transforms a function into a generic function, which can have different
    behaviours depending upon the type of its first argument. The decorated
    function acts as the default implementation, and additional
    implementations can be registered using the register() attribute of the
    generic function.

    Also allows lazy registration of types that may or may not be available
    using the `register_lazy()` method. To use lazy registration, the default
    implementation of the generic function *must* raise NotImplementedError.

    Parameters
    ----------
    func : function
        The function to be registered. This function must have at least one
        argument.

    Returns
    -------
    LazyDispatcher
        A wrapper around `functools.singledispatch` that allows lazy registration
    """
    return LazyDispatcher(func)
