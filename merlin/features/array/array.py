import logging
from functools import singledispatch
from typing import Any, Generic, Type, TypeVar, Union

_ArrayType = TypeVar("_ArrayType")
_FakeTypeDict = dict()


@singledispatch
def to_dlpack(input_array):
    raise NotImplementedError("Implement to_dlpack.")


@singledispatch
def from_dlpack(output: _ArrayType, capsule) -> _ArrayType:
    raise NotImplementedError("Implement from_dlpack.")


@singledispatch
def to_array(input_array):
    raise NotImplementedError("Implement to_array.")


@singledispatch
def from_array(output: _ArrayType, array) -> _ArrayType:
    raise NotImplementedError("Implement from_array.")


@singledispatch
def to_cuda_array(input_array):
    raise NotImplementedError("Implement to_cuda_array.")


@singledispatch
def from_cuda_array(output: _ArrayType, array) -> _ArrayType:
    raise NotImplementedError("Implement from_cuda_array.")


def register_framework(array_type, array):
    pass


def check_tf():
    try:
        import tensorflow as tf

        register_framework(tf.Tensor, tf.constant(0.0))

        _FakeTypeDict[tf.Tensor] = tf.constant(0.0)

        @to_dlpack.register
        def tf_to_dlpack(input_array: tf.Tensor):
            logging.debug(f"Converting {input_array} to DLPack")
            return tf.experimental.dlpack.to_dlpack(input_array)

        @from_dlpack.register
        def tf_from_dlpack(output: tf.Tensor, capsule) -> tf.Tensor:
            logging.debug(f"Converting {capsule} to tf.Tensor")
            return tf.experimental.dlpack.from_dlpack(capsule)

        @to_array.register
        def tf_to_array(input_array: tf.Tensor):
            logging.debug(f"Converting {input_array} to np.ndarray")
            return input_array.numpy()

        @from_array.register
        def tf_from_array(output: tf.Tensor, array) -> tf.Tensor:
            logging.debug(f"Converting {array} to tf.Tensor")
            return tf.convert_to_tensor(array)

        return tf.Tensor
    except ImportError:
        pass


def check_cudf():
    try:
        import cudf

        _FakeTypeDict[cudf.Series] = cudf.Series()

        @to_dlpack.register
        def cudf_to_dlpack(input_array: cudf.Series):
            logging.debug(f"Converting {input_array} to DLPack")
            return input_array.to_dlpack()

        @from_dlpack.register
        def cudf_from_dlpack(output: cudf.Series, capsule) -> cudf.Series:
            logging.debug(f"Converting {capsule} to cudf.Series")
            return cudf.io.from_dlpack(capsule)

        @to_array.register
        def cudf_to_array(input_array: cudf.Series):
            logging.debug(f"Converting {input_array} to np.ndarray")
            return input_array.to_numpy()

        @from_array.register
        @from_cuda_array.register
        def cudf_from_array(output: cudf.Series, array) -> cudf.Series:
            logging.debug(f"Converting {array} to cudf.Series")
            return cudf.Series(array)

        return cudf.Series
    except ImportError:
        pass


def check_numpy():
    try:
        import numpy as np

        _FakeTypeDict[np.ndarray] = np.ndarray(shape=(1,))

        @from_dlpack.register
        def np_from_dlpack(capsule, output: np.ndarray) -> np.ndarray:
            try:
                return np._from_dlpack(capsule)
            except AttributeError as exc:
                raise NotImplementedError(
                    "NumPy does not implement the DLPack Standard until version 1.22.0, "
                    f"currently running {np.__version__}"
                ) from exc

        @to_array.register
        def np_to_array(input_array: np.ndarray):
            return input_array

        @from_array.register
        @from_cuda_array.register
        def np_from_array(output: np.ndarray, array) -> np.ndarray:
            return np.array(array)

        return np.ndarray
    except ImportError:
        pass


def check_cupy():
    try:
        import cupy as cp

        _FakeTypeDict[cp.ndarray] = cp.ndarray(shape=(1,))

        @to_dlpack.register
        def cupy_to_dlpack(input_array: cp.ndarray):
            logging.debug(f"Converting {input_array} to DLPack")
            try:
                return input_array.to_dlpack()
            except AttributeError:
                return input_array.toDlpack()

        @from_dlpack.register
        def cupy_from_dlpack(output: cp.ndarray, capsule) -> cp.ndarray:
            logging.debug(f"Converting {capsule} to cp.ndarray")
            try:
                return cp.from_dlpack(capsule)
            except AttributeError:
                return cp.fromDlpack(capsule)

        @to_array.register
        def cupy_to_array(input_array: cp.ndarray):
            logging.debug(f"Converting {input_array} to np.ndarray")
            return cp.asnumpy(input_array)

        @from_array.register
        @from_cuda_array.register
        def cupy_from_array(output: cp.ndarray, array) -> cp.ndarray:
            logging.debug(f"Converting {array} to cp.ndarray")
            return cp.asarray(array)

        @to_cuda_array.register
        def cudf_to_cuda_array(input_array: cp.ndarray):
            logging.debug(f"Converting {input_array} to cp.ndarray")
            return input_array

        return cp.ndarray
    except ImportError:
        pass


def find_datatypes():
    types = []
    _tf = check_tf()
    _cudf = check_cudf()
    _numpy = check_numpy()
    _cupy = check_cupy()

    if _tf:
        types.append(_tf)

    if _cudf:
        types.append(_cudf)

    if _numpy:
        types.append(_numpy)

    if _cupy:
        types.append(_cupy)

    return tuple(types)


ArrayType = Union[find_datatypes()]
ToType = TypeVar("ToType")


class MerlinArray:
    def __init__(self, arr: ArrayType):
        self.array = arr

    def to(self, to: Type[ToType]) -> ToType:
        return convert(self.array, to)


def convert(input: Any, to: Type[ToType]) -> ToType:
    if isinstance(input, to):
        return input

    _to_instance = _FakeTypeDict[to]

    # 1. Try through cuda-array
    try:
        return from_cuda_array(_to_instance, to_cuda_array(input))
    except Exception:
        pass

    # 2. Try to DLPack
    try:
        return from_dlpack(_to_instance, to_dlpack(input))
    except Exception:
        pass

    # 3. Try through array
    try:
        return from_array(_to_instance, to_array(input))
    except Exception:
        pass

    # TODO: Check step here

    raise TypeError(
        f"Can't create {input} array from type {to}, "
        "which doesn't support any of the available conversion interfaces."
    )
