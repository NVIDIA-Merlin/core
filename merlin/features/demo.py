import logging

import cudf
import cupy as cp
import numpy as np
import tensorflow as tf

from merlin.features.array.array import convert

logging.getLogger().setLevel(logging.DEBUG)


if __name__ == "__main__":
    _from: tf.Tensor = tf.constant((1.0, 2.0))

    series: cudf.Series = convert(_from, cudf.Series)
    np_array: np.ndarray = convert(_from, np.ndarray)
    cp_array: cp.ndarray = convert(_from, cp.ndarray)
    tf_tensor: tf.Tensor = convert(_from, tf.Tensor)
