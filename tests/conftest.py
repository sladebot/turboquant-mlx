import sys
import types

import numpy as np


def _install_mlx_stub() -> None:
    mx = types.ModuleType("mlx.core")

    def array(value, dtype=None):
        return np.array(value, dtype=dtype)

    mx.array = array
    mx.abs = np.abs
    mx.mean = np.mean
    mx.concatenate = np.concatenate
    mx.argmin = np.argmin
    mx.zeros = np.zeros
    mx.stack = np.stack
    mx.sqrt = np.sqrt
    mx.sum = np.sum
    mx.sign = np.sign
    mx.where = np.where
    mx.max = np.max
    mx.float16 = np.float16
    mx.float32 = np.float32
    mx.int8 = np.int8
    mx.int32 = np.int32

    mlx = types.ModuleType("mlx")
    mlx.core = mx

    sys.modules["mlx"] = mlx
    sys.modules["mlx.core"] = mx


_install_mlx_stub()
