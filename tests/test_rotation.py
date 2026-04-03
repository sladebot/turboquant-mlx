import numpy as np
import pytest


def test_fast_rotation_roundtrip():
    from turboquant_mlx.rotation import Rotation

    rotation = Rotation(8, fast=True, seed=0)
    x = np.arange(16, dtype=np.float32).reshape(2, 8)

    restored = rotation.inverse(rotation.apply(x))

    np.testing.assert_allclose(restored, x, atol=1e-5)


def test_dense_rotation_roundtrip():
    from turboquant_mlx.rotation import Rotation

    rotation = Rotation(6, fast=False, seed=0)
    x = np.arange(12, dtype=np.float32).reshape(2, 6)

    restored = rotation.inverse(rotation.apply(x))

    np.testing.assert_allclose(restored, x, atol=1e-5)


def test_fast_rotation_rejects_non_power_of_two():
    from turboquant_mlx.rotation import Rotation

    with pytest.raises(ValueError, match="power of 2"):
        Rotation(6, fast=True)
