import math
import numpy as np


def test_qjl_matches_definition_1_formula():
    from turboquant_mlx.qjl import QJL

    qjl = QJL(4, seed=0)
    x = np.array([[1.0, -2.0, 0.5, 0.0]], dtype=np.float32)

    signs = qjl.quantize(x)
    decoded = qjl.dequantize(signs, np.array([[2.0]], dtype=np.float32))

    expected_signs = np.where(x @ np.array(qjl._ST) >= 0, 1, -1).astype(np.int8)
    expected = math.sqrt(math.pi / 2.0) / 4.0 * 2.0 * (expected_signs.astype(np.float32) @ np.array(qjl._S))

    np.testing.assert_array_equal(signs, expected_signs)
    np.testing.assert_allclose(decoded, expected, atol=1e-6)


def test_qjl_quantize_outputs_binary_signs_for_zero_vector():
    from turboquant_mlx.qjl import QJL

    qjl = QJL(8, seed=0)
    x = np.zeros((2, 8), dtype=np.float32)

    signs = qjl.quantize(x)
    restored = qjl.dequantize(signs, np.zeros((2, 1), dtype=np.float32))

    assert set(np.unique(signs).tolist()) == {1}
    np.testing.assert_allclose(restored, np.zeros((2, 8), dtype=np.float32))
