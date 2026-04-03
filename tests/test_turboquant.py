import numpy as np


def _make_quantizer(seed: int):
    from turboquant_mlx.codebook import Codebook
    from turboquant_mlx.polar_quant import PolarQuant
    from turboquant_mlx.qjl import QJL
    from turboquant_mlx.rotation import Rotation

    centroids = {
        0: np.array([0.0], dtype=np.float32),
        1: np.array([-0.6, 0.6], dtype=np.float32),
        2: np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32),
        3: np.linspace(-1.0, 1.0, 8, dtype=np.float32),
    }
    rotation = Rotation(8, fast=False, seed=seed)
    qjl = QJL(8, seed=seed + 1000)
    return PolarQuant(rotation, Codebook(centroids, head_dim=8), qjl=qjl)


def test_quantize_mse_stores_norms_and_rescales_dequantization():
    quantizer = _make_quantizer(seed=0)
    x = np.array([[3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    quantized = quantizer.quantize_mse(x, bits=2)
    restored = quantizer.dequantize_mse(quantized, bits=2)

    np.testing.assert_allclose(quantized.norms, np.array([[5.0]], dtype=np.float32))
    assert restored.shape == x.shape


def test_quantize_prod_matches_algorithm_2_structure():
    quantizer = _make_quantizer(seed=1)
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(1, 8)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)

    quantized = quantizer.quantize_prod(x, bits=3)
    mse_hat = quantizer._dequantize_mse_unit(quantized.indices, bits=2)
    residual = x - mse_hat

    np.testing.assert_allclose(quantized.residual_norms, np.linalg.norm(residual, axis=-1, keepdims=True), atol=1e-6)
    np.testing.assert_array_equal(quantized.qjl_signs, quantizer.qjl.quantize(residual))


def test_turboquantprod_is_empirically_unbiased_while_mse_is_biased():
    x = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    y = np.array([[0.25, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1, -0.35]], dtype=np.float32)
    target = float(np.sum(x * y))

    prod_estimates = []
    mse_estimates = []
    for seed in range(200):
        quantizer = _make_quantizer(seed)
        prod_hat = quantizer.dequantize_prod(quantizer.quantize_prod(x, bits=2), bits=2)
        mse_hat = quantizer.dequantize_mse(quantizer.quantize_mse(x, bits=1), bits=1)
        prod_estimates.append(float(np.sum(prod_hat * y)))
        mse_estimates.append(float(np.sum(mse_hat * y)))

    prod_mean = float(np.mean(prod_estimates))
    mse_mean = float(np.mean(mse_estimates))

    assert abs(prod_mean - target) < 0.05
    assert abs(mse_mean - target) > 0.05


def test_prod_one_bit_uses_zero_bit_mse_plus_qjl_residual():
    quantizer = _make_quantizer(seed=2)
    x = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    quantized = quantizer.quantize_prod(x, bits=1)

    assert np.max(np.array(quantized.indices)) == 0
    assert np.min(np.array(quantized.indices)) == 0
    assert set(np.unique(np.array(quantized.qjl_signs)).tolist()) <= {-1, 1}
