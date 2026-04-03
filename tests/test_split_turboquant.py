import numpy as np


def _make_split_quantizer():
    from turboquant_mlx.codebook import Codebook
    from turboquant_mlx.polar_quant import PolarQuant, SplitQuantizationConfig, SplitTurboQuant
    from turboquant_mlx.qjl import QJL
    from turboquant_mlx.rotation import Rotation

    config = SplitQuantizationConfig(
        head_dim=8,
        outlier_indices=(1, 5),
        outlier_bits=3,
        regular_bits=2,
    )

    outlier_quantizer = PolarQuant(
        Rotation(2, fast=False, seed=1),
        Codebook(
            {
                2: np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32),
                3: np.linspace(-1.0, 1.0, 8, dtype=np.float32),
            },
            head_dim=2,
        ),
        qjl=QJL(2, seed=101),
    )
    regular_quantizer = PolarQuant(
        Rotation(6, fast=False, seed=2),
        Codebook(
            {
                1: np.array([-0.6, 0.6], dtype=np.float32),
                2: np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32),
            },
            head_dim=6,
        ),
        qjl=QJL(6, seed=102),
    )
    return SplitTurboQuant(config, outlier_quantizer, regular_quantizer)


def test_split_config_effective_bits_matches_weighted_average():
    from turboquant_mlx.polar_quant import SplitQuantizationConfig

    config = SplitQuantizationConfig(
        head_dim=8,
        outlier_indices=(1, 5),
        outlier_bits=3,
        regular_bits=2,
    )

    assert config.effective_bits == 2.25


def test_split_mse_uses_two_independent_quantizers():
    split_quantizer = _make_split_quantizer()
    x = np.arange(8, dtype=np.float32).reshape(1, 8)

    quantized = split_quantizer.quantize_mse(x)
    restored = split_quantizer.dequantize_mse(quantized)

    outlier = x[..., [1, 5]]
    regular = x[..., [0, 2, 3, 4, 6, 7]]
    expected_outlier = split_quantizer.outlier_quantizer.quantize_mse(outlier, bits=3)
    expected_regular = split_quantizer.regular_quantizer.quantize_mse(regular, bits=2)

    np.testing.assert_array_equal(quantized.outlier.indices, expected_outlier.indices)
    np.testing.assert_array_equal(quantized.regular.indices, expected_regular.indices)
    assert restored.shape == x.shape


def test_split_prod_delegates_to_each_sub_quantizer():
    split_quantizer = _make_split_quantizer()
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(1, 8)

    quantized = split_quantizer.quantize_prod(x)
    restored = split_quantizer.dequantize_prod(quantized)

    outlier = x[..., [1, 5]]
    regular = x[..., [0, 2, 3, 4, 6, 7]]
    expected_outlier = split_quantizer.outlier_quantizer.quantize_prod(outlier, bits=3)
    expected_regular = split_quantizer.regular_quantizer.quantize_prod(regular, bits=2)

    np.testing.assert_array_equal(quantized.outlier.indices, expected_outlier.indices)
    np.testing.assert_array_equal(quantized.regular.indices, expected_regular.indices)
    np.testing.assert_array_equal(quantized.outlier.qjl_signs, expected_outlier.qjl_signs)
    np.testing.assert_array_equal(quantized.regular.qjl_signs, expected_regular.qjl_signs)
    assert restored.shape == x.shape
