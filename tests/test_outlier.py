import numpy as np


def test_outlier_handler_counts_tokens_and_roundtrips():
    from turboquant_mlx.outlier import OutlierHandler

    handler = OutlierHandler(head_dim=8, threshold=2.0, warmup=4, max_outlier_channels=2, ema_decay=0.0)
    x = np.ones((5, 8), dtype=np.float32)
    x[:, 3] = 10.0

    handler.update(x)

    assert handler.warmed_up
    assert handler._token_count == 5

    main, outliers, indices = handler.split(x)

    assert indices.tolist() == [3]
    restored = handler.merge(main, outliers, indices)
    np.testing.assert_allclose(restored, x)


def test_outlier_handler_selects_largest_channels_when_capped():
    from turboquant_mlx.outlier import OutlierHandler

    handler = OutlierHandler(head_dim=8, threshold=1.1, warmup=1, max_outlier_channels=2, ema_decay=0.0)
    x = np.ones((1, 8), dtype=np.float32)
    x[0, 1] = 5.0
    x[0, 4] = 9.0
    x[0, 6] = 7.0

    handler.update(x)

    assert handler._outlier_indices.tolist() == [4, 6]


def test_outlier_handler_builds_split_config_from_selected_outliers():
    from turboquant_mlx.outlier import OutlierHandler

    handler = OutlierHandler(head_dim=128, threshold=2.0, warmup=1, max_outlier_channels=32, ema_decay=0.0)
    x = np.ones((1, 128), dtype=np.float32)
    x[0, :32] = 10.0

    handler.update(x)
    config = handler.build_split_config(outlier_bits=3, regular_bits=2)

    assert len(config.outlier_indices) == 32
    assert config.outlier_indices == tuple(range(32))
    # Arithmetic cross-check: 32 * 3 + 96 * 2 = 288 bits over 128 channels => 2.25 bits/channel.
    assert config.effective_bits == 2.25


def test_outlier_handler_can_build_split_quantizer():
    from turboquant_mlx.codebook import Codebook
    from turboquant_mlx.outlier import OutlierHandler
    from turboquant_mlx.polar_quant import PolarQuant, SplitTurboQuant
    from turboquant_mlx.qjl import QJL
    from turboquant_mlx.rotation import Rotation

    handler = OutlierHandler(head_dim=8, threshold=2.0, warmup=1, max_outlier_channels=2, ema_decay=0.0)
    x = np.ones((1, 8), dtype=np.float32)
    x[0, [1, 5]] = 10.0
    handler.update(x)

    outlier_quantizer = PolarQuant(
        Rotation(2, fast=False, seed=1),
        Codebook({3: np.linspace(-1.0, 1.0, 8, dtype=np.float32), 2: np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32)}, head_dim=2),
        qjl=QJL(2, seed=101),
    )
    regular_quantizer = PolarQuant(
        Rotation(6, fast=False, seed=2),
        Codebook({2: np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32), 1: np.array([-0.6, 0.6], dtype=np.float32)}, head_dim=6),
        qjl=QJL(6, seed=102),
    )

    split_quantizer = handler.build_split_quantizer(
        outlier_quantizer,
        regular_quantizer,
        outlier_bits=3,
        regular_bits=2,
    )

    assert isinstance(split_quantizer, SplitTurboQuant)
    assert split_quantizer.config.outlier_indices == (1, 5)
