import numpy as np
import pytest


def test_build_and_load_support_zero_bit_codebook(tmp_path):
    from turboquant_mlx.codebook import Codebook

    path = tmp_path / "codebook.npz"
    codebook = Codebook.build_and_save(str(path), head_dim=128, bits_list=[0, 1, 2])

    assert sorted(codebook._centroids) == [0, 1, 2]

    loaded = Codebook.load(str(path))

    assert loaded.head_dim == 128
    assert sorted(loaded._centroids) == [0, 1, 2]


def test_load_validates_saved_head_dim(tmp_path):
    from turboquant_mlx.codebook import Codebook

    path = tmp_path / "codebook.npz"
    Codebook.build_and_save(str(path), head_dim=64, bits_list=[0, 1])

    with pytest.raises(ValueError, match="head_dim mismatch"):
        Codebook.load(str(path), head_dim=128)


def test_zero_bit_quantization_uses_single_centroid():
    from turboquant_mlx.codebook import Codebook

    codebook = Codebook({0: np.array([0.0], dtype=np.float32)}, head_dim=4)
    x = np.array([[0.2, -0.1, 0.4, -0.3]], dtype=np.float32)

    indices = codebook.quantize(x, 0)
    decoded = codebook.dequantize(indices, 0)

    np.testing.assert_array_equal(indices, np.zeros_like(x, dtype=np.int8))
    np.testing.assert_allclose(decoded, np.zeros_like(x))


def test_fractional_bits_raise_instead_of_silent_truncation(tmp_path):
    from turboquant_mlx.codebook import Codebook

    path = tmp_path / "codebook.npz"

    with pytest.raises(ValueError, match="only supports integer bit-widths"):
        Codebook.build_and_save(str(path), head_dim=128, bits_list=[2.5])

    codebook = Codebook({2: np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32)}, head_dim=4)
    x = np.zeros((1, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="only supports integer bit-widths"):
        codebook.quantize(x, 2.5)


def test_large_codebooks_preserve_indices_beyond_int8_range():
    from turboquant_mlx.codebook import Codebook

    codebook = Codebook({9: np.arange(512, dtype=np.float32)}, head_dim=1)
    x = np.array([[128.0], [255.0], [256.0], [511.0]], dtype=np.float32)

    indices = codebook.quantize(x, 9)
    decoded = codebook.dequantize(indices, 9)

    np.testing.assert_array_equal(indices[:, 0], np.array([128, 255, 256, 511], dtype=np.int32))
    np.testing.assert_allclose(decoded[:, 0], x[:, 0])


def test_built_codebook_matches_paper_low_bit_centroids():
    from turboquant_mlx.codebook import build_codebook

    head_dim = 1024
    c1, _ = build_codebook(1, head_dim, n_samples=50_000, n_iter=60)
    c2, _ = build_codebook(2, head_dim, n_samples=50_000, n_iter=60)

    expected_b1 = np.array(
        [-np.sqrt(2.0 / np.pi) / np.sqrt(head_dim), np.sqrt(2.0 / np.pi) / np.sqrt(head_dim)],
        dtype=np.float32,
    )
    expected_b2 = np.array(
        [-1.51 / np.sqrt(head_dim), -0.453 / np.sqrt(head_dim), 0.453 / np.sqrt(head_dim), 1.51 / np.sqrt(head_dim)],
        dtype=np.float32,
    )

    np.testing.assert_allclose(c1, expected_b1, atol=2e-3)
    np.testing.assert_allclose(c2, expected_b2, atol=4e-3)
