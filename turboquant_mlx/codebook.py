from __future__ import annotations

import numpy as np
import mlx.core as mx


def _require_integer_bits(bits: float | int) -> int:
    bits_float = float(bits)
    if not bits_float.is_integer():
        raise ValueError(
            "This implementation only supports integer bit-widths per TurboQuant instance. "
            "Use separate high/low-bit quantizers for the paper's 2.5/3.5-bit outlier scheme."
        )
    bits_int = int(bits_float)
    if bits_int < 0:
        raise ValueError("bits must be non-negative")
    return bits_int


def build_codebook(bits: int, head_dim: int, n_samples: int = 500_000, n_iter: int = 100):
    """Run Lloyd's algorithm on the exact coordinate law of a random point on S^(d-1)."""
    rng = np.random.default_rng(42)
    bits = _require_integer_bits(bits)
    z = rng.standard_normal(size=n_samples).astype(np.float32)
    w = rng.chisquare(df=max(head_dim - 1, 1), size=n_samples).astype(np.float32)
    samples = (z / np.sqrt(z * z + w)).astype(np.float32)
    k = 2 ** bits
    centroids = np.quantile(samples, np.linspace(0, 1, k)).astype(np.float32)
    for _ in range(n_iter):
        dists = np.abs(samples[:, None] - centroids[None, :])  # (N, k)
        assignments = dists.argmin(axis=1)
        new_centroids = np.array([
            samples[assignments == j].mean() if (assignments == j).any() else centroids[j]
            for j in range(k)
        ], dtype=np.float32)
        if np.allclose(centroids, new_centroids, atol=1e-7):
            break
        centroids = new_centroids
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids, boundaries


class Codebook:
    """Stores Lloyd's centroids for several bit-widths and quantizes/dequantizes MLX arrays."""

    def __init__(self, centroids_dict: dict, head_dim: int):
        self._centroids = {b: mx.array(c) for b, c in centroids_dict.items()}
        self.head_dim = head_dim

    @classmethod
    def build_and_save(cls, path: str, head_dim: int, bits_list: list) -> "Codebook":
        centroids_dict = {}
        save_dict = {}
        scalar_bits = {_require_integer_bits(bits) for bits in bits_list}
        for bits in sorted(scalar_bits):
            c, _ = build_codebook(bits, head_dim)
            centroids_dict[bits] = c
            save_dict[f"centroids_{bits}bit"] = c
        save_dict["head_dim"] = np.array(head_dim)
        np.savez(path, **save_dict)
        return cls(centroids_dict, head_dim)

    @classmethod
    def load(cls, path: str, head_dim: int | None = None) -> "Codebook":
        data = np.load(path)
        saved_head_dim = int(data["head_dim"]) if "head_dim" in data.files else None
        if head_dim is None:
            if saved_head_dim is None:
                raise ValueError("head_dim must be provided when the saved codebook has no head_dim metadata")
            head_dim = saved_head_dim
        elif saved_head_dim is not None and head_dim != saved_head_dim:
            raise ValueError(f"head_dim mismatch: expected {saved_head_dim}, got {head_dim}")
        centroids_dict = {}
        for key in data.files:
            if key.startswith("centroids_") and key.endswith("bit"):
                bits_str = key.split("_", 1)[1].replace("bit", "")
                bits = float(bits_str)
                b = int(bits) if bits.is_integer() else bits
                centroids_dict[b] = data[key]
        return cls(centroids_dict, head_dim)

    def quantize(self, x: mx.array, bits: float) -> mx.array:
        """Quantize x (..., head_dim) -> integer indices of same shape."""
        return self._quantize_scalar(x, _require_integer_bits(bits))

    def dequantize(self, indices: mx.array, bits: float) -> mx.array:
        """Dequantize integer indices -> float values."""
        return self._dequantize_scalar(indices, _require_integer_bits(bits))

    def _quantize_scalar(self, x: mx.array, bits: int) -> mx.array:
        centroids = self._centroids[bits]  # (2^bits,)
        dists = mx.abs(x[..., None] - centroids)  # (..., d, 2^bits)
        # Keep indices wide enough for larger codebooks; int8 corrupts bits >= 9.
        return mx.argmin(dists, axis=-1).astype(mx.int32)

    def _dequantize_scalar(self, indices: mx.array, bits: int) -> mx.array:
        centroids = self._centroids[bits]
        return centroids[indices.astype(mx.int32)]
