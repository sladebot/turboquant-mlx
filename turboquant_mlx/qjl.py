from __future__ import annotations

import mlx.core as mx
import numpy as np
import math


class QJL:
    """Quantized Johnson-Lindenstrauss sketch from Definition 1 in the paper.

    Qqjl(x) = sign(S x)
    Qqjl^{-1}(z) = sqrt(pi / 2) / d * S^T z
    """

    def __init__(self, head_dim: int, seed: int = 0):
        self.head_dim = head_dim
        rng = np.random.default_rng(seed)
        S = rng.standard_normal((head_dim, head_dim)).astype(np.float32)
        self._S = mx.array(S)
        self._ST = mx.array(S.T)
        self._scale = math.sqrt(math.pi / 2.0) / head_dim

    def quantize(self, x: mx.array) -> mx.array:
        """x: (..., d) -> sign(Sx) in {-1, +1}^d."""
        projected = x @ self._ST                                # (..., d)
        return mx.where(projected >= 0, 1, -1).astype(mx.int8)

    def dequantize(self, signs: mx.array, gamma: mx.array | float = 1.0) -> mx.array:
        """signs: (..., d) int8, gamma: (..., 1) -> gamma * sqrt(pi/2)/d * S^T signs."""
        s = signs.astype(mx.float32)
        g = gamma.astype(mx.float32) if hasattr(gamma, "astype") else gamma
        return self._scale * g * (s @ self._S)                  # (..., d)

    # Backwards-compatible aliases used by the pre-paper API.
    def compress(self, x: mx.array) -> tuple[mx.array, mx.array]:
        norm = mx.sqrt(mx.sum(x ** 2, axis=-1, keepdims=True)).astype(mx.float16)
        return self.quantize(x), norm

    def decompress(self, signs: mx.array, norm: mx.array) -> mx.array:
        return self.dequantize(signs, norm.astype(mx.float32))
