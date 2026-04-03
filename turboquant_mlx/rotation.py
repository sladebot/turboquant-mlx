from __future__ import annotations

import mlx.core as mx
import numpy as np


class Rotation:
    """Random rotation used by Algorithm 1.

    The paper uses a dense QR rotation. A fast Walsh-Hadamard variant is kept as
    an opt-in extension, but is not the default paper path.
    """

    def __init__(self, head_dim: int, fast: bool = False, seed: int = 42):
        self.head_dim = head_dim
        self.fast = fast
        rng = np.random.default_rng(seed)

        if fast:
            if head_dim <= 0 or head_dim & (head_dim - 1):
                raise ValueError("fast rotation requires head_dim to be a positive power of 2")
            signs = rng.choice([-1.0, 1.0], size=head_dim).astype(np.float32)
            self._signs = mx.array(signs)
            self._matrix = None
        else:
            G = rng.standard_normal((head_dim, head_dim)).astype(np.float32)
            Pi, _ = np.linalg.qr(G)
            self._matrix = mx.array(Pi)
            self._signs = None

    def apply(self, x: mx.array) -> mx.array:
        """Rotate x: (..., head_dim) -> (..., head_dim)."""
        if self.fast:
            return self._fwht(x * self._signs)
        return x @ self._matrix.T

    def inverse(self, x: mx.array) -> mx.array:
        """Inverse rotation: (..., head_dim) -> (..., head_dim)."""
        if self.fast:
            return self._fwht(x) * self._signs
        return x @ self._matrix

    def _fwht(self, x: mx.array) -> mx.array:
        """Iterative Walsh-Hadamard Transform (head_dim must be power of 2)."""
        n = x.shape[-1]
        result = x
        h = 1
        while h < n:
            shape = result.shape[:-1] + (n // (2 * h), 2, h)
            result = result.reshape(shape)
            a = result[..., 0, :]
            b = result[..., 1, :]
            result = mx.stack([a + b, a - b], axis=-2)
            result = result.reshape(result.shape[:-3] + (n,))
            h *= 2
        return result / mx.sqrt(mx.array(float(n), dtype=result.dtype))
