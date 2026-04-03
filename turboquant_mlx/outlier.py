from __future__ import annotations

# turboquant/outlier.py
import mlx.core as mx
import numpy as np

class OutlierHandler:
    """Track per-channel magnitudes via EMA for mixed-precision TurboQuant.

    After warmup, channels with EMA magnitude > threshold × median are marked
    as outliers so they can be routed to a separate higher-bit TurboQuant
    instance, matching the paper's split-channel setup.
    """

    def __init__(self, head_dim: int, threshold: float = 3.0,
                 warmup: int = 32, max_outlier_channels: int = 8,
                 ema_decay: float = 0.9):
        self.head_dim = head_dim
        self.threshold = threshold
        self.warmup = warmup
        self.max_outlier_channels = max_outlier_channels
        self.ema_decay = ema_decay
        self._ema = np.zeros(head_dim, dtype=np.float32)
        self._token_count = 0
        self._outlier_indices = None  # fixed after warmup

    @property
    def warmed_up(self) -> bool:
        return self._token_count >= self.warmup

    def update(self, x: mx.array) -> None:
        """x: (..., head_dim). Updates EMA channel magnitudes."""
        mag = np.array(mx.mean(mx.abs(x.reshape(-1, self.head_dim)), axis=0).astype(mx.float32))
        self._ema = self.ema_decay * self._ema + (1 - self.ema_decay) * mag
        batch_tokens = int(np.prod(x.shape[:-1])) if len(x.shape) > 1 else 1
        self._token_count += batch_tokens
        if self._outlier_indices is None and self._token_count >= self.warmup:
            median = np.median(self._ema)
            outlier_mask = self._ema > self.threshold * median
            candidates = np.where(outlier_mask)[0]
            if len(candidates) > self.max_outlier_channels:
                order = np.argsort(self._ema[candidates])[::-1]
                candidates = candidates[order[:self.max_outlier_channels]]
                candidates = np.sort(candidates)
            indices = candidates
            self._outlier_indices = indices

    def split(self, x: mx.array) -> tuple:
        """Split x into (main_channels, outlier_values, outlier_indices).

        If not warmed up or no outliers, returns (x, empty, empty array).
        """
        if not self.warmed_up or self._outlier_indices is None or len(self._outlier_indices) == 0:
            return x, mx.zeros((*x.shape[:-1], 0)), np.array([], dtype=np.int32)

        idx = self._outlier_indices
        outlier_vals = x[..., idx.tolist()]
        all_idx = np.arange(self.head_dim)
        main_idx = np.delete(all_idx, idx)
        main = x[..., main_idx.tolist()]
        return main, outlier_vals, idx

    @property
    def outlier_indices(self) -> np.ndarray:
        """Return the fixed outlier set selected after warmup."""
        if self._outlier_indices is None:
            return np.array([], dtype=np.int32)
        return self._outlier_indices.astype(np.int32, copy=False)

    def build_split_config(self, outlier_bits: int, regular_bits: int):
        """Build the paper-style mixed-precision split config from selected outliers."""
        from turboquant_mlx.polar_quant import SplitQuantizationConfig

        return SplitQuantizationConfig(
            head_dim=self.head_dim,
            outlier_indices=tuple(int(i) for i in self.outlier_indices.tolist()),
            outlier_bits=outlier_bits,
            regular_bits=regular_bits,
        )

    def build_split_quantizer(
        self,
        outlier_quantizer,
        regular_quantizer,
        *,
        outlier_bits: int,
        regular_bits: int,
    ):
        """Construct a SplitTurboQuant wired to the selected outlier channels."""
        from turboquant_mlx.polar_quant import SplitTurboQuant

        config = self.build_split_config(
            outlier_bits=outlier_bits,
            regular_bits=regular_bits,
        )
        return SplitTurboQuant(config, outlier_quantizer, regular_quantizer)

    def merge(self, main: mx.array, outlier_values: mx.array,
              outlier_indices: np.ndarray) -> mx.array:
        """Reconstruct full head_dim tensor from split components."""
        if len(outlier_indices) == 0:
            return main
        out_np = np.zeros((*main.shape[:-1], self.head_dim), dtype=np.float32)
        all_idx = np.arange(self.head_dim)
        main_idx = np.delete(all_idx, outlier_indices)
        out_np[..., main_idx] = np.array(main.astype(mx.float32))
        out_np[..., outlier_indices] = np.array(outlier_values.astype(mx.float32))
        return mx.array(out_np)
