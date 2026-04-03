from __future__ import annotations

# turboquant/polar_quant.py
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from turboquant_mlx.codebook import Codebook
from turboquant_mlx.qjl import QJL
from turboquant_mlx.rotation import Rotation


@dataclass
class MSEQuantizedVector:
    indices: mx.array
    norms: mx.array


@dataclass
class ProductQuantizedVector:
    indices: mx.array
    qjl_signs: mx.array
    residual_norms: mx.array
    norms: mx.array


@dataclass(frozen=True)
class SplitQuantizationConfig:
    head_dim: int
    outlier_indices: tuple[int, ...]
    outlier_bits: int
    regular_bits: int

    @property
    def regular_indices(self) -> tuple[int, ...]:
        all_idx = tuple(range(self.head_dim))
        return tuple(i for i in all_idx if i not in set(self.outlier_indices))

    @property
    def effective_bits(self) -> float:
        n_out = len(self.outlier_indices)
        n_reg = self.head_dim - n_out
        return (n_out * self.outlier_bits + n_reg * self.regular_bits) / self.head_dim


@dataclass
class SplitMSEQuantizedVector:
    outlier: MSEQuantizedVector
    regular: MSEQuantizedVector


@dataclass
class SplitProductQuantizedVector:
    outlier: ProductQuantizedVector
    regular: ProductQuantizedVector


class PolarQuant:
    """Paper-aligned TurboQuant implementation.

    `quantize_mse` / `dequantize_mse` implement Algorithm 1.
    `quantize_prod` / `dequantize_prod` implement Algorithm 2.
    """

    def __init__(self, rotation: Rotation, codebook: Codebook, qjl: QJL | None = None):
        self.rotation = rotation
        self.codebook = codebook
        self.qjl = qjl if qjl is not None else QJL(rotation.head_dim)

    def _normalize(self, x: mx.array) -> tuple[mx.array, mx.array]:
        norms = mx.sqrt(mx.sum(x ** 2, axis=-1, keepdims=True))
        safe_norms = mx.where(norms > 0, norms, 1.0)
        return x / safe_norms, norms

    def _quantize_mse_unit(self, x_unit: mx.array, bits: int) -> mx.array:
        y = self.rotation.apply(x_unit)
        return self.codebook.quantize(y, bits)

    def _dequantize_mse_unit(self, indices: mx.array, bits: int) -> mx.array:
        y_hat = self.codebook.dequantize(indices, bits)
        return self.rotation.inverse(y_hat)

    def quantize_mse(self, x: mx.array, bits: int) -> MSEQuantizedVector:
        """Algorithm 1 with stored L2 norms for non-unit inputs."""
        x_unit, norms = self._normalize(x)
        indices = self._quantize_mse_unit(x_unit, bits)
        return MSEQuantizedVector(indices=indices, norms=norms.astype(mx.float32))

    def dequantize_mse(self, quantized: MSEQuantizedVector, bits: int) -> mx.array:
        x_unit_hat = self._dequantize_mse_unit(quantized.indices, bits)
        return x_unit_hat * quantized.norms

    def quantize_prod(self, x: mx.array, bits: int) -> ProductQuantizedVector:
        """Algorithm 2 using TurboQuant_mse(b - 1) plus QJL on the residual."""
        if bits < 1:
            raise ValueError("inner-product TurboQuant requires bits >= 1")
        x_unit, norms = self._normalize(x)
        indices = self._quantize_mse_unit(x_unit, bits - 1)
        x_mse_hat = self._dequantize_mse_unit(indices, bits - 1)
        residual = x_unit - x_mse_hat
        residual_norms = mx.sqrt(mx.sum(residual ** 2, axis=-1, keepdims=True)).astype(mx.float32)
        qjl_signs = self.qjl.quantize(residual)
        return ProductQuantizedVector(
            indices=indices,
            qjl_signs=qjl_signs,
            residual_norms=residual_norms,
            norms=norms.astype(mx.float32),
        )

    def dequantize_prod(self, quantized: ProductQuantizedVector, bits: int) -> mx.array:
        x_mse_hat = self._dequantize_mse_unit(quantized.indices, bits - 1)
        x_qjl_hat = self.qjl.dequantize(quantized.qjl_signs, quantized.residual_norms)
        return (x_mse_hat + x_qjl_hat) * quantized.norms

    # Compatibility wrappers for the pre-paper API names.
    def encode_v(self, v: mx.array, bits: int) -> MSEQuantizedVector:
        return self.quantize_mse(v, bits)

    def decode_v(self, quantized: MSEQuantizedVector, bits: int) -> mx.array:
        return self.dequantize_mse(quantized, bits)

    def encode_k(self, k: mx.array, bits: int) -> ProductQuantizedVector:
        return self.quantize_prod(k, bits)

    def decode_k(self, quantized: ProductQuantizedVector, bits: int) -> mx.array:
        return self.dequantize_prod(quantized, bits)


class SplitTurboQuant:
    """Two independent TurboQuant instances for outlier/non-outlier channel groups.

    This is the composition used in Section 4 for non-integer effective bit-widths such as 2.5.
    """

    def __init__(
        self,
        config: SplitQuantizationConfig,
        outlier_quantizer: PolarQuant,
        regular_quantizer: PolarQuant,
    ):
        self.config = config
        self.outlier_quantizer = outlier_quantizer
        self.regular_quantizer = regular_quantizer
        self._outlier_idx = tuple(config.outlier_indices)
        self._regular_idx = tuple(config.regular_indices)
        if len(set(self._outlier_idx)) != len(self._outlier_idx):
            raise ValueError("outlier_indices must be unique")
        if len(self._outlier_idx) + len(self._regular_idx) != config.head_dim:
            raise ValueError("outlier_indices must partition the full head dimension")
        if outlier_quantizer.rotation.head_dim != len(self._outlier_idx):
            raise ValueError("outlier quantizer head_dim must match the number of outlier channels")
        if regular_quantizer.rotation.head_dim != len(self._regular_idx):
            raise ValueError("regular quantizer head_dim must match the number of regular channels")

    def _split(self, x: mx.array) -> tuple[mx.array, mx.array]:
        outlier = x[..., list(self._outlier_idx)]
        regular = x[..., list(self._regular_idx)]
        return outlier, regular

    def _merge(self, outlier: mx.array, regular: mx.array) -> mx.array:
        out_shape = outlier.shape[:-1] + (self.config.head_dim,)
        merged = np.zeros(out_shape, dtype=np.array(outlier).dtype)
        merged[..., list(self._outlier_idx)] = np.array(outlier)
        merged[..., list(self._regular_idx)] = np.array(regular)
        return mx.array(merged)

    def quantize_mse(self, x: mx.array) -> SplitMSEQuantizedVector:
        outlier, regular = self._split(x)
        return SplitMSEQuantizedVector(
            outlier=self.outlier_quantizer.quantize_mse(outlier, self.config.outlier_bits),
            regular=self.regular_quantizer.quantize_mse(regular, self.config.regular_bits),
        )

    def dequantize_mse(self, quantized: SplitMSEQuantizedVector) -> mx.array:
        outlier = self.outlier_quantizer.dequantize_mse(quantized.outlier, self.config.outlier_bits)
        regular = self.regular_quantizer.dequantize_mse(quantized.regular, self.config.regular_bits)
        return self._merge(outlier, regular)

    def quantize_prod(self, x: mx.array) -> SplitProductQuantizedVector:
        outlier, regular = self._split(x)
        return SplitProductQuantizedVector(
            outlier=self.outlier_quantizer.quantize_prod(outlier, self.config.outlier_bits),
            regular=self.regular_quantizer.quantize_prod(regular, self.config.regular_bits),
        )

    def dequantize_prod(self, quantized: SplitProductQuantizedVector) -> mx.array:
        outlier = self.outlier_quantizer.dequantize_prod(quantized.outlier, self.config.outlier_bits)
        regular = self.regular_quantizer.dequantize_prod(quantized.regular, self.config.regular_bits)
        return self._merge(outlier, regular)
