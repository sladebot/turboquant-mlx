# turboquant-mlx

`turboquant-mlx` packages a small MLX-oriented reference implementation of the
TurboQuant building blocks described in arXiv:2504.19874:

- orthogonal rotations
- scalar codebooks
- polar quantization for keys and values
- a 1-bit QJL sketch
- outlier channel handling

## Installation

```bash
python -m pip install turboquant-mlx
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Runtime requirements

The package depends on `mlx` and therefore targets Apple Silicon macOS
environments where MLX is supported. The test suite uses a lightweight stub for
`mlx.core` so package logic can still be validated in environments where the
Metal runtime is unavailable.

## Package contents

- `Rotation`: fast Walsh-Hadamard or dense QR rotations
- `Codebook`: scalar quantization codebooks and persistence helpers
- `PolarQuant`: key/value encode-decode helpers
- `SplitTurboQuant`: paper-style mixed-precision split quantization for 2.5/3.5-bit setups
- `QJL`: 1-bit quantized Johnson-Lindenstrauss sketch
- `OutlierHandler`: EMA-based outlier channel selection for routing channels into the higher-bit split quantizer
