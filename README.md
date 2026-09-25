<p align="center"><img src="docs/assets/sushi-logo.png" alt="sushi" width="256"></p>

# SUSHI

A detached fork from ddalcu's MLX-serve masterpiece, focus only to support selected models in Apple Silicon using custom sushi quant. Sushi uses both EXL3 and affine mixed format tailored for M5+ Max class, other chips can still run well.

## Model support list

* Qwen3.8-Flash-Next-sushi-3bpw (Require 64GB+)
* Qwen3.8-Flash-Next-sushi-4bpw (Require 96GB+)

## Streaming

SSD expert streaming serves the bf16 Qwen3.8-Flash-Next checkpoint.

## Install

```bash
curl -L https://github.com/beamivalice/sushi/releases/latest/download/sushi-bin-macos-arm64.tar.gz | tar xz
./sushi-macos-arm64/sushi --version
```

The server listens on `127.0.0.1:12345` by default.
