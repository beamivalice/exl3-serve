<p align="center"><img src="docs/assets/sushi-logo.png" alt="sushi" width="256"></p>

# sushi

A fork from ddalcu's MLX-serve, focus only to support selected EXL3 models in Apple Silicon.

## Model support list

* Qwen3.8-Flash-Next (EXL3 packs, loaded resident)

Any other `model_type` or checkpoint format is refused at load by name.

## Streaming

SSD expert streaming serves the bf16 Qwen3.8-Flash-Next checkpoint. EXL3 packs do not stream: they load resident.

## Install

```bash
curl -L https://github.com/beamivalice/sushi/releases/latest/download/sushi-bin-macos-arm64.tar.gz | tar xz
./sushi-macos-arm64/sushi --version
```

The binary is ad-hoc signed, not notarized. A copy downloaded with a browser is quarantined by macOS; clear it with
`xattr -dr com.apple.quarantine sushi-macos-arm64`. The server listens on `127.0.0.1:12345` by default.
