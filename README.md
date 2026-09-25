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

## Recommended launch

The model's own MTP draft head and the 8-bit KV cache are on by default.

**64 GB Mac, Sushi-3bpw**

```bash
hf download beamster/Qwen3.8-Flash-Next-Sushi-3bpw --local-dir ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-3bpw
sudo sysctl iogpu.wired_limit_mb=58000   # let the GPU use 58000 MB of the 64 GB, until the next reboot
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-3bpw \
  --kv-quant 8 --mtp-head-kv-quant --prefill-chunk 2048 --prefix-cache-disk 10GB
```

**96 GB+ Mac, Sushi-4bpw**

```bash
hf download beamster/Qwen3.8-Flash-Next-Sushi-4bpw --local-dir ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-4bpw
sudo sysctl iogpu.wired_limit_mb=88000   # let the GPU use 88000 MB of the 96 GB, until the next reboot
./sushi-macos-arm64/sushi serve --model ~/.sushi/models/Qwen3.8-Flash-Next-Sushi-4bpw --mtp-head-kv-quant
```

- `--mtp-head-kv-quant` stores the MTP head's own KV at 8 bits too.
- `--prefill-chunk 2048` caps the prompt tokens forwarded per step, which keeps the prefill peak small.
- `--prefix-cache-disk 10GB` keeps seen prompt prefixes on the SSD, so a repeated prompt skips its prefill, across
  restarts too.
- `--mtp-typical 0.2` makes sampled decoding 15-20% faster (Sushi-4bpw, temperature 1.0) with no measurable change in
  the NLL of the generated text; greedy output is unchanged.

## Memory

GPU memory in GiB (what `sushi run` reports); the n-gram table stays on the SSD.

| | Sushi-3bpw | Sushi-4bpw |
|---|---|---|
| model weights | 47.51 | 61.58 |
| MTP head | 0.98 | 1.27 |
| vision tower | 0.84 | 0.84 |
| **weights loaded** | **49.33** | **63.68** |
| KV cache, 256k tokens | 4.06 | 4.06 |
| KV cache, 512k tokens | 8.12 | 8.12 |
| KV cache, 1M tokens | 16.25 | 16.25 |
| **total at 256k / 512k / 1M** | **53.4 / 57.5 / 65.6** | **67.7 / 71.8 / 79.9** |

The KV cache is for one request at 8 bits with MTP on and `--mtp-head-kv-quant`: 16,640 bytes per token of context.
Each concurrent request adds its own. Leave room for the hot prefix cache (`--prefix-cache-mem`, 2 GiB by default) and
the prefill buffers.

## Quality

<p align="center"><img src="docs/assets/kld-chart.png" alt="KLD vs size" width="100%"></p>

KLD against the bf16 model: 16 prompts x 512 tokens scored to the first EOS, kv8, every pack run by the same sushi
build. The light rings are the sushi packs with a 4-bit n-gram table (Sushi-3bpw ships that table; either table works
with either pack). Numbers: [docs/quality-kld.md](docs/quality-kld.md).

## Speed

Sushi-3bpw on an M5 Max 128 GB: `--ctx-size 1048576 --kv-quant 8 --mtp`, llmprobe `--bench-only`, quiet box.

<p align="center"><img src="docs/assets/perf-sushi3bpw-1m.png" alt="decode, first token and prefill vs context" width="100%"></p>

| context | decode tok/s | prefill tok/s | first token | tokens per step |
|---|---|---|---|---|
| 4k | 96.1 | 1702 | 2.5 s | 3.37 |
| 8k | 92.5 | 1940 | 4.2 s | 3.31 |
| 16k | 93.2 | 1949 | 8.4 s | 3.20 |
| 33k | 96.2 | 1961 | 16.8 s | 3.00 |
| 66k | 80.7 | 1945 | 33.7 s | 2.56 |
| 131k | 80.9 | 1900 | 69.0 s | 2.63 |
| 262k | 73.4 | 1826 | 143.7 s | 3.20 |
| 524k | 55.1 | 1686 | 311.1 s | 3.37 |
| 1004k | 56.6 | 1465 | 685.2 s | 3.31 |

Smaller Macs have less memory bandwidth, so expect lower numbers. More in [docs/perf-baselines.md](docs/perf-baselines.md).
