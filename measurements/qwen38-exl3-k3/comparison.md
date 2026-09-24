# Qwen3.8 Flash Next: quantization, memory and cosine loss

All logit scores use the same BF16-source teacher fixture:
`<models>/kld-teacher/mlx-serve-bf16-16x512-raw`.
The primary score includes 7,186 positions through the teacher's first EOS,
inclusive, across 16 prompts. Each row compares all 248,320 raw logits with
float64 dot products and norms. Loss (%) is `100 * (1 - cosine similarity)`.

Inference measurements used engine base `8d04cedcc5b9d8039e0fd5cb8ffe13f8f1c66010`
plus the cosine/memory instrumentation in this change. Before publication,
upstream `53110f0` was fast-forwarded and the ReleaseFast build and full Zig
test suite passed again; the model measurements were not repeated on that update.

## Measured results

| Pack | Model active (GiB) | Peak active (GiB) | Logit cosine loss (%) ↓ | KLD (nats) ↓ |
| --- | ---: | ---: | ---: | ---: |
| EXL3 K3/MUL1 | 47.55 | 47.85 | 2.8644 | 0.092296 |
| EXL3 K4/MUL1 | 61.62 | 61.91 | **2.1824** | **0.059733** |
| Existing affine 4/g64, mixed trunk, BF16 n-grams | 67.93 | 68.49 | 2.5709 | 0.074932 |
| Fresh affine 3/g128, K3 donor trunk | 50.63 | 51.19 | 4.7235 | 0.167118 |

On this fixture, K3 is both smaller and closer to the teacher than plain
uncalibrated affine3/g128. K4 has the lowest measured logit loss and KLD.
These are corpus-specific distortion measurements, not capability-loss percentages.

**Memory is not total system RAM.** Model active is the MLX allocator's active
bytes immediately after loading. Peak active is measured during this short
teacher-forced workload, after resetting the peak counter. Neither includes
the allocator cache, general host allocations, or mmap/page-cache-backed
n-gram storage. Final allocator caches were 5.14 GiB (K3), 5.12 GiB (K4) and
7.78 GiB (affine4) and 7.75 GiB (affine3). A long-context workload needs additional KV/activation RAM.
The raw BF16 n-gram table is 95.37 GiB on disk and demand-paged, not fully resident.

KV is BF16; MTP drafting is inactive (the resident load still loads its head).
All candidates retain their current YaRN factor-4 configuration. K3/K4 share
their non-expert weights. The existing affine4 pack has a different mixed
non-expert trunk, so these are whole-pack comparisons, not an isolation of
routed-expert quantization alone. The affine4 pack's on-disk n-grams are 4-bit,
but its comparison explicitly reads the original BF16 n-grams via the override.

The teacher is the BF16 checkpoint through this engine, not an all-BF16 HF
forward; its decode attention requantization remains active. See `README.md`
for reference-control evidence and the historical oq8e comparison.

## Reproduce

Build with the pinned Zig:

```sh
zig build -Doptimize=ReleaseFast
```

Each command ran in a separate process, without concurrent GPU inference.

```sh
FIXTURE=<models>/kld-teacher/mlx-serve-bf16-16x512-raw
MODELS=<models>
OUT=measurements/qwen38-exl3-k3

./zig-out/bin/mlx-serve kld compare \
  --model "$MODELS/Qwen3.8-Flash-Next-EXL3-K3" --fixture "$FIXTURE" \
  --kv-quant off --label K3-cosine-memory-raw16x512 \
  --json "$OUT/logits-k3-memory.json"

./zig-out/bin/mlx-serve kld compare \
  --model "$MODELS/Qwen3.8-Flash-Next-EXL3-K4" --fixture "$FIXTURE" \
  --kv-quant off --label K4-cosine-raw16x512 \
  --json "$OUT/logits-k4.json"

MLX_SERVE_NGRAM_BF16_DIR="$MODELS/Qwen/Qwen3.8-Flash-Next" \
./zig-out/bin/mlx-serve kld compare \
  --model "$MODELS/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit" \
  --fixture "$FIXTURE" --kv-quant off \
  --label affine4-bf16ng-cosine-raw16x512 \
  --json "$OUT/logits-affine4.json"

./zig-out/bin/mlx-serve kld compare \
  --model "$MODELS/Qwen3.8-Flash-Next-Affine3-G128" \
  --fixture "$FIXTURE" --kv-quant off \
  --label affine3-g128-cosine-raw16x512 \
  --json "$OUT/logits-affine3.json"
```

The same stems with `.log` contain load and kernel-engagement evidence.
K3's repeat with memory reporting reproduced every per-prompt numerical result
from `logits.json`. Independent position-weighted aggregate checks passed for
all four JSON files. The full Zig test suite passed after adding memory reporting
and again after the final affine3 run. The 11 Python measurement/converter tests
passed (one existing NumPy overflow warning in the shared conversion helper).

## Configuration fingerprints

SHA-256 of `config.json`:

- K3: `3845514255e0c3aff7af1c6d236164a9c762242ab615c1f43416bf280362d2a7`
- K4: `ad52154df8c75b05a9731086027858559c4c4ceffffaf720650347b08d211e6a`
- Existing affine4: `56652acf3aa61a2ef4a029d0d13cbe665c410b90338a016381a2ce6c41c225d8`
- Fresh affine3: `7f53681be01fe806cfb9f5cfd782fa92c907ffeff8c9fc8eff85dee531b4c620`

All four tokenizer hashes match the teacher. Configuration hashes do not
identify the complete tensor payload.

## Fresh affine3/group128 conversion

Output: `<models>/Qwen3.8-Flash-Next-Affine3-G128`.
The converter reads original BF16 routed experts, not dequantized affine4
weights. This is **plain, uncalibrated MLX affine quantization**. All 48 trunk
layers and the MTP layer use 3-bit/group128; the K3 donor supplies unchanged
8-bit non-expert weights, configuration and raw BF16 n-grams.

```sh
PYTHONPATH=tests PYTHONUNBUFFERED=1 \
<sashimi>/.venv/bin/python \
tests/convert_qwen38_flash_next_affine_graft.py \
  --src <models>/Qwen/Qwen3.8-Flash-Next \
  --donor <models>/Qwen3.8-Flash-Next-EXL3-K3 \
  --dst <models>/Qwen3.8-Flash-Next-Affine3-G128 \
  --bits 3 --expert-gs 128 --batch-experts 64 --cpu-threads 4
```

The destination must not exist. Conversion used CPU-only bounded batches.
Unchanged donor files are hard-linked, never modified; the mixed MTP shard
is rewritten. The completed pack has 54 shards and 3,403 indexed tensors,
including 441 affine route tensors and 147 explicit 3/g128 route specifications.
The complete header audit validated gate/up weights `[512,640,240]`, down
weights `[512,2560,60]`, and their BF16 scales/biases. The live comparison
log confirms `bits=3 gs=128` at the routed kernels.

Indexed shards occupy 57,236,106,466 bytes including headers. Fresh routed payload is
50,095,718,400 bytes; new non-donor storage is approximately 50.19 GB, with
the remaining weights and the 102.40 GB n-gram table shared by hard link.
These disk sizes are not resident-memory measurements.

## Separate weight-reconstruction metric

K3's sampled routed-expert **weight** cosine loss is **0.9838%**, not 2.8644%.
That CPU-only measurement covers 576 matrices (four fixed expert IDs per layer,
48 layers, gate/up/down), about 944 million weights. It excludes MTP and
non-expert weights. See `weights.json` and `README.md` for sampling, decoder
and percentile details. Do not compare that number directly to logit loss.
