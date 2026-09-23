# Architecture: Qwen3.8-Flash-Next (`qwen4_exp`)

How the engine serves Qwen3.8-Flash-Next: the three blocks it adds around the qwen3_5 GDN+MoE trunk, which
module state is shared, where the oracle lives, and the facts a change to this arch must respect. Read this before
touching `forwardQwen4With`, `src/qwen4_exp.zig` or `src/hc_prefill.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-exl3-experts](engine-exl3-experts.md),
[engine-expert-streaming](engine-expert-streaming.md), [engine-mtp](engine-mtp.md),
[engine-qsa-long-context](engine-qsa-long-context.md), [perf-baselines](perf-baselines.md),
[quality-kld](quality-kld.md).

## The model

125B-A6B MoE + 51B n-gram table + 4B MTP head. Dispatch is on `config.json` `model_type`; a qwen4_exp checkpoint
is NOT a qwen3_5 pack: three blocks sit around the qwen3_5 GDN+MoE trunk. 48 layers, 512 routed experts, top-k 10,
hidden 2560, expert intermediate 640.

## Code map

| File | Role |
|---|---|
| `src/transformer.zig` | Flash Next trunk = `forwardQwen4With` (hyper-connections, PLE, QSA mask over `gatedFullAttnWith`); EXL3 MoE = `moeExl3` |
| `src/qwen4_exp.zig` | Host side: n-gram hash (splitmix multipliers, per-head primes, eos-segment shifts) + the mmapped `ngram_table.bin` row gather (2/3/4/5/6/8-bit or raw bf16) + `PrefetchPool`/`startWarm` |
| `src/hc_prefill.zig` | Fused hyper-connection norm/mix prefill kernels (chunk width as a scalar INPUT, never a template) |
| `src/mtp.zig` | The native MTP head (`Qwen4Mtp`), see [engine-mtp](engine-mtp.md) |

## Trunk rules

- **Trunk**: 4 hyper-connection streams (`hcRead`/`hcWrite`, norms folded by the converter), n-gram PLE at layer 1
  (host gather from `ngram_table.bin`, never resident), QSA sparse attention past 2048 tokens (`qsaMask`),
  `hyper_connection_mixer` replaces `model.norm`. The residual stream is bf16 like the checkpoint; the f32 fixture
  is the MATH oracle (`QWEN4_STREAM_F32=1`).
- **Module state is READ-ONLY**: text slots batch-decode (`forwardMoeBatchedDecode`), prefix cache ON. Vision
  (Qwen3-VL tower, `model.visual.` prefix, added at conversion) decodes serially and is excluded from
  streamed loads.
- **The deferred PLE leaf is filled before anything evaluates the build**: see [engine-mtp](engine-mtp.md#ple-defer).
  A host token read inside the graph build serialized the build with the GPU.
- **Decode kernels**: the fused hc read is LATENCY-bound (`MLX_SERVE_HC_FUSED=0`; `hcWrite` DEFERS into the next
  read); HC + GDN prefill fusions take the chunk WIDTH as a scalar input. Details in
  [engine-kernels](engine-kernels.md).
- **A GDN trunk's `KVCache.step` is 0 forever**: see [engine-kv-cache](engine-kv-cache.md#gdn).

## The n-gram table (PLE)

- `mx.quantize` packs DENSELY (element i at bit offset `i*bits`, straddling words at 3/5/6 bits; `dequantRow` tested
  at every width) or raw bf16 (bits-16 arm).
- A random read into a cold 32 GB mmap is a serial SSD fault, so `gather` rides a `PrefetchPool`
  (`QWEN4_PLE_PREFETCH=0` disables) and `startWarm` preads the table at load (`MLX_SERVE_NGRAM_WARM=0`).
- The n-gram hash's eos is the TEXT config's (`ngram_eos`).
- `MLX_SERVE_NGRAM_BF16_DIR=<hf checkpoint>` serves any pack with the ORIGINAL bf16 n-gram table, so `kld compare`
  isolates the PLE table's cost.

## Oracle and fixtures

- `tests/dump_qwen4_exp_fixtures.py` is the oracle dumper. HF `hidden_states[i]` is the INPUT of layer i: compare
  layer-i output with stream_{i+1}.
- The converters, allocators and imatrix drivers live in the private converter repo; the format they owe us is
  [pack-format](pack-format.md).
- **Ties**: a tiny MoE oracle ties everywhere and the tie RATE is scale-invariant (relu leaves exact-zero block
  scores). The fixture dumps the reference's OWN margins; `Qwen4Ties` acquits under 3% by those, never by our
  output. Selection coverage for a k < E fixture is the MTP head's one MoE layer (`--topk 2`, `route_gap`).

## Packs on this box

| pack | path | notes |
|---|---|---|
| MCG K3 w12 (served target) | `/Users/beam/llm/models/Qwen3.8-Flash-Next-EXL3-K3-w12-mcg-plugged` | our MCG experts plugged into turboderp's K3 dense |
| turboderp K3 / K4 | `/Users/beam/llm/models/Qwen3.8-Flash-Next-EXL3-K3`, `...-K4` | restacked from exllamav3, MUL1 w16 |
| affine 4/8 control | `/Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit` | the control column in every table |
| bf16 source | `/Users/beam/llm/models/Qwen/Qwen3.8-Flash-Next` | teacher + imatrix source; streams (335 GB) |

Bytes read per decoded token on the MCG K3 pack: trunk 4.01 GB (affine-8), lm_head 0.66, routed 0.91 = 5.58 GB,
a 107 tok/s ceiling at 600 GB/s against 61.8 measured (58%). Numbers and their sources: [perf-baselines](perf-baselines.md).
