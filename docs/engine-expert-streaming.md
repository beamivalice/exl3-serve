# Engine: expert streaming (`--ssd-budget-gb` / `--expert-cache-gb` / per-model `ssd_budget_gb`)

How a checkpoint whose routed experts do not fit in memory is served from SSD: the budget ledger, the per-layer LRU,
the zero-copy slab I/O and the correctness bars. Read this before touching `src/expert_stream.zig` or
`src/expert_io.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-exl3-experts](engine-exl3-experts.md),
[engine-memory-admission](engine-memory-admission.md), [arch-qwen4exp](arch-qwen4exp.md),
[arch-mimo-v2](arch-mimo-v2.md), [server-lifecycle](server-lifecycle.md#settings).

## Code map

| File | Role |
|---|---|
| `src/expert_stream.zig` | `ExpertStore` spans, per-layer group-exact LRU + union bridge, zero-copy slabs, `BudgetLedger` (`--ssd-budget-gb`), MTP refusal |
| `src/expert_io.zig` | SSD→Metal I/O: F_NOCACHE positioned-read `FillPool`, `PageSlab` epoch leases, verified zero-copy `importSlab` |
| `src/expert_bf16_kernels.zig` | bf16 selected-expert kernels over a slab |
| `src/imatrix.zig` | imatrix capture on the streamed forward |

## What streams

Any qwen4_exp checkpoint whose routed experts are leading-index banks streams: the HF fused bf16 layout
(`mlp.experts.gate_up_proj` `[512,1280,2560]` + `down_proj` `[512,2560,640]`, 335 GB total, `streaming_required`),
or the MLX split layout (`switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`). An EXL3 pack serves resident
only: `ModelConfig.streamsExperts` is false for it, so an SSD budget or expert cache asked of one is ignored with the
non-streaming warning. MiMo's original MXFP4 checkpoint streams too ([arch-mimo-v2](arch-mimo-v2.md)). Trunk + MTP resident; routed experts come from SSD through
zero-copy slabs. With no budget a pack loads resident as before.

## Budget

- `expert_stream.budgetLedger`, one `[expert-stream] ssd budget` boot line. `--ssd-budget-gb N` is a TOTAL resident
  target of N GiB = trunk + MTP + the 512-expert union workspace + selected slab + bounce; the remainder is a uniform
  per-layer LRU. `--expert-cache-gb` overrides (decimal GB of expert cache).
- Precedence: `--expert-cache-gb` > `--ssd-budget-gb` > setting > `ExpertStreamingRequired` 503 naming all three.
  An explicit launch flag always beats `model-settings.json` ([server-lifecycle](server-lifecycle.md#settings)).
- Admission `budget + planned KV <= wired limit`; the refusal names the `iogpu.wired_limit_mb` that would admit.
  Under `--no-mtp` the head is not loaded at all.
- An imatrix capture's accumulators live in GPU headroom that admission reads: budgets for capture runs drop
  (MiMo 100 → 94 GB; Flash-Next 96 → 80 GiB no longer admits higher under current bills).

## Cache policy

- `GroupCache`: plain per-layer LRU, prefill misses at MRU, every HIT of a route touched before any admit, surplus
  misses fall to the union workspace. Batched decode rides the union path.
- **MTP is refused at the door** (`ExpertStreamingMtpUnsupported`; `enable_mtp:true` = named 400): it prices at 1.27x
  expert bytes per committed token and the streamed forward declines spec's per-position SSM capture. Only an
  explicit `--mtp` refuses the load; the engine default resolves off (`[mtp] off (streaming; default)`), a
  `model-settings.json` `mtp: true` is dropped with a warning.
- **Load-time cache warm**: preload the lowest expert IDs into `floor(0.8 * slots_per_layer)` slots per MoE layer
  before kernel warmup and readiness, within the existing budget. These are ordinary LRU entries, not predicted
  routes; dense prefix layers are skipped.

## I/O

- `FillPool` = F_NOCACHE + F_RDAHEAD 0 positioned preads, fd cache validated by (dev, ino, size, mtime), spans sorted
  and coalesced to 64 MiB, page-aligned bounce otherwise.
- `PageSlab` epoch leases (`free → filling → ready → leased → readers_complete → reclaimable`, CPU writes only in
  `filling`).
- `importSlab` = `mlx_array_new_data_managed_payload` verified by pointer identity (`ExpertSlabImportCopied`
  refuses). Bench: `tests/ssd_fill_bench.sh`.
- **MLX releases an IMPORTED host buffer asynchronously**: `mlx_array_free` returns BEFORE the payload deleter runs;
  wait for the deleter (`SlabOperand.destroy`), leak (counted, logged on the breakdown line) rather than unmap what
  MLX still holds.

## Compute

- bf16 checkpoint → `expert_bf16_kernels` (`downKernelPreferred(rows) = rows >= 2`: the in-dispatch k-reduction tail
  loses at one row; `SUSHI_EXPERT_BF16_KERNELS=0` restores the `gather_mm` composite).
- Quantized packs → the RESIDENT fused kernels over the slab with remapped ids, bit-identical to the resident load
  (bytes and top-20 logprobs on greedy prompts). A warm quantized forward pays the per-layer barrier, not the fills.

## Correctness bars

- Store-level same-expert byte identity (`real qwen expert store spans and source bytes are exact`); teacher replay
  via `kld compare` (the affine pack is the control); greedy determinism.
- Cross-day comparisons must match forwards on `hits` + `fill_bytes_per_row` (the SSD's delivered rate drifts).
- `SUSHI_NGRAM_BF16_DIR=<hf checkpoint>` serves any pack with the ORIGINAL bf16 n-gram table so `kld compare`
  isolates the PLE table's cost.

<a id="imatrix"></a>
## Imatrix capture

Imatrix capture rides the streamed bf16 forward (`SUSHI_IMATRIX_OUT=<abs>.safetensors`, `src/imatrix.zig`):
per-layer per-expert sum(x²) and routed counts accumulate ON the GPU keyed by GLOBAL expert ids (slab slots are
remapped), in the collector's contract the converter reads; the flush runs on the INFERENCE thread (loop exit or
`/v1/unload-model`), never on `Scheduler.deinit`'s caller thread. MiMo's o_proj and lm_head inputs ride the same
file as per-channel mean squares under their source weight names ([arch-mimo-v2](arch-mimo-v2.md)). The drivers that feed it a corpus live in the private
converter repo. Routed counts reconcile to
tokens x top-k exactly on every layer; the two load-time warmup forwards add a few tokens.

## Discovery

A dense qwen4_exp checkpoint with a complete streaming index registers as a streaming stub
(`streamingStubMarker`); `/v1/models` carries `streaming`, `streaming_required`, `ssd_budget_gb` at top level and
`input_modalities: ["text"]` when it must stream. Guards: `tests/test_bf16_streaming.sh`,
`tests/test_model_settings.sh` [5].
