# Engine: memory, admission and context sizing

How the engine decides what fits on a unified-memory Mac: the GPU ceiling, load-time preflight, auto-context,
prefill chunk width, the admission line for a long prompt, and why under-billing is fatal. Read this before touching
any `*Bytes` bill, `Scheduler.init`, the preflight, or the admission path.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-kv-cache](engine-kv-cache.md),
[engine-qsa-long-context](engine-qsa-long-context.md#admission-and-load-time-bills),
[engine-prefix-cache](engine-prefix-cache.md#budget), [engine-expert-streaming](engine-expert-streaming.md#budget),
[arch-mimo-v2](arch-mimo-v2.md#bills-the-bill-follows-the-storage-in-the-same-commit).

## Metal OOM

- **Metal OOM is UNCATCHABLE and Metal at the working-set edge returns ZEROS before it aborts**: all-zero logits from
  healthy inputs = MEMORY symptom.
- `currentGpuMemoryCeiling` must see EXTERNAL pressure; under-billing is a Metal OOM, so a bill goes down only where
  the bytes are gone.
- The box: M5 Max 128 GB; the default wired limit admits about 120 GB; a resident MiMo EXL3 pack is ~97 GB, so two
  heavy GPU jobs at once risk an OOM for both (and concurrent conversions have died together in a GPU reset).

## Load-time preflight

- The preflight's available figure is free RAM CAPPED at Metal's working-set limit (`effectiveAvailableBytes`): a
  lowered `iogpu.wired_limit_mb` binds below free RAM, and a load past it failed warmup, then every request.
- Preflight refusals → `InsufficientMemory` → 503 + entry reset to `.unloaded`. A refusal quotes the number it
  COMPARED (`loadRequirementBytes`) and the flag that would admit (`--wired-margin-gib`, `--skip-mem-preflight`,
  `iogpu.wired_limit_mb`).
- Resident Flash-Next EXL3 with an explicit context bills weights plus min(flat headroom, 2 GiB load/warmup scratch
  + `sizerCtxKvBytes`); auto context, other layouts/architectures, sidecars and ANE keep flat headroom (min(weights/8,
  6 GiB) + 1 GiB). This is a load gate, not the request admission bill.
- `modelDiskBytes` bills the shards the INDEX names; an index that names NO shard on disk is STALE (every shard
  loads, one warning). Every size sum stats THROUGH symlinks (HF-cache models).
- Load-time bills run INSIDE `Scheduler.init` ([engine-qsa-long-context](engine-qsa-long-context.md)).
- **The kernel unwires a freed Metal buffer asynchronously** (~0.5 s for 50 GB): an unload and an eviction-before-load
  wait until most of the freed bytes left the wired set (`waitForUnwire`, bounded at 3 s), or the next preflight reads
  them as taken (45 GB free where 95 GB was a moment later).
- A ready entry's `bytes_resident` (the registry's resident-memory gate, `/v1/models`) is the weights the preflight
  billed (`residentWeightBytes`): a boot `--model` entry has no discovery `bytes_on_disk`, so it measures the shards.

### Explicit-context warmup envelope

`ad4a3ce0` plus the context-bill change, ReleaseFast binary mtime 2026-09-26 15:26:43 local, M5 Max 128 GB:
`test_load_context_preflight.sh`, `--ctx-size 1248 --kv-quant 8`, vision enabled, MTP as below. Each row is the
larger pre-request `/props` peak across startup and cold `/v1/load-model`; both paths then completed a short chat.
QoS `taskpolicy -a`, one `load-context-<pid>` GPU lock per run, conversion concurrent (memory validation, not timing).
The baseline is the existing flat formula, not an old-binary rerun.

| Pack | MTP | Old requirement (GiB) | Context requirement (GiB) | Load/warmup peak (GiB) |
|---|---|---:|---:|---:|
| Sushi-3bpw | on | 56.33 | 51.35 | 49.9073 |
| Sushi-3bpw | off | 56.33 | 51.35 | 47.6341 |
| Sushi-4bpw | on | 70.68 | 65.70 | 64.2628 |
| Sushi-4bpw | off | 70.68 | 65.70 | 61.6967 |

The 2 GiB allowance covers load/warmup scratch and fixed state outside the context bill, not arbitrary prompt
activations. Separate MTP sidecar files, assistant drafters and ANE retain flat headroom; other expert layouts and
architectures need their own measured envelope. These runs do not simulate a 64 GB host or establish a timing result.

## Context and chunk

- **Auto-context is PINNED at load** (`pinAutoContext`, 85% margin on the memory ceiling); ask
  `getEffectiveContextLength`. It bills KV at the CONFIGURED width and activations ONCE.
- The prefill CHUNK is a machine decision (`resolvePrefillChunk`, ladder 8192→512 at ≤ a quarter of the serving
  budget; `--prefill-chunk` wins). `prefillMemoryNeeded` takes STORED and SCORED widths as two parameters.
- A per-request arch (`perRequestPrefillChunk`: qwen4_exp and the ringed mimo_v2) re-picks the width for every
  request: the widest rung whose admission bill fits live memory (`chooseRequestPrefillChunk`), stepping down per
  chunk under pressure; the load-time pin is only the fallback. `boundedPrefillChunk` still caps the rung per arch
  (4096 at qk 192).
- An explicit `--ctx-size` outranks auto-context and `model-settings.json` `ctx_size`.
- Disconnect cancellation takes effect at the next prefill chunk boundary; a wall-time cancellation test must bound
  its chunk size rather than assume the auto-sized chunk fits a fixed deadline.

## Admission

- One `[admission] needed=… available=… reclaimable=… width=… verdict=…` line per decision.
- A long prefill evicts the hot cache on the INFERENCE thread to be admitted (`evictLruToAdmit`), crediting only
  provably reclaimable bytes; `PrefillDoesNotFit` → 400 by name.
- **Concurrent arrivals are each billed against the SAME free memory** on their connection threads. The gated arch
  (qwen4_exp) re-bills live memory before each prefill in `runPrefill`; an ungated one (mimo_v2) is re-billed at the
  pending drain (`admitsWithinMemory`: live requests plus this tick's earlier admits). One that does not fit beside
  company waits in `pending` (`[admission] held`); alone it proceeds.
- The hot-cache budget is clamped at load and follows residency ([engine-prefix-cache](engine-prefix-cache.md#budget)).
- Context-overflow 400s name BOTH counts.
- MiMo MTP adds a constant per-request and load-time reserve (`mimo_mtp.State.billedBytes`) for all three sliding head KVs, retained hiddens, and catch-up/concatenation buffers; it is zero with MTP off and never scales with context.
- **A vision encode is billed before it runs** (`towerFitFault`, `server.visionEncodeBill`): the largest block's tower
  scratch (`qwen_vision.encodeScratchBytes`, fitted >= 25% over the measured peak) plus every block's float32 pixels
  and three bf16 copies of its soft-token rows (group outputs, video concatenation, request concatenation); past what
  the GPU has left it is a named 400. The tower evaluates per block, so the peak is one block's f32 score sheet
  (heads x N^2) and rows; table in [arch-qwen4exp](arch-qwen4exp.md#vision-tower).
- **A video's block is ONE temporal group**, never the whole video: `forwardVideo` encodes and evaluates each group
  alone, so the bill grows linearly with the group count (the old N^2 over all groups billed 79 GB at 8x46x82).
  Measured peak (pixel upload to evaluated output) = one group's scratch + all pixels + the earlier groups' rows:

  | video (t x h x w patches) | peak | old bill | bill | bill / peak |
  |---|---|---|---|---|
  | 1 x 46x82 | 1249 MB | 1626 MB | 1658 MB | 1.33x |
  | 2 x 46x82 | 1277 MB | 5.6 GB | 1696 MB | 1.33x |
  | 4 x 46x82 | 1333 MB | 20.6 GB | 1771 MB | 1.33x |
  | 8 x 46x82 | 1445 MB | 79.1 GB | 1922 MB | 1.33x |
  | 2 x 24x42 | 172 MB | 590 MB | 246 MB | 1.43x |
  | 8 x 24x42 | 217 MB | 6.3 GB | 306 MB | 1.41x |
  | 2 x 96x96 (1536² cap) | 6237 MB | 30.3 GB | 8270 MB | 1.33x |
  | 8 x 96x96 (1536² cap) | 6648 MB | 460 GB | 8822 MB | 1.33x |

  `qwen vision ubench` on `242a5545` plus this change, Sushi-3bpw tower, random pixels, 3 passes (the image rows'
  peaks reproduced to the MB), `taskpolicy -a`, GPU lock `video-bill`, 2026-09-25; pinned by `visionEncodeBill covers
  each measured video peak`.

## Observing memory

`/props` reports `active_bytes`, `memory.cache_bytes`, `batching`; RSS is blind to Metal.
