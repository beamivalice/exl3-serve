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

- Preflight refusals → `InsufficientMemory` → 503 + entry reset to `.unloaded`. A refusal quotes the number it
  COMPARED (`loadRequirementBytes`) and the flag that would admit (`--wired-margin-gib`, `--skip-mem-preflight`,
  `iogpu.wired_limit_mb`).
- `modelDiskBytes` bills the shards the INDEX names; an index that names NO shard on disk is STALE (every shard
  loads, one warning). Every size sum stats THROUGH symlinks (HF-cache models).
- Load-time bills run INSIDE `Scheduler.init` ([engine-qsa-long-context](engine-qsa-long-context.md)).

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
- **A vision encode is billed before it runs** (`towerFitFault`): the largest block's tower scratch
  (`qwen_vision.encodeScratchBytes`, fitted >= 25% over the measured peak) plus every block's soft-token rows; past
  what the GPU has left it is a named 400. The tower evaluates per block, so the peak is one block's f32 score sheet
  (heads x N^2) and rows; table in [arch-qwen4exp](arch-qwen4exp.md#vision-tower).

## Observing memory

`/props` reports `active_bytes`, `memory.cache_bytes`, `batching`; RSS is blind to Metal.
