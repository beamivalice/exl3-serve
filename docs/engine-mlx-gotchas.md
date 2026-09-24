# Engine: MLX and Zig gotchas

The ways MLX, mlx-c and Zig have bitten this engine: dtype promotion, lazy evaluation and barriers, view vs copy
semantics, error handling, the allocator pool, and a few Zig and tokenizer traps. Read this before writing graph-
building code or anything that crosses the mlx-c boundary.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-kernels](engine-kernels.md),
[engine-memory-admission](engine-memory-admission.md), [server-lifecycle](server-lifecycle.md).

## The one MLX thread

The inference thread is the SOLE mlx caller (even frees). `Slot.deinit` runs on conn threads: it stores marks, the
inference thread frees. A pointer-keyed cache is invalidated by an ATOMIC MARK, never an off-thread free.

## Errors

- **An MLX failure is CATCHABLE; mlx-c's DEFAULT handler `exit(-1)` is what killed us** (`mlx.installErrorHandler`
  once in `main()`): `checkError` per prefill chunk (before snapshot/persist) + `checkErrorDecode` per tick; a
  latched error never 200s; streaming shares `mapGenerationError`.
- A swallowed failure must DROP the latch it raised (`dropLatchedErrorUnless(had_error)`, passing the
  `errorPending()` read BEFORE the op). Guard: `tests/test_mlx_error_recovery.sh`.
- A client-supplied PATH is proven on OUR side of the mlx boundary (stat → 400; an MLX error there latches).

## Dtypes

- **An f32 SCALAR array promotes every bf16 operand it touches**: scalars go through `scalarOf(v, dtype)`; a chain
  that returns f32 BY DESIGN makes the CALLER own the dtype; `[dtype-trace] residual widened` is the tell.
- A load-time constant table in the WRONG DTYPE silently widens every read (`constTableAs`).
- Dtype-gate any `mlx_array_data_float32` load-time read.

## Laziness and barriers

- **A host read inside a layer loop is a GPU BARRIER**: defer non-consumed reads into ONE batched eval.
- A rollback that re-forwards is a SECOND forward: capture only what verify overwrites, truncate the rest by offset.
- A multi-token forward is not a prefill (`prefillEvalCadenceApplies`, seq ≥ 32).
- A weight outside every warmup forward is still LAZY at serve time (`appendWeightArrays`).
- Helpers that could take a block use the `_axis` op.

## Views, copies, ownership

- Slice-born weights into gather_qmm/quantized_matmul are `mlx_contiguous`-materialized at load; mlx
  `Copy`/`contiguous` are VIEW ops (a slice OUTLIVING its parent goes through `materializedOwnedCopy`).
- A raw data-pointer read must PROVE row-major contiguity; a helper that materializes a VIEW owns it (`takeContig`).
- A weights MAP outliving the model pins every buffer; mlx-c `iterator_next` hands a +1; `mlx_array_new_data`
  COPIES shape-worth of bytes.
- **A refcount-shared snapshot makes every later write copy the whole buffer** (MLX donates only a sole owner's
  buffer): a spec rollback that can truncate by offset takes no `KVCache.snapshot`.
- MLX releases an IMPORTED host buffer asynchronously ([engine-expert-streaming](engine-expert-streaming.md#io)).

## The allocator pool

- `mlx_clear_cache()` once per CHUNK and per emitted block, INTERVAL-based (`step -| last_clear >= 256`),
  un-skippable; `Generator.advanceStep` is the one step mover.
- `active` flat while phys climbs = the POOL (`memory.cache_bytes`). RSS is blind to Metal.

## Zig

- A struct inside a generic fn that captures NO comptime param is memoized to ONE type.
- `openDirAbsolute` on an empty/relative path is ReleaseFast UB — guard every site.
- `.string` on unchecked `std.json.Value` panics.
- A test that aliases embedded bytes through an alignment cast is a coin flip per binary: copy fixtures to aligned
  storage (Debug builds catch what ReleaseFast hides).

## Tokenizer

- Special-token splitting only in `Tokenizer.encode` (first-byte buckets); any per-position loop over a
  vocab-derived collection needs an index.
- A hand-rolled pretokenizer is calibrated to ONE tokenizer.json — digit GROUPING is per-model (`digit_group`);
  cross-check `/tokenize` vs HF at bring-up.
