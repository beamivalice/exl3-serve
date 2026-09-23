# Server: lifecycle, loading, settings and scheduling

How a model gets from a path to a serving slot and back: discovery, the arch gate, the one weight-loader decision,
load and unload on the inference thread, settings precedence, the scheduler's slots and batching, threads, and the
ownership rules that keep request data alive. Read this before touching `src/scheduler.zig`, `src/model_registry.zig`,
`src/model_settings.zig`, `src/main.zig` or `src/cli.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [server-http-apis](server-http-apis.md),
[engine-memory-admission](engine-memory-admission.md), [engine-mlx-gotchas](engine-mlx-gotchas.md).

## Entry points

- `src/main.zig`: entry, CLI flags + subcommands (`run/pull/list/serve/launch/kld`).
- `src/cli.zig`: alias → HF repo, resumable pull into `~/.sushi/models/<org>/<repo>`, `list`, `run` REPL.
- **The embedded REPL uses in-process HTTP**: never fork `curl` from the resident engine for readiness checks or chat
  turns. Test `run` on a real TTY; a serving-only smoke test does not exercise its client.
- **An arg loop with no else branch is a silent flag eater** (`cli.classifyUnparsedArg`): every `--flag` any script
  passes must be in main.zig's match list. Removed flags are rejected by name, never eaten.

## What loads

- The arch gate: the loader refuses any `model_type` outside `model.served_model_types` (`qwen4_exp`, `mimo_v2`) by
  name (`ArchitectureUnsupported` → 503). A `.gguf` is refused by name (`GgufEngineUnsupported` → 503; `--model`
  exits).
- **The weight loader is ONE decision** (`model.loadWeightsForConfig`: streaming index > MiMo source trunk > vision >
  plain). A second site builds a model the server never serves — a MiMo pack read without its source trunk binds the
  raw FP8 fused QKV and its logits stop following the routed experts.
- **A missing tensor is a load ERROR, never `unreachable`** (`error.MissingWeight` → named 503 via
  `loadErrorFromName`); a load failure crosses the inference thread by NAME (`req.error_name`).
- Discovery (`src/model_discovery.zig` / `src/model_registry.zig`): two-level org/name, multi-root, streaming stubs,
  multi-model registry. **`--model-dir` is REPEATABLE** (`discoverModelsMany` merges roots FIRST-WINS). One path never
  registers under TWO ids (`registry.peekByPath`).
- **A reload FREES the CPU state `unloadResident` retains** while the entry is `.loading` (`releaseRetainedCpuState`):
  a reader holding no refcount takes the mutex AND skips them while `.loading`.

<a id="settings"></a>
## Settings precedence

- **An explicit launch flag outranks `model-settings.json`**, which outranks the default (`model_settings.pick`;
  `--ctx-size 0` = not given). Applies to `--mtp/--no-mtp`, `--kv-quant`, `--ctx-size`, `--mtp-typical/--mtp-tokenv3`,
  `--ssd-budget-gb/--expert-cache-gb`; a request's own field still applies on top. Design reviews reject "file beats
  flag".
- A flag that shapes a LOAD is retained on the Scheduler with its `*_explicit` bit (`ensureLoaded`'s cold-load
  `LoadRequest` is a SECOND site); read via `server.manualContext` / `kvCacheFor` / `mtpChoiceFor`. Each load logs its
  resolved value and source (`[kv-cache] kv8 (source); ctx N (source)`, `[mtp] on|off (source)`; `/props
  settings.mtp.source`). Guard: `tests/test_cold_load_launch_flags.sh`, `tests/test_model_settings.sh`.
- Per-model settings live in `~/.sushi/model-settings.json` (`src/model_settings.zig`: `ctx_size`, `kv_quant`,
  `mtp`, `mtp_acceptance`, `ssd_budget_gb`), stamped at BOTH load construction sites and resolved ONCE in
  `doLoadOnInferenceThread`; read via `server.manualContext(config)` / `configuredKvQuantFor(config)`, never the raw
  server config.
- A new per-model setting or launch flag follows this order, carries an `*_explicit` bit through both load sites and
  cold loads, and logs its resolved value with its source at load.

## Scheduler and batching

- `src/scheduler.zig`: slots, inference thread (sole MLX caller), queues, batching, admission, spec wiring, hot-cache
  budget revise.
- Text slots BATCH-decode on `qwen4_exp` (`configBatchesDecode`); `--max-concurrent` sizes the submit queue. A
  batched group is capped by PADDING WASTE (`batchedKvKeepCount`, `MAX_PAD_WASTE` 1.5 < 2.0), not slot count.
- A cold prefill YIELDS to decode ticks at chunk boundaries (`scheduler.interleaveDecodeTick`;
  `SUSHI_PREFILL_INTERLEAVE=0` restores). Greedy byte-identical.
- **Serial ≠ exclusive**: only a slot driving a module-owned decode state is exclusive (`slotExclusiveDecode`);
  qwen4's state is read-only shared and batches freely. The batched-decode gate reads DISPATCH, not ARMED flags
  (`slotTicksRegular` asks `specTickMode`). A batched decode guard that only runs at N=1 pins nothing:
  `tests/test_batched_equivalence.sh` runs a real two-stream arm.
- `src/generate.zig`: generation, sampling, MTP orchestration, `StallClock`, prefill chunking, loop-stop tiers,
  `commitForcedTokens`. `src/tokenize_cache.zig`: per-LoadedModel LRU of rendered+encoded prompts.

## Threads

- Detach every per-connection `std.Thread` immediately; on teardown drain conn threads before `scheduler.deinit`.
- Sleep inhibition follows the inference-thread wait.
- `Slot.deinit` runs on conn threads: it stores marks, the inference thread frees.

## Request ownership and media

- **`messages.deinit(allocator)` frees the Message array and NOTHING it points at**: request media is owned by ONE
  `server.RequestMedia`; `Message` BORROWS. Ownership by PROVENANCE (`{slice, owned}` returns), never
  free-unless-equals-literal.
- **A media placeholder id occurs in ordinary TEXT**, so a media boundary is gated on the request CARRYING media
  (`firstMediaPlaceholder(has_media)`); active-turn media is selected from WIRE METADATA before decoding; a decode
  failure is a NAMED 400, never a silent drop; media on a tower-less/streamed load is refused by NAME
  (`mediaRejectReason`).
- Media INPUT code: `src/vision.zig` / `src/qwen_vision.zig` / `src/mrope.zig` (Qwen3-VL image/video tower, M-RoPE
  positions, audio embedder forward); `stb_image` + libwebp decode image input.

## Config reading

- `generation_config.json` `eos_token_id` is part of the stop set (additive). Read `text_config` FIRST, then root,
  PER FIELD. A config field HF allows in two SHAPES must be read as both (`chat_template` string OR list); `.string`
  on unchecked `std.json.Value` panics.
- When an arch's reference IGNORES a config field, that field is not the truth.
- A reference probe with SYNTHETIC dtypes proves the reference's SEMANTICS, not the checkpoint; parity fixtures for
  deep stacks are dumped fp32 on CPU.
