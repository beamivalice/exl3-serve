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
- **`--parent-pid <pid>`** is for a host that runs sushi as its engine (`src/parent_watch.zig`): a thread polls the pid
  once a second and, once it is gone (or has reparented sushi), sends this process SIGTERM. Mid-load that ends the
  process; in the serve loop it is the ordinary graceful shutdown. Test: `tests/test_parent_pid.sh`.
- **`sushi --guest-manifest`** prints the JSON such a host checks before routing (`version.writeGuestManifest`):
  version, commit, `guest_api`, the mlx/mlx-c pins, `min_macos` from the binary's own target, `model_types`
  (`version.guest_model_types`, a subset of the served types) and the EXL3 codebooks and window range the decoder
  accepts. The release tarball ships it as `guest.json`, next to a `.sha256` of the tarball.

## What loads

- **Bind**: `server.resolveBind` defaults to `127.0.0.1:11234`; `--host` takes an IPv4 literal, `0.0.0.0` or
  `localhost` (= 127.0.0.1; anything else is refused by name, never widened). Before any model loads,
  `ensurePortFree` probes the address with a connect AND a bind, so a listener or a bound-but-silent socket both refuse
  with "port N is already in use"; the listener binds with SO_REUSEADDR only, so a racing second sushi fails its bind
  with the same message instead of co-binding. Guard: `tests/test_port_conflict.sh`.
- The arch gate: the loader refuses any `model_type` outside `model.served_model_types` (`qwen4_exp`, `mimo_v2`) by
  name (`ArchitectureUnsupported` → 503). A checkpoint in an unsupported file format is refused by name
  (`ModelFormatUnsupported` → 503; `--model` exits).
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
- MTP's default is ON for the served archs (source `default`; `/props settings.mtp.default_on` true); the flag and
  the file can only turn it off, or on for an SSD-streamed pack.
- `[pld] on|off (source)` and `/props settings.pld` report what a slot runs (`server.pldReport`): a module-wired arch
  (qwen4_exp) reads `off (module spec wiring)` whatever `--pld` says, since `scheduler.specInitWiring` never runs it.
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
- A request's sampling state (`think_bound`, `constraint`) lives in its handler's frame: `complete` waits out any
  inference pass holding the slot (`Slot.in_pass`, taken under `queue_mu`) before the handler may free it.
  Guard: `tests/test_cancel_mid_tick.sh`.

## Request ownership and media

- **`messages.deinit(allocator)` frees the Message array and NOTHING it points at**: request media is owned by ONE
  `server.RequestMedia`; `Message` BORROWS. Ownership by PROVENANCE (`{slice, owned}` returns), never
  free-unless-equals-literal.
- **A media placeholder id occurs in ordinary TEXT**, so a media boundary is gated on the request CARRYING media
  (`firstMediaPlaceholder(has_media)`); media on a tower-less/streamed load is refused by NAME (`mediaRejectReason`).
- **Every message's media is decoded and placed where it was sent**: user parts, OpenAI `tool` messages, Anthropic
  `tool_result` blocks, Responses `input_image` (tool outputs too). The wire walk (`readOpenAiMessages`,
  `readAnthropicMessages`, `responses.parseInput`) records each part's offset in the joined text
  (`Message.media_parts`); the serializer hands the template a typed part list, so the TEMPLATE renders each
  placeholder; `prepareRequestMedia` expands every pad to its block's rows and encodes all blocks in prompt order.
  The engine never inserts pads itself: a template that renders fewer placeholders than blocks is a named 400.
- **Media refusals are named**: an undecodable image is a 400 naming `messages[i]`/`input[i]` and the reason
  (`imageRejectReason`), an `input_audio` part a 400 unless the model encodes audio (`RequestMedia.accepts_audio`),
  more than `chat.MAX_REQUEST_IMAGES` (64) a 400 with both counts, a prompt the media pushes past the context a 400
  naming the media's tokens, an encode that does not fit a 400 (`towerFitFault`), a failed encode a 500
  (`MediaFault`); never a text-only answer.
- Media INPUT code: `src/vision.zig` / `src/qwen_vision.zig` / `src/mrope.zig` (Qwen3-VL image/video tower, M-RoPE
  positions over every block); `stb_image` + libwebp decode image input.

## Config reading

- `generation_config.json` `eos_token_id` is part of the stop set (additive). Read `text_config` FIRST, then root,
  PER FIELD. A config field HF allows in two SHAPES must be read as both (`chat_template` string OR list); `.string`
  on unchecked `std.json.Value` panics.
- When an arch's reference IGNORES a config field, that field is not the truth.
- A reference probe with SYNTHETIC dtypes proves the reference's SEMANTICS, not the checkpoint; parity fixtures for
  deep stacks are dumped fp32 on CPU.
