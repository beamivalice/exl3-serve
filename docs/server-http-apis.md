# Server: HTTP APIs, streaming, sampling and constrained output

What each HTTP surface promises its clients: the OpenAI chat/completions/Responses and Anthropic Messages contracts,
streaming and usage chunks, logprobs, seeds and sampling, reasoning budgets and constrained JSON, plus the agent
launcher. Read this before touching `src/server.zig`, `src/responses.zig`, `src/launch.zig` or the sampler.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [server-tool-calling](server-tool-calling.md),
[server-lifecycle](server-lifecycle.md), [engine-mtp](engine-mtp.md).

## Surfaces

- **OpenAI chat/completions + Responses**: usage ALWAYS carries `prompt_tokens_details.cached_tokens`; thinking
  opt-ins = `reasoning_effort` OR `enable_thinking` (`reasoning_budget_tokens` outranks); `n>1` 400s.
- **Effort vocabulary** `off low medium high xhigh max` (`none` = off; `minimal` keeps the legacy 1024 budget):
  each served arch accepts a subset (`model.effortArms`), listed as `reasoning_efforts` on its `/v1/models` row; any
  other word 400s on chat, Responses and Anthropic with the accepted list, never rounded. qwen4_exp: off, low (2048),
  medium (8192), xhigh (uncapped); the template reads the word. mimo_v2: off, low (2048), medium (8192), high, xhigh,
  max (uncapped); the template has only on/off, the budget is the whole effect. Uncapped = `--reasoning-budget`. `/v1/responses`: `sequence_number` on every event, stateful via
  `ResponseStore`, WS via Upgrade. Continuing a partial reply: `continue_final_message` explicit on chat, INFERRED on
  `/v1/messages`.
- **Anthropic `/v1/messages`** (Claude Code): typed blocks, `input_schema`→`parameters`, stop-reason map incl.
  `stop_sequence` echo, full SSE block lifecycle; a `system`-role message past index 0 FOLDS into the leading system
  message (`foldSystemMessages`); `developer` reads as `system` (`canonicalRole`).
- `/v1/models` rows carry `context_length` + `max_model_len` at TOP level. Context-overflow 400s name BOTH counts.
- Endpoint EXISTENCE never depends on model state and the 404 is answered BEFORE the model resolves (`ROUTE_PATHS`);
  a status route never reaches `ensureLoaded` (`handlePropsNoModel`). Removed upstream routes answer named 404s.
- A content array's text parts JOIN in order (`joinedTextParts`).

## Streaming

- **A stream and a non-stream answer are the SAME BYTES**; leading whitespace is the one thing a stream may withhold
  (`streamContentLead`). A spent reasoning budget WITHHOLDS the rest of the thought; a non-stream tool-call reply
  carries the pre-markup text (`visibleToolPreamble`); a non-stream disconnect reports `client_disconnect`, never
  `length`; a stop sequence cuts at its INDEX (`stopSequenceCut`); request ints clamp (`parseRequestSeed`,
  `clampJsonI32`).
- **`stream_options.include_usage` chunk ships `"choices": []`** (`sendSSEUsageChunk`); the ending appears on exactly
  ONE chunk; a client cannot time our stream — use the final chunk's server `timings`.
- Liveness is a property of the SOCKET: `beatStreamKeepalive` at the bottom of every streaming loop, emit on 5 s
  byte-silence. `--timeout` is a STALL timeout (`StallClock`).
- **NO string built from model bytes is guaranteed UTF-8**: sanitizing lives INSIDE the escaper (`chat.utf8Next`
  under every `jsonEscape`/`appendJsonString`); logprobs `bytes` keeps the exact bytes. Hand-written error text is
  escaped at the SINK.

## Seeds, logprobs, sampling

- **A `seed` binds EVERY sampler with a fresh key PER DRAW** (`generate.seedKey`).
- Logprobs are the MODEL's distribution (pre-temperature), ids travel WITH values, entry belongs to the RETURNED
  token (one-token delay); `logprobs.content` describes `message.content` (`contentTokenRange`); streaming logprobs
  are a SIBLING of `delta` shipped EXACTLY once against a high-water mark. logprobs>0 + grammar disable spec.
- Sampling defaults for omitted fields: body > launch flags > model `generation_config.json` > hardcoded.
- Top-k and top-p are ONE pass (`filterTopKTopP`); a filter cuts by RANK, never by value (bf16 ties at the top
  constantly; `ranksDescending` ties by lowest id); the nucleus is the mass STRICTLY above each rank, cumsum in f32;
  `top_p` 0 is GREEDY (`applyTopP` floors at `floatMin(f32)`).
- A sampler never draws a RESERVED special or PADDING row (`installSuppressMask`; logprobs stay RAW).

## Reasoning budget

Enforced at DECODE (`server.armThinkBound` → `SamplingParams.think_bound`, `scheduler.thinkBoundTick`): at the budget
the early-stop line + the atomic closer commit as ONE multi-token forward (`commitForcedTokens`); the whole closed
thought is delivered. Guard: `tests/test_reasoning_budget_stream.sh`. Effort budgets = pi's ladder
(`model.effortArms` for served arches, `responses.effortBudget` for the rest).

## Constrained JSON

- The payload offset is AUTHORITATIVE (`reasoning_protocol.Delivery`, all surfaces, stream + non-stream).
- The grammar mask never walks the whole vocabulary (`token_mask.buildMask`); every grammar state has a legal byte;
  no whitespace OUTSIDE the root value, the model's OWN layout inside (`MAX_FREE_WS` 16).
- Every schema-mask surface uses ONE thinking policy (`schemaMasksThinking`); tools present = no mask. Per-model
  grammar table lives on `LoadedModel`.
- Code: `src/json_schema.zig` / `src/json_grammar.zig` / `src/token_mask.zig` / `src/regex.zig` (schema IR →
  streaming grammar → per-token mask), `src/reasoning_protocol.zig`.

## Security and observability

- `--api-key`: loopback exempt, `/health` + OPTIONS open, `constTimeEql`.
- `--metrics`: zero cost off; TTFT at prefill completion; live tok/s via ONE atomic per tick; `/metrics(.json)`.

## Agent launcher (`sushi launch <agent>`)

- `src/launch.zig` (claude/pi/omp/opencode/codex/hermes/aider): reads `/v1/models`, writes agent configs into
  `~/.sushi/<agent>/`. Launcher env: `ANTHROPIC_BASE_URL` + dummy keys + `ANTHROPIC_DEFAULT_*_MODEL=sushi`.
- Agent budgets (`launch.budgetForContext` + `compactionReserve`): output share ctx/2, compaction reserve ctx/4
  capped at 20000, carried into pi's `settings.json` and opencode's `compaction` + `limit.output`. A launch below the
  agent's context floor WARNS (claude 64k, opencode 32k, others 16k).
