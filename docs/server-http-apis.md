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
  max (uncapped); the template has only on/off, the budget is the whole effect. Uncapped = `--reasoning-budget`.
  A thinking request that names NO effort gets no server budget (decided): Qwen3.8 renders it as low
  (`chat.qwen38EffortFor`), whose preamble shortens the thought without truncating it. `/v1/responses`:
  `sequence_number` on every event, stateful via
  `ResponseStore`, WS via Upgrade. Continuing a partial reply: `continue_final_message` explicit on chat, INFERRED on
  `/v1/messages`.
- **Anthropic `/v1/messages`** (Claude Code): typed blocks, `input_schema`→`parameters`, stop-reason map incl.
  `stop_sequence` echo, full SSE block lifecycle; a `system`-role message past index 0 FOLDS into the leading system
  message (`foldSystemMessages`, also run by `responses.parseInput`, since Codex sends a mid-input `developer` turn);
  `developer` reads as `system` (`canonicalRole`).
- `/v1/models` rows carry `context_length` + `max_model_len` at TOP level. Context-overflow 400s name BOTH counts.
- `/v1/models` `meta.quantization` reports EXL3’s configured expert rate and dense width (e.g. `EXL3 3bpw experts, 8-bit dense`) for loaded and unloaded packs; affine labels remain `{bits}-bit`, and `/props` numeric quantization fields retain their dense-trunk meaning.
- Endpoint EXISTENCE never depends on model state and the 404 is answered BEFORE the model resolves (`ROUTE_PATHS`);
  a status route never reaches `ensureLoaded` (`handlePropsNoModel`). Removed upstream routes answer named 404s.
- A content array's text parts JOIN in order (`joinedTextParts`); its media parts render at the offset they sat at.
- **Media is read from EVERY message** on all three surfaces: chat `image_url`/`video_url` parts in any role
  (`tool` included), Anthropic `image` blocks beside the text or inside a `tool_result`, Responses `input_image` in
  a message or a `function_call_output` array. Only base64 data URLs decode (remote URLs are refused, not fetched);
  a failure is a 400 naming the message index and the reason, an `input_audio` part a 400 on a model without an
  audio encoder, more than 64 images a 400 with both counts, an encode failure a 500 (Anthropic: `api_error`).
  Stored Responses history keeps text only.

## Streaming

- **A stream and a non-stream answer are the SAME BYTES**; leading whitespace is the one thing a stream may withhold
  (`streamContentLead`). A spent reasoning budget WITHHOLDS the rest of the thought; a non-stream tool-call reply
  carries the pre-markup text (`visibleToolPreamble`); a non-stream disconnect reports `client_disconnect`, never
  `length`; a stop sequence cuts at its INDEX (`stopSequenceCut`); request ints clamp (`parseRequestSeed`,
  `clampJsonI32`). A streamed tool call that never completes is the one exception: closed JSON, `length`
  ([server-tool-calling](server-tool-calling.md)).
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
- Logprobs are `logits - logsumexp` in f32 (`computeLogprobs`): `log(softmax)` in bf16 lands on bf16's grid, 0.125
  apart between -16 and -32.
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
Every surface arms it with one precedence: explicit budget (`reasoning_budget_tokens`, Anthropic `budget_tokens`) > the
effort word's budget > `--reasoning-budget`. `/v1/responses` parsed the word and dropped the budget.

## Constrained JSON

- The payload offset is AUTHORITATIVE (`reasoning_protocol.Delivery`, all surfaces, stream + non-stream).
- The grammar mask never walks the whole vocabulary (`token_mask.buildMask`); every grammar state has a legal byte;
  no whitespace OUTSIDE the root value, the model's OWN layout inside (`MAX_FREE_WS` 16).
- Every schema-mask surface uses ONE thinking policy (`schemaMasksThinking`); tools present = no mask. Per-model
  grammar table lives on `LoadedModel`.
- Code: `src/json_schema.zig` / `src/json_grammar.zig` / `src/token_mask.zig` / `src/regex.zig` (schema IR →
  streaming grammar → per-token mask), `src/reasoning_protocol.zig`.

## Security and observability

- `--api-key`: loopback exempt, `/health` + OPTIONS + `GET` of the chat page open, `constTimeEql`.
- `--metrics`: zero cost off; TTFT at prefill completion; live tok/s via ONE atomic per tick; `/metrics(.json)`.

## Chat page (`GET /`, `GET /chat`)

- One self-contained file, `src/webui/index.html` (CSS, JS and the logo inline, no external fetch), embedded with
  `@embedFile` and served as `text/html; charset=utf-8`. Any other method on those two paths is a 405 answered BEFORE
  model resolution, so it can never cold-load a model. Guards: `tests/test_webui.sh`, the `chat page:` tests.
- It speaks only the public API: `/v1/models` for the picker (`reasoning_efforts` fills the effort select, `vision` or
  an `image` input modality shows the attach button), `/v1/chat/completions` streamed with `include_usage` (the
  readout is the final chunk's `usage` + `timings`), `reasoning_content` shown collapsed. Stop aborts the fetch; the
  server cancels on disconnect.
- Under `--api-key` the page is served without the key (it holds no data), asks for it on the first 401 and sends it as
  a Bearer token. Its fetches use `credentials: "omit"`: the 401's Basic challenge would otherwise open the browser's
  own login dialog.
- Conversations live in the browser's `localStorage` (every access guarded); attached images stay in memory only, as a
  few photos would fill the storage quota.
- Startup prints `chat in your browser: <url>` once (`chatPageUrl`: a `0.0.0.0` bind shows as `127.0.0.1`); `sushi run`
  prints it under its banner, since its log is quieted to warn.

## Agent launcher (`sushi launch <agent>`)

- `src/launch.zig` (claude/pi/omp/opencode/codex/hermes/aider): reads `/v1/models`, writes agent configs into
  `~/.sushi/<agent>/`. Launcher env: `ANTHROPIC_BASE_URL` + dummy keys + `ANTHROPIC_DEFAULT_*_MODEL=sushi`.
- Agent budgets (`launch.budgetForContext` + `compactionReserve`): output share ctx/2, compaction reserve ctx/4
  capped at 20000, carried into pi's `settings.json` and opencode's `compaction` + `limit.output`. A launch below the
  agent's context floor WARNS (claude 64k, opencode 32k, others 16k).
- pi sends its thinking level as `reasoning_effort` through a per-model `thinkingLevelMap` built from the row's
  `reasoning_efforts` (`launch.piEffortFor`: exact, else the next accepted word up, else down; Qwen3.8 high → xhigh).
  pi's `thinkingFormat: qwen` sent only `enable_thinking`, so low/medium never reached the server.
- omp (a pi fork) has no off entry in its maps: off rides the qwen dialect (`enable_thinking: false`), `whenThinking`
  switches thinking requests to `reasoning_effort`, and a per-model `thinking` block remaps each level with the same
  rule; `requiresEffort: false` stops omp clamping off to the lowest effort.

## `sushi run` research tools (client-side)

- **The REPL runs the tools, the server never does** (`src/repl_tools.zig`, loop `cli.runToolTurn`): it sends `tools`,
  runs the returned calls, appends `tool` messages and asks again. OFF by default: `--tool on|off`, `/tool on|off`,
  bare `/tool` shows the state and list. One dim trace line per call (`search:`, `fetch:`, `read:` …).
- Tools: `web_search` (GET html.duckduckgo.com, top 8 title/url/snippet, `uddg=` unwrapped, ads dropped),
  `fetch_url` (GET, ≤5 redirects, 10 s wall clock, 2 MB, HTML → text ≤20k chars), `read_file` (≤256 KB),
  `list_dir`, `search_files` (substring or regex, ≤100 hits), `view_image` (only when `/v1/models` lists `vision`).
- **8 tool rounds per user turn**, then a user nudge and one request WITHOUT tools for the final answer.
- **Only the latest USER turn's images are decoded** (`server.activeWireMediaIndex`): a tool image rides a synthetic
  user turn after the tool results. `/image <path>` attaches to the next message; a pasted path is never attached.
- **File tools are confined to one folder by REAL path**, the start folder until `/cd <folder>` moves it
  (`changeRoot`: absolute, `~` or relative to the current folder; must be a directory, symlinks resolved, a path with
  a secret name refused; bare `/cd` shows it). `..`, outside absolutes and escaping symlinks are refused, as are dot
  entries and secret names (`.env*`, `*.pem`, `*.key`, `id_*`, `*.p12`, `credentials*`, `*.keychain*`, `.ssh`,
  `.aws`, `.gnupg`), checked both as typed and after resolution (`confinePath`).
- An outside path's refusal tells the model the folder is fixed and the user can type `/cd <folder>`, so it asks for
  that instead of guessing other paths.
- The user's `/image <path>` (`loadUserImage`): a RELATIVE path resolves in the `/cd` folder under the same
  confinement; an ABSOLUTE or `~` path (typed or dragged in) may leave it, but a secret name anywhere on it or a hidden
  file name is refused, as typed and resolved (`userPathRefusal`). The model's `view_image` stays confined.
- Every prompt carries that folder and the tools state, dim: `~/project · tools on >>> ` (`formatPromptStatus`: `~` for
  `$HOME`, `…` and the tail past 32 characters); it is rebuilt before each input, so `/cd` and `/tool` show at once.
  The ready banner prints the same pair.
- **Web tools reach public hosts only**: http/https, no userinfo, local names refused, EVERY resolved address and the
  connected peer (`getpeername`, defeats DNS rebinding) must classify public (`classifyIp4/6`; mapped, NAT64 and 6to4
  judged by their IPv4); each redirect hop re-checked; no cookies, auth headers or POST.
- Every failure is a short tool-result string; results are data, never executed. A DuckDuckGo bot check (HTTP 202,
  `anomaly-modal`) reads as "search unavailable", never as zero results.
