# Server: chat templates, thinking and tool calling

How prompts are rendered and how model output is split into reasoning, content and tool calls: the Jinja render and
its silent fallback, the tool-call parse chain and its one chokepoint, think-tag handling, loop stops, and the
replay-pinned invariants. Read this before touching `src/chat.zig`, the tool paths in `src/server.zig`, or
`src/format_corpus_test.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [server-http-apis](server-http-apis.md),
[tests/CLAUDE.md](../tests/CLAUDE.md).

## The pipeline

- With `tools`, tokens buffer for detection; thinking buffers separately. Parse chain strict → tolerant repairs →
  truncation salvage, then the ONE chokepoint `server.parseToolCallsForRequest` = parse → inferred-name filter →
  parallel clamp → buried-param hoist → schema coercion (last two gated by `--no-tool-autocorrect`; emitted
  `arguments` ALWAYS valid JSON).
- Serialization `chat.serializeMessagesJson`: role "tool" native, args as JSON STRINGS, every string via
  `appendJsonString`. Streaming: full args in ONE SSE delta, thinking → `reasoning_content`.
- **Hard invariants (replay-pinned)**: emitted args ALWAYS valid JSON; every converter escapes + dedups; coercion
  never worsens conformance; a parsed NAME never contains `<|`; no tag leaks. Harness:
  `src/tool_traffic_replay_test.zig` over `src/fixtures/tool_traffic.jsonl`.
- A live failure revealing a CLASS ships the instance test plus a corpus entry or invariant in
  `src/format_corpus_test.zig` plus a rule here. Capture traffic: `SUSHI_RAW_DUMP_FILE=<abs>` →
  `tests/harvest_tool_traffic.py`. Reproduce tool bugs `stream:false` first.

## Templates

- **Control bytes**: ONE raw byte <0x20 in history kills the strict render → SILENT `fallbackFormatChat` (model loses
  its stop token). Everything through `appendJsonString`; wrong-family tags out ⇒ suspect silent fallback first. A
  NUL byte truncated the rendered prompt (`jinja_render_chat` returns its LENGTH; tell: the same `prompt=` count on
  consecutive turns).
- **A `chat_template` value can be a POINTER** (`{% include 'chat_template.jinja' %}`): `chat.isIncludeStub` reads it
  as "no inline template" so the sidecar loads. Grep the log for `jinja` first.
- A template can raise on OUR extra-context values: `serializeExtraContext` sniffs the family; tool-call `arguments`
  stay OBJECTS; history tool_calls carry `"id"`; only a refusing template gets `noThinkTailSuffix`.
- **A system turn past index 0 renders where the template allows it**: a template that raises on it (Qwen3.8) or
  drops it gets it folded into the leading system (`templateProbeRendersLateSystem`, every surface); MiMo's role
  loop keeps it in place, byte for byte.
- **A generic ChatML role header preserves tool roles**: absence of a literal `'tool'` branch does not license
  rewriting tool results as user text (`templateReferencesToolRole`).
- **Assistant-history reasoning round-trips** (`Message.reasoning_content`, OMITTED when absent). A contract COMMENT
  is read as a spec — pin it with a test.

## Parsing tool calls

- **A `<tool_call>` body carrying `<function=` is the XML dialect and is read FIRST** (qwen 3.5+ template mandates
  it); a parameter VALUE never decides the call. A `<parameter>` VALUE may spell the dialect's own close tags
  (`hermesValueEnd` = LAST `</parameter>` before the next opener). A Hermes value keeps its own whitespace
  (`stripHermesValueFraming`).
- **A JSON call cut INSIDE the object still names its tool** (`truncatedJsonCallName`): recover NAME + `{}`, NEVER ship
  partial values, never ship raw markup as content. Model-mangled arg JSON → `looseRepairToolCallJson`, never drop
  the whole call. A tag parser never bails on ONE missing delimiter.
- **A `</think>` inside a tool ARGUMENT is payload** (`thinkCloseIsToolCallPayload`): decline a close whose nearest
  preceding tool opener is still OPEN AND whose block closes afterwards.
- **Types come from the SCHEMA, never the value's spelling** (`coerceToolArgsToSchema`; undecidable → untouched).
  Buried required params hoist only on all-schema-read unanimity. A container string with a key repeated at the SAME
  value still coerces (`parseContainerAllowingRepeats`). Heuristic raw-JSON inference must name a DECLARED tool
  (`filterInferredBySchema`).

## Thinking

- Strip pos-0 unclosed openers; `trimTrailingThinkClosers`; unparsed tool markup never rides out as reasoning OR
  content (`trimLeakedToolMarkup`, ONE cut).
- Whether a prompt ends inside a think block is a property of the RENDERED BYTES (`promptOpensThink`), never ANDed
  with `enable_thinking`; `in_think_block` seeds from `prompt_opened_think` ALONE at every stream site; a model can
  open its OWN block (`modelThinkOpener`).
- **Streaming + tools + thinking**: buffer until pattern resolution; reasoning streams INCREMENTALLY on the tools
  path (`.hold_thinking` + `unstreamedReasoning`, never a resend); the think gate scans with a CURSOR (`ThinkScan`).
- Thinking-off is enforced in the PROMPT; generated reasoning is ALWAYS delivered (every site splits via
  `splitThinkBlock(text, true, …)`).

## Loop stops

A short exact cycle convicts on SPAN (`degenerate_loop_min_span` 128; a 24-wide map row is legit), near-repeat needs
THREE low ratios incl. PROGRESS (1024-token window, `near_repeat_min_span` 4096), long-period tier 9..64 at 10 reps.
Cuts are intentional stops: `finish_reason "stop"`, `finish_details:{"type":"repetition_loop"}`, `[loop-stop]`
logged, non-streaming trimmed to the span start. Guard: `tests/test_loop_stop_signal.sh`.
