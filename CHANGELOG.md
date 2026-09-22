# Changelog

## Unreleased

- **The original bf16 Qwen3.8 Flash Next serves from a 128 GB Mac by streaming experts from SSD.** Point `--model` at the HF checkpoint and set `--ssd-budget-gb <GiB>` (the total resident target), `--expert-cache-gb`, or the per-model `ssd_budget_gb` in Model Settings; the app shows an `SSD` badge and an SSD budget row for such a checkpoint. Speculative decoding is refused by name on a streamed model.
- **A load the server refuses by name no longer restarts it from the tray**; the selection reverts to the model still being served and the refusal shows as an error card.
- **Two EXL3 packs on different codebooks can be loaded at once**; each model's forward now decodes with its own codebook instead of the one the most recent load installed.
- **An EXL3 pack whose routed-expert trellis disagrees with its `config.json` is refused by name at load**, rather than serving under a memory plan that under-counts the expert bytes.
- **An EXL3 pack can name the codeword window its search hashed** (`expert_quant.window`, 12 to 16, absent means 16): every decode arm masks the sliding window to the model's own width, and a width this build cannot decode is refused by name at load.
- **A MiMo affine expert pack serves resident.** Routed experts packed as `switch_mlp` weight/scales/biases under the arch's own nesting are recognized, load beside the source FP8 trunk, and resolve their (bits, group size) per layer and projection; such a pack is no longer reported as streaming-required.

## Exl3-serve - v26.9.5

- Repo changed to exl3-serve.

## v26.9.4 — Correctness Fixes, Chinese Translation, Benchmarks

### Highlights

- **Benchmark your Mac from the menu bar.** Run a standardized context-and-coding benchmark against the loaded model using the server's own timings. Results stay local, with optional anonymous sharing to the community benchmarks at mlxserve.com/benchmarks.

- **Qwen3.8 Flash Next now handles 32 concurrent streams.** Fixed a crash that occurred when running more than 10 streams. On an M4 Max, 32 streams can now decode at **185 tok/s aggregate**.

- **`top_p: 0` is now greedy.** It behaves like `top_k: 1`, selecting only the highest-probability token.

- **Invalid images now return useful errors.** Unreadable images, bad base64, and unsupported image payloads now return a clear **400 error** instead of silently disappearing from the prompt.

- **More reliable structured output.** Invalid `json_schema` requests are rejected properly, empty stop sequences no longer terminate responses immediately, and JSON output now starts and ends cleanly at the root value.

- **Better Ollama and embeddings compatibility.** Ollama's model-load handshake now works correctly, and empty embedding requests return a proper 400 instead of a server error.

- **`/props` now reports active serving settings.** Inspect the effective KV quantization, MTP, drafter, PLD, attention quantization, prefill chunking, and other settings used by the loaded model.

- **Model-less requests now use the latest loaded model.** Requests without a `model` no longer accidentally reload an older model and evict the one currently loaded.

- **Prompt-cache hits now produce identical greedy output.** Fixed a hybrid-model issue where warm and cold requests could produce different tokens due to different kernel tiling.

- **Concurrent speculative decoding is more reliable.** Fixed `generation failed` errors when sampled requests with different draft lengths shared a Qwen 3.5/3.8 verification pass.

---

## v26.9.3 — Flash Next on 64 GB, speculation for everyone, Neural Engine media

### Highlights

- **Qwen 3.8 Flash Next fits on a 64 GB Mac.** New 3.3-bit pack `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-3.3bpw`, 52 GB resident, 85.6% top-1 agreement with bf16 (70 GB pack: 89.1%). Its 3-bit layers use the fused MoE kernels, and a faster expert down-projection lifts every MoE model: 52 to 56.5 tok/s on an M4 Max (4-8 bit pack 54.3 to 56.3).
- **Flash Next speculative decoding no longer dips between 16k and 32k of context.** The verify-width sparse gather now waits for 32k tokens on unquantized KV, where it starts paying for itself.
- **Flash Next is faster on long conversations.** Sparse-attention block selection is 3x faster at long contexts, and M5 Macs prefill on the new neural accelerators (+12% at 160k). Output unchanged. (@beamivalice)
- **More context in the same RAM.** Hybrid models need 2.5 GB less for long prompts (27B: 7.5 GB), a phantom 1.6 GB charge at 512k is gone, cached conversations hold half the checkpoint data, and the planner drops a 1.6 GB score sheet. (#366 @nikolai-vysotskyi, #397 @beamivalice.) `--wired-margin-gib 6` frees another 2 GB on 96 GB Macs with a raised `iogpu.wired_limit_mb`.
- **Concurrent speculative decoding.** Flash Next drafts and verifies concurrent chats together: +19 to 30% per user at four chats, +27% at two (#412, #420, #421, @beamivalice). Dense Qwen 3.5/3.8 verify concurrent drafts in one pass (27B, M4 Max: two users 56 to 72 tok/s, four 52 to 84), MoE Qwen batches chats like dense, and the spec kernels compile at load. Settings > Concurrent requests and `/props` say whether a model batches; the log names why a request runs alone.
- **Image, video and music generation on the Neural Engine.** `--ane-image`, `--ane-video`, `--ane-audio` (Settings > Neural Engine) split each Krea, MiniMax-H3 or ACE-Step step between GPU and ANE: 1.3x Krea, 1.33x ACE-Step, 1.22x per H3 step on an M4 Max, more on smaller GPUs. Off by default; refused by name when the Mac cannot hold it.
- **Apple Intelligence in the model picker.** macOS answers on-device: no download, no server. Tools work; 4k window, no thinking. Shown only when Apple Intelligence is on.
- **Structured output thinks first.** A JSON schema no longer turns thinking off, on every format we serve; a tag inside a JSON string stays data. (#407 @perretv)
- **Idle models unload.** `--idle-evict-secs N` (Settings > Server > Unload idle models, off by default). (#398 @latent-variable)
- **K2-Horizon 7B runs natively.** `mlx-serve pull k2`. Thinking at all three efforts, JSON schema, tool calls, 512k window. Terminators declared only in `generation_config.json` now stop generation.
- **Spark-X2.5 1.7B/4B run natively.** `mlx-serve pull spark`. Thinking, tools, 1M context.
- **Greedy sampling is greedy again.** `top_k: 1` and `top_p` near 0 cut by rank, so a bf16 tie no longer samples among tied tokens. With `top_k` set the sampler no longer ranks the whole vocabulary; the nucleus accumulates in f32.
- **Speculation survives temperature.** Flash Next draws sampled drafts from the 32 candidates it already scores exactly: at two chats the speculated share goes from 19% to 98%. Greedy unchanged.
- **`--kv-quant turbo2` / `turbo4` are gone.** A third slower than affine on long prompts and no fused path ever read them; `4` and `8` are the schemes.
- **Concurrent sampled chats share one verify pass on Flash Next.** One filtered block of verify probabilities and one GPU submission per round instead of one each, so a later request in the group no longer waits on the earlier ones. (#434)
- **Sampling with both `top_k` and `top_p` set makes one pass** over the vocabulary instead of two. Byte-identical.
- **Speculative decoding can no longer emit a reserved token.** At temperature 0 the MTP, drafter and DFlash verifies skipped the reserved-token mask every other path applies.
- **Sampled speculation can trade exactness for speed.** Typical or TokenV3 acceptance per model (Model Settings > MTP acceptance, or `--mtp-typical 0.2` / `--mtp-tokenv3 0.95`): 7 to 27% faster decode at temperature 1 on an M5 Max, 1K to 1M. Off by default. (#427)
- **Flash Next prefills in 8192-token steps when memory allows.** +8% at 16K to 128K, +13% at 350K on an M5 Max, ~3 GB more peak. (#423)
- **FLUX.2 klein base takes `guidance_scale` and `negative_prompt`.** Distilled klein packs unaffected. (#298)
- **OpenCode 2 dashboard.** `mlx-serve launch opencode2` installs a plugin showing tok/s, memory, model and prefill progress. Needs `--metrics`. (#387, #396, @beamivalice)
- **Long pastes fold** at 15 lines, and the transcript no longer renders blank after switching chats. (#414 @lojza3d)

### Changes

- Qwen3.8 Flash Next processes prompts faster by fusing hyper-connection and GatedDeltaNet prefill operations.

- Restarting the server now reuses the whole of a long conversation from the SSD cache again. A text prompt that happened to contain the id the model uses for images made the disk cache treat the conversation as if it began there, so a 73k-token chat resumed from 16k and spent 34 seconds re-reading itself instead of 1.6.

- New app icon. The tray footer is four tiles like the media row, and the power glyph is a red Quit.
- The launcher offers a plain Shell beside the coding agents, on this Mac and in the sandbox.
- A plain Shell terminal opens on click in the working folder from Settings; the coding agents (pi, opencode, Claude Code, …) ask which folder to work in.
- `--mtp-head-kv-quant` lets Flash Next's speculative head store its cache at the model's `--kv-quant` precision, about 1 GB saved at 1M tokens with no measurable loss in acceptance. Off by default.
- A reply cut for repeating itself ends with `finish_reason: "stop"` and `finish_details: {"type": "repetition_loop"}`, not a token-limit look-alike. (#327)
- Stopping a turn mid-thought or after a tool result leaves a footer (time, Regenerate, delete); the trash under an agent reply removes the whole turn. (#426)
- Voice recordings and audio attachments are saved beside the chat as 16 kHz WAV. (#430)
- Flash Next with MTP and the SSD cache no longer answers a random 500 `generation failed` on a later request: a failed draft-state write to disk left its error latched for the next decode tick, and a head with no raw history was written at all. A raise past the checkpoints no longer frees the flush record twice. (#435, #436 @Sinojen)
- Restarting the server reuses a whole long conversation from the SSD cache again; a prompt containing the image placeholder id used to truncate the restore (73k chat resumed from 16k).
- New app icon; tray footer is four tiles; the power glyph is a red Quit.
- The launcher offers a plain Shell beside the coding agents, local and sandboxed. Shell opens in the Settings working folder; agents ask which folder.
- `--mtp-head-kv-quant` stores Flash Next's speculative-head cache at `--kv-quant` precision: ~1 GB saved at 1M tokens, no measurable acceptance loss. Off by default.
- `--ssm-checkpoint-max` defaults to 16 (was 32).
- The speculative cost table is not saved across restarts unless `MLX_SERVE_ROUND_COST_PERSIST=1`.
- The Neural Engine compile cache is capped by free disk, so a full disk no longer ships a half-built offload.
- Embedded engines: llama.cpp v0.4.0 (b10809), ds4 September 14 head. ds4's own GGUFs (DeepSeek V4.1 Flash, GLM 5.3 Flash, Qwen3.8 Flash Next) route to ds4.
- GGUFs with their own MTP head speculate on greedy and sampled requests (`--mtp` default on, `--no-ds4-mtp` opts out): 35 to 47 tok/s on the Flash Next Q2 pack, M4 Max.

### Fixes

- A file of near-identical lines (tile maps, bitmaps) is no longer cut as a repetition loop; the guards now need a much longer run before cutting. A real loop still ends within a minute.
- Claude Code's hook output arrives as a second system message; Qwen's template refused it and fell back to a generic format. It is folded into the system prompt.
- A reasoning budget ends the thought instead of hiding it: at the budget the think block is closed and the model answers, as vLLM and SGLang do. `reasoning_effort` maps to a budget on Qwen 3.8 again; `reasoning_budget_tokens`, Anthropic `thinking.budget_tokens` and `--reasoning-budget` are enforced while decoding.
- `reasoning_effort` budgets follow pi's ladder: minimal 1024, low 2048, medium 8192, high/xhigh uncapped.
- pi and opencode compact correctly on small windows; their 200k-sized reserves are now scaled to the window.
- Launching an agent below its context floor warns (Claude Code 64k, opencode 32k, pi 16k) instead of failing quietly.
- Agent launchers give half the window as output budget, not a quarter; thinking shares it and a 6144-token budget came back empty.
- A row of identical short tokens (a map row of `1`s, a zeroed array) is no longer cut as a loop; a short cycle must run 128 tokens first.
- ds4 GGUF models keep one session per model, not per request: four 128k requests no longer take 60 GB extra, and repeated prompts reuse their prefix.
- Embeddings on a GGUF model return a named 400 instead of crashing.
- Flash Next agent sessions sharing a long system prompt no longer re-read the whole conversation every turn. (#390 @d-b)
- Split embedding batches returned wrong vectors after the first chunk. (#403 @josk0)
- An MLX error while writing the KV cache fails that request instead of crashing later. (#405 @josk0)
- The app can reach a server bound to a specific LAN address. (#389 @t2tx)
- Reloading a model no longer leaks its tokenizer, config and chat template.
- Status polls (`/props`, `/api/tags`, `/api/show`) no longer load a model or reset the idle-evict clock.
- Tool arguments whose items repeat a key with the same value coerce to the declared type again. (#402)
- Schema-constrained answers no longer stall on whitespace and end as an empty `length` reply.
- OpenCode 2 launches again; it was passed a `--model` flag it does not have.
- The M5 sparse-attention kernels self-check against the stock path at load and fall back on mismatch.
- The SSD cache and ANE compile cache measure free disk as Finder does; `df` hid on-demand space, so 117 GB read as 36 and nothing was written.
- GGUF models on the embedded engines report `reasoning_tokens`.
- A model re-uploaded with fewer shards than its index lists loads again (Gemma 3 12B). Regressed in 26.8.11.

## v26.9.2 — Per-model settings, chat providers, faster Flash Next

### Highlights

- **Every model can have its own settings.** Right-click a model in My Models > Model Settings to give it its own context size, KV cache precision and speculative-decoding default. They apply every time that model loads, and a model that is already running picks them up on the spot. Headless: `~/.mlx-serve/model-settings.json`.
- **Chat with other servers from the same picker.** Settings > Providers takes any OpenAI-compatible chat server (a cloud API, another Mac, a local runtime) with its key, and its models appear in the model picker as `<model>@<name>`. Chat only for now; provider models are never shared over the LAN. Headless: `~/.mlx-serve/providers.json`.
- **Qwen 3.8 Flash Next is faster across the board.** Speculative decoding costs less per step, reaches full speed on the first request instead of the fifteenth, and picks its draft from a small shortlist instead of reading the whole vocabulary. Long prompts process faster too. On an M4 Max the headline goes 83 to 93 tokens per second, with +50% at short prompts and +30% at 64k and 128k. Generated text is unchanged.
- **Long Flash Next conversations can live on SSD.** With `--prefix-cache-disk`, memory holds the model and the conversation you are in; every other conversation is written to disk in the background and comes back in seconds instead of minutes when you return to it.
- **Speculative decoding knows when to stop helping.** Past some conversation length a speculative step costs more than it saves; Flash Next now measures both and switches speculation off there and back on when it pays again. `--max-mtp-ctx <n>` sets a hard cutoff if you want one.
- **Structured (JSON schema) output at full speed.** Constrained decoding used to crawl at about one token per second on Flash Next; it now runs at the model's normal speed. (#380)
- **The chat reads the way you want.** Pick Narrow, Medium or Wide from Settings or F1 to F3. Your own messages get the same hover actions as replies, photos lay out in a grid, the reasoning block collapses out of the way and shows how long the model thought, tables no longer squeeze their headers, and numbered lists, quotes and inline code render properly. (#339, thanks @lojza3d)
- **My Models shows how much disk space is left.** (#328, thanks @justinluque)

### Fixes

- Claude Code no longer loses its SessionStart hook output, `CLAUDE.md` or any other context a client puts in a `system` message inside `messages` on `/v1/messages`. It was discarded without a warning, so the reply looked plausible on a third less prompt. (#365, thanks @nikolai-vysotskyi)
- A `developer` message is read as the system turn instead of being dropped for an unknown role.
- Mage-Flow Edit loads again. It had been refused for a missing vision tower, which the loader was dropping before the backend saw it.
- Gemma 4, LFM2.5-8B-A1B and Muse-Glimmer no longer show their thinking as the answer when streaming. When the model opened its own thought rather than the prompt template, the streamed reply carried the whole chain of thought as text while the same request unstreamed split it correctly.
- A reply cut off in the middle of a character (an emoji, an accented letter) no longer makes the whole response unreadable to the client.
- Pasting binary data into a chat (for example `grep -a` output) no longer silently cuts the conversation short at that point, and no longer knocks the prompt back to a generic format. The model used to answer with nothing and the agent's turn ended empty.
- Tool calls survive tricky arguments: a value that happens to contain the tool format's own closing tags, a call cut off mid-way, or a number-like value such as `0755` is now passed through as written instead of being emptied, dropped, renamed or turned into a number.
- Streaming and non-streaming replies agree in more places: a spent reasoning budget no longer leaks the rest of the thinking into the answer, a tool-calling reply keeps the sentence the model said before the call, `stop` sequences cut at the exact match, a client that hangs up is reported as a disconnect instead of a token limit, and `seed: -1` means unseeded instead of crashing the server.
- A long hybrid-model conversation no longer suddenly re-reads its whole history from scratch (nine minutes at 390k tokens). Two causes: a cached turn did not inherit the restore points of the entry it grew from, and a text turn after a run of image turns could not find any of them. Both fixed, and a miss with a matching prompt is now logged.
- Running out of GPU memory during a very long prompt no longer kills the server: that request gets an error and the next one is served. Requests are also admitted more accurately, so a long prompt that fits is no longer refused for memory it would never use, and the cache is evicted to make room instead of the request being turned away. (#353)
- Raising `iogpu.wired_limit_mb` is honoured: a 448k-token Flash Next conversation used to be refused because the guard looked at RAM other apps left free rather than the limit you set.
- The prefix cache budget follows what is actually loaded. It used to be fixed at startup against every model on the machine, so a model loaded next to a large one could keep a near-zero cache for the whole session. (#364)
- A batch of unrelated requests no longer evicts your live conversation from the cache; each workload evicts its own entries first. (#378)
- Speculative decoding on Flash Next no longer pays for a full draft round when the model's very first token ends the reply, and that prompt still lands in the cache for the next turn.
- Very long prompts no longer process at the narrowest width on a 1M-context server; the width is chosen per request.
- Long Flash Next sessions no longer end in "Failed to create Metal shared event": a small buffer per generated token was never released.
- Models with padding rows past the end of their vocabulary (Flash Next has 243) can no longer pick one; a reply used to lose a step when that happened.
- A model shipping both a drafter and an MTP head now uses the drafter, as intended; `--no-drafter` hands the round back to MTP. The drafter also samples 8 rounds before giving up on a request instead of 3.
- A stale speculative-decoding cost table from an older build is cleaned up at load instead of steering the planner away from the fastest width. (#382)
- A malformed Flash Next checkpoint is refused at load instead of served.
- `--max-tokens N` in serve mode sets the reply budget for clients that do not send one, and `mlx-serve launch claude` passes the server's real context size instead of assuming 200k.
- The server log says `auto` instead of a billion tokens when a client omits `max_tokens`.
- Show log in the image, video, audio and 3D panes opens the Server Log window, and a recommended model that ships a draft head is rated at the speed it actually runs.

## v26.9.1 — Terminals in the sidebar, 1M context, faster Flash Next

### Highlights

- **Your coding agents moved in with your chats.** The separate Sandbox window is gone. Claude Code, pi, opencode, codex and the rest now sit in the Sessions list beside your conversations. Drag to reorder, jump anywhere with Cmd+1..9, rename, pick a theme per terminal, or pop one out into its own window. Close a window and the session keeps running. Each terminal asks for a folder and mounts it at `/projects/<name>`, so you can work several projects at once. Themes live in Settings > Interface.
- **Qwen 3.8 Flash Next reads a million tokens.** YaRN rope scaling to 1,048,576 tokens, compatible with HF and vLLM and checked against the reference past the trained window (#323, thanks @beamivalice). `--config-overrides` turns it on without touching the model folder.
- **And it got quicker doing it.** Decode is up 10% on an M4 Max (63 -> 69 tok/s short, 55 -> 61 at 8.5k). Long prompts prefill a lot faster: 32k 589 -> 699, 128k 395 -> 654, 256k 267 -> 551 tok/s. Long-context decode holds up too, a 128k prompt now decodes at 47 tok/s instead of 40, and the gap grows with context (thanks @beamivalice). The first long prompt after a restart no longer takes three times longer (38k: 174 s -> 55 s). Direct-index idea from Jonathan Spangler's oMLX (jundot/omlx #3244).
- **Long agent sessions stop starting over.** Hybrid models (Qwen 3.5, 3.8, Flash Next) used to cold-prefill a 200k prompt every turn once a session got big. Four cache fixes put an end to that: checkpoints stay spread across the prompt (#307, #310, thanks @kartalbas), entries rank by what can actually be restored (#312, thanks @IridiumMaster), an entry that outgrows the budget keeps the part that fits instead of being thrown away (#330, thanks @d-b), and a restore is no longer counted twice (#326, thanks @ViRb3).
- **Image chats stay snappy.** Only the current turn's picture is decoded; earlier ones ride the cache, and swapping a picture keeps everything before it (#318, #320, #314, thanks @IridiumMaster).
- **Watch your video take shape.** Video Generation gained a live preview toggle: every denoise step sends back a small still or filmstrip for LTX and MiniMax H3. Off by default, and free when off (#208, thanks @Rhystic1).
- **Your Mac stays awake while it works** (#251, thanks @JustasMonkev). `--no-prevent-sleep` if you'd rather it didn't.
- **Make it yours.** Settings > Interface: light, dark or system, accent colour, chat text size, compact mode, and your own global shortcut for the quick launcher (#143, thanks @deanputney).
- **Lighter chat history.** Images are stored as files instead of base64 in the history, so a typical history shrinks from 1.5 MB to 80 KB. HEIC, TIFF and raw photos are converted so the model actually sees them (#313, thanks @lojza3d).
- **MiniCPM5 V3 tool calls** are understood natively, even when cut off mid-call (#315, thanks @uncle9x9).

### Fixes

- Chat templates using `|min` or `|max` on an array silently fell back to the wrong prompt format, so the model lost its stop token. Every MiniCPM5 multi-turn tool conversation hit this (#335, thanks @uncle9x9).
- JSON-schema output with thinking on returned the JSON as reasoning with empty content on `/v1/chat/completions` and `/v1/responses` (#331, thanks @perretv). A schema request now turns thinking off everywhere, like `/v1/messages` already did.
- Flash Next packs converted with `--ngram-bits 3/5/6` served noise from the n-gram table (#305, thanks @Sinojen). 2/4/8-bit packs were fine.
- A big `--prefix-cache-mem` next to a big pack could pass the load and then die on a long prompt. The cache budget is now capped by what the weights leave free.
- Flash Next prefills no longer die around 400k tokens: cache snapshots were cloning the growing sparse-attention history at every stride, tens of GB of it. It is stored once per cached prompt now (thanks @beamivalice).
- LTX video crashed the server at "Decoding video" on 26.8.11 (#321, thanks @hermitdave, @jedisct1). MLX 0.32.2 changed how 3D convolutions run and the decoder's working set ballooned; peak memory is back from 67 GB to 36 GB at 97 frames 1024x576, same speed.
- Image turns after a tool response landed at the wrong spot in the prompt on ChatML models.
- Sparse-attention RoPE was missing the YaRN scale; the SSD cache fingerprint now includes `--config-overrides`.
- An LTX download whose `.partial` file vanished could never finish.
- Prose that merely mentions `<function name="...">` is no longer turned into a tool call.
- Failed terminal rows offer Start Server / Retry instead of an alert; file pickers show hidden files.
- Attachments the server cannot decode (HEIC, TIFF, camera raw) no longer drop out of the prompt silently.
- Model Browser sizes and RAM fit for quantized repos were 4x too high after Hugging Face changed how it counts packed weights. They match the real download again.