# Changelog

sushi began as a fork of [mlx-serve](https://github.com/ddalcu/mlx-serve) and was detached from it on 2026-09-17 at
mlx-serve commit `ef5e667` (two commits after mlx-serve v26.9.4). This file covers sushi's own changes since then;
earlier history is mlx-serve's, in that project's changelog.

## Unreleased

- **sushi binds `127.0.0.1:11234` by default and refuses a port that is already in use**: `sushi run` and `sushi serve` need no `--host`/`--port`, `--host localhost` means loopback, and a second server on a busy port exits with `port N is already in use` before loading a model.
- **MiMo-V2.6-Flash's prompt lookup decoding no longer slows long contexts**: a draft verify reads the 8-bit KV cache in place, row by row like a decode step, and rolls back without copying the cache. Greedy output is unchanged, and live decode at 244k tokens rises from 10 to 25 tok/s.
- **Every load logs `[pld] <on|off> (<source>)`**, and `/props` reports `settings.pld.source`; Qwen3.8 Flash Next reads `off (module spec wiring)`, since it never runs prompt lookup decoding.
- **One thinking-effort vocabulary, `off low medium high xhigh max`**: each model lists the words it accepts as `reasoning_efforts` in `/v1/models` and answers any other with a 400 naming them (Qwen3.8 Flash Next: off, low, medium, xhigh). MiMo-V2.6-Flash now thinks by default, and `sushi run <model> --think [effort]` plus the chat's `/think <effort>` set it, with the thought shown dimmed before the answer.
- **The engine is renamed sushi**: the binary is `sushi`, environment variables take the `SUSHI_` prefix, settings, logs and caches live under `~/.sushi`, and `/v1/models` reports `owned_by: sushi`.
- **sushi serves two models: Qwen3.8 Flash Next and MiMo-V2.6-Flash.** Any other `model_type` and any `.gguf` is refused by name at load.
- **MiMo-V2.6-Flash serves from an MCG EXL3 pack**: routed experts in EXL3, the FP8 attention trunk read as the checkpoint stores it, and `o_proj`, `lm_head` and `embed_tokens` as 8-bit affine stored in the pack.
- **MiMo-V2.6-Flash drafts with its three trained MTP heads** under `--mtp`; greedy output stays byte-identical to decoding without them.
- **MiMo-V2.6-Flash verifies MTP drafts faster**: draft rows routed to the same expert share its weight reads, with each row's output unchanged.
- **Qwen3.8 Flash Next serves EXL3 expert packs** (K2 to K4), resident or streamed.
- **EXL3 expert decode is faster on both models**, with bit-identical output.
- **MiMo-V2.6-Flash decodes long contexts from its 8-bit KV cache in place**, on the matrix units on M5-class Macs and through a split-K kernel elsewhere, choosing per step by cache length.
- **MiMo-V2.6-Flash prefills its sliding-window layers in one fused attention kernel** instead of building a full score sheet, and picks its prefill chunk per request.
- **`sushi run` shows prefill tok/s and the cached prefix after each turn.**
- **The original bf16 Qwen3.8 Flash Next serves from a 128 GB Mac by streaming experts from SSD.** Point `--model` at the HF checkpoint and set `--ssd-budget-gb <GiB>` (the total resident target), `--expert-cache-gb`, or the per-model `ssd_budget_gb` in `model-settings.json`. Speculative decoding is refused by name on a streamed model.
- **Two EXL3 packs on different codebooks can be loaded at once**; each model's forward now decodes with its own codebook instead of the one the most recent load installed.
- **An EXL3 pack whose routed-expert trellis disagrees with its `config.json` is refused by name at load**, rather than serving under a memory plan that under-counts the expert bytes.
- **An EXL3 pack can name the codeword window its search hashed** (`expert_quant.window`, 8 to 16, absent means 16): every decode arm masks the sliding window to the model's own width, and a width this build cannot decode is refused by name at load.
- **MiMo-V2.6-Flash serves long context at a fraction of the KV.** Its 39 sliding-window layers now keep a short ring instead of a full-length cache, so a token costs only the 9 global layers' keys and values; a prefix whose match falls below the retained window cold-prefills instead of restoring.
- **The KV cache is 8-bit by default.** `--kv-quant off` (or `4`), the per-model `kv_quant` setting and the per-request `kv_quant` field still choose another scheme; every load logs its choice as `[kv-cache] <scheme> (<source>)`, and `/props` and `/v1/models` report it as `kv_cache`.
- **An explicit launch flag now outranks `model-settings.json`**: `--mtp`/`--no-mtp`, `--kv-quant`, `--ctx-size` and `--mtp-typical`/`--mtp-tokenv3` win over the model's `mtp`, `kv_quant`, `ctx_size` and `mtp_acceptance`; each load logs `[mtp] <on|off> (<source>)` and `/props` reports `settings.mtp.source`.
- **`sushi kld` scores a resident MiMo pack through the model the server serves.** It loads the source FP8 trunk like every other path, so two packs that differ only in their routed experts no longer compare identical.
- **MiMo-V2.6-Flash prefills long prompts faster on M5-class Macs**: its global and sliding attention run on the GPU's matrix units (`SUSHI_ATTN_PD_NAX=0` restores the previous kernel), and the previous kernel itself is faster on every Mac.
- **Qwen3.8 Flash Next attends 16 or more query rows on an 8-bit KV cache without rebuilding the whole cache where that is slower**: a prompt's final span, short follow-up turns and wide verify blocks read the packed cache directly, and prefill chunks below 8k keys gather instead of taking the dense-mask path.
