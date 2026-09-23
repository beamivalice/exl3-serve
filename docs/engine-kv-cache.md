# Engine: KV cache and kv-quant

What the KV cache stores, how a quantized cache is read, how it grows, and which settings give byte-stable
output. Read this before touching `KVCache` in `src/transformer.zig`, `src/kv_quant.zig`, or any attention path that
reads cached keys.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-prefix-cache](engine-prefix-cache.md),
[engine-qsa-long-context](engine-qsa-long-context.md), [arch-mimo-v2](arch-mimo-v2.md) (sliding-window ring),
[engine-memory-admission](engine-memory-admission.md), [engine-kernels](engine-kernels.md).

## Defaults and flags

- **kv8 is the engine default.** Every load logs `[kv-cache] <scheme> (<source>)`, i.e.
  `[kv-cache] <kv8|kv4|off> (<default|--kv-quant|model-settings.json>)`
  (scheduler adds `; ctx N (source)`); `/props` reports `settings.kv_cache`, `/v1/models` `meta.kv_cache`.
  `kld capture|compare` teachers stay dense; the MTP head KV is dense unless `--mtp-head-kv-quant`.
- `--kv-quant 4|8|off` (`src/kv_quant.zig`, `configuredKvQuantFor(config)`); an explicit flag beats
  `model-settings.json` `kv_quant`, which beats the default. A per-model setting of `kv_quant: off` still wins over
  the default.
- `--kv-attn-mode auto|dense|fused` picks the packed-read arm; `--decode-attn-quant` (default ON, LOSSY) requants
  dense attention at decode AND verify.

## The kv-quant contract

- Attention reads every cache through `KVCache.denseView` on EVERY path (batched included). The returned
  `DenseKVView` carries the packed triple when the cache is quantized (`has_quant_triple`), so a kernel either reads
  it packed or gets the dense rebuild; nothing else touches the storage.
- Schemes extend via enum + two switch arms; a prefix-cache entry records its scheme (`Entry.quant_config`), so a
  slot running `kv_quant=4` never restores from an entry committed at another width
  (`tests/test_kv_quant_per_request.sh`).
- Packed reads are kernel-or-DENSE per WIDTH (`kvAttnFusedEligible` t_q==1, `kvAttnVerifyEligible` t_q 2..8; verify
  kernel OFF on G17, `SUSHI_KV_ATTN_VERIFY=1|0`; floor 2048). Guard: `tests/test_kv_quant_fused_equivalence.sh`.
- Arch-specific packed readers: QSA on qwen4_exp ([engine-qsa-long-context](engine-qsa-long-context.md)); the
  matmul2d decode kernel and the fused prefill on MiMo's global layers ([arch-mimo-v2](arch-mimo-v2.md#attention-kernels)).
- `server.kvDequantScratchBytes` bills a dense rebuild as ONE layer at the rows that layer stores (per forward width
  on QSA).

## Growth and lifetime

- KV growth is PROPORTIONAL (`nextCapacity` +25%, capped 8192). Past 32k a request reserves its capacity up front
  (`KVCache.reservedTokens`); a ringed arch reserves always (`ModelConfig.reservesKvCapacity`).
- **A fallible re-init BEHIND a `deinit` leaves a freed object on the error path** — build first, then swap
  (`KVCache.reinit`). A handle freed before a fallible op is reset AT the free (`updateDense`).
- A lazily copied side-channel state is not in the residual's graph: name the owned copy in the cadence eval vector
  (`evalCadencePoint`, the `conv1dWithCache` tail).

<a id="gdn"></a>
## GDN trunks

A GDN trunk's `KVCache.step` is 0 forever (it advances on layer 0, a linear layer): batched rope offsets read the
slot's `moe_seq_offset`; the pad-waste cap reads `KVCache.kvLenForBatching`. Batched N=2 acquits near-ties
(≤ 0.15 nats).

## Byte stability

- INT4 long-greedy divergence is legit (the AR/verify INT4 kernel float-noise tail).
- Byte-stable greedy ⇒ no spec + `--kv-quant off/8` + `--prefix-cache-entries 0`.
- A restore is not bit-identical on a hybrid ([engine-prefix-cache](engine-prefix-cache.md)).
