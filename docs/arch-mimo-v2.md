# Architecture: MiMo-V2.6-Flash (`mimo_v2`)

How the engine serves MiMo-V2.6-Flash-RL: the source checkpoint's layout, the resident trunk, the MXFP4 and EXL3
expert paths, the hybrid global/sliding attention with its ring, and the bills that follow the storage. MiMo is an
experimental TEXT-ONLY bring-up; the supported product is the MCG EXL3 pack. Read this before touching
`src/mimo_source.zig`, the MiMo arms of `src/transformer.zig`, or anything that bills MiMo's KV.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-exl3-experts](engine-exl3-experts.md),
[engine-expert-streaming](engine-expert-streaming.md), [engine-kv-cache](engine-kv-cache.md),
[engine-memory-admission](engine-memory-admission.md), [quality-kld](quality-kld.md),
[perf-baselines](perf-baselines.md).

## Product policy

- **MCG EXL3 only.** The served MiMo target is the K2.5 MCG EXL3 pack
  (`/Users/beam/llm/models/exl3/MiMo-V2.6-Flash-RL-mcg-k2.5-w12-cal`). The affine pack
  (`MiMo-V2.6-Flash-RL-affine-iq2.7`) stays on disk and servable as a BENCHMARK REFERENCE only: no fixes, features
  or tuning for it, but do not delete its serving path while it is the reference.
- The TINY K2.5 pack (`MiMo-V2.6-Flash-RL-tiny-k2.5-w12-cal`) is retired with the TINY codebook.

## The checkpoint

- 302.8 B routed-expert weights (256 experts, top-8, 47 MoE layers, hidden 4096, expert intermediate 2048) stored
  MXFP4; ~8 B dense (FP8 e4m3 with 128x128 f32 block scales, plus bf16); an MTP head. No bf16 release exists.
- 48 layers: layer 0 is a dense MLP (`first_moe_layer` = 1); 9 global layers (4 KV heads, key 192, value 128) and
  39 sliding layers at a 128-token window (8 KV heads).
- The FP8 part is `qkv_proj` + the layer-0 MLP (3.07 GB e4m3); the bf16 3.32 GB is mostly `o_proj` (in
  `ignored_layers`). lm_head and embed_tokens are bf16, 1.25 GB each, untied.
- **Loaded by the engine: text only.** The checkpoint also ships a vision tower (mimovl, 360 tensors, video via
  `temporal_patch_size` 2), an audio encoder + 20 speech-embedding tables + a separate 24 kHz audio tokenizer, 3 MTP
  layers (`model.mtp.*`, 48 tensors; ~396 MB each, 1.19 GB, draft cost dominated by the shared lm_head) and a 5-layer
  DFlash drafter (block 8, target layers 0/11/23/35/47). `mimo_source` skips `visual.`, `audio_encoder.`,
  `speech_embeddings.` and `model.mtp.`; `model.zig` sets `has_vision=false` for mimo_v2; `dflash.zig`/`mtp.zig` have
  no mimo_v2 arm.

## Source checkpoint and packs

- **Original checkpoint**: `.mxfp4_individual` streams per-expert U8 payloads directly into U32 slabs without
  changing bytes. The FP8 trunk (`qkv_proj`, layer-0 MLP) stays resident AS STORED: e4m3 codes + f32 128x128
  tile scales, served by `fp8_block.zig` (f32 decode GEMV for 1-4 rows, staged x for 5-16, one linear dequantized to
  billed bf16 scratch + MLX matmul for wider forwards). That is the checkpoint's exact math, so the KLD teacher
  carries no quantization of its own ([quality-kld](quality-kld.md#teacher-path)); sources are read-only, MTP/media
  excluded, residency billed as stored plus `server.fp8DequantScratchBytes` at prefill. All three kernels are plain
  SIMD Metal (no NAX/matmul2d), so M4 runs the same code.
- **Converted MXFP4 pack**: the private MiMo pack converter optionally restacks the same
  MXFP4 bytes into `model.layers.N.mlp.switch_mlp` U32 weights + U8 e8m0/32 scales, without biases, and prepares
  the trunk ahead of time. Both source layouts use the same streaming kernels.
- **Packed QKV is rank-local**: each rank's FP8 rows and their scale blocks are written straight into global
  Q/K/V (`[Q_rank | K_rank | V_rank]` regrouped without dequantizing). Extra scale rows belong to partial
  rank-local tiles, not trailing padding on the full tensor. Tensor-parallel 4 is solved from geometry.
- **A MiMo EXL3 pack serves RESIDENT**: see [engine-exl3-experts](engine-exl3-experts.md#mimo). The trunk takes
  the source loader (`usesMimoSourceTrunk`), billed as stored by `mimoSourceResidentBytes`.
- **The weight loader is ONE decision** (`model.loadWeightsForConfig`): a MiMo pack read without its source trunk
  binds the raw FP8 fused QKV and its logits stop following the routed experts (two packs sharing a hard-linked
  trunk produced bit-identical logits until `kld` took the served loader).
- **Stored-affine trunk**: a SERVED pack stores o_proj, lm_head and embed_tokens as affine triples (8-bit g64,
  imatrix-weighted for o_proj and lm_head, written by the private converter); the loader serves and bills them as
  stored (lm_head via quantized matmul, embed via the quantized row gather), with no load-time step. The source
  checkpoint stores them bf16, so the teacher keeps bf16. Contract: [pack-format](pack-format.md). Measured on the
  MCG K2.5 w12 pack against the load-time product: decode 44.2 vs 44.0 tok/s, 16x512 KLD 0.07793 vs 0.07761 (NLL
  and top-1 slightly better), same 94.32 GB bill, same boot time ([quality-kld](quality-kld.md#mimo)).
- History: the load-time `trunk_quant` policy (4cb68cc..a1fb67f, MLX's minmax packer at every load) measured on the
  MCG K2.5 w12 pack (kv8, no MTP, ctx 32768): decode 32.4 (bf16 trunk) -> 39.0 (FP8 native) -> 43.5 tok/s
  (+ o_proj affine-8); resident 107.03 -> 103.96 -> 102.45 -> 101.28 GB (+ lm_head, embed); 16x512 KLD to EOS
  0.07761 / 0.07756 / 0.07745 / 0.07761 (product). Affine-8 for the FP8 linears instead: 43.3 tok/s, 0.07774 —
  no faster, lossy, not shipped. FP8-native vs old bf16-rounded teacher: 0.0034 KLD. A config still carrying
  `trunk_quant` is refused (`TrunkQuantRetired`). Details: [perf-baselines](perf-baselines.md#mimo-decode).

## Geometry and math

- `hybrid_layer_pattern` 0 = global, 1 = sliding; read heads, KV heads and K/V widths per layer. Rotate only the
  first `int(head_dim * partial_rotary_factor)` channels (64 dims); multiply V by `attention_value_scale` BEFORE
  caching. Attention scale is on the 192-wide key; two rope bases.
- **Routing/sinks**: sigmoid routing uses f32 inputs/weights, selection-only correction bias and unbiased
  normalized scores (1e-20 normalization). A sink is an extra softmax denominator column, not a real key; its
  presence follows the layer type (sliding layers only).
- mlx-lm's MiMo support (upstream PR 1219) agrees with this engine on every mechanism except that it computes the
  router matmul in bf16 (changes the top-8 set for 2.8% of tokens per layer; patch it to f32 before using it as a
  cross-check). `attention_chunk_size` is read by nothing in the engine; whether the reference's chunked attention
  differs from a plain sliding band is an open question.

## Expert streaming and imatrix

- `first_moe_layer` preserves absolute layer indices while excluding dense prefix layers from expert slabs and cache
  budgets. MXFP4 has six operands in nine stable component slots; absent biases acquire no slab or lease. MTP
  remains refused while streaming. Streaming engine: [engine-expert-streaming](engine-expert-streaming.md).
- **Imatrix** keys by ARCH (`imatrix.Arch.mimo_v2` → `model.layers.{L}.mlp.experts.*`, one flat entry per layer) and
  reaches the streamed QUANTIZED layer through the routing override's tap; armed, it forces the SORTED expert arm —
  the fused decode kernels never materialize the activation rows the down statistic needs. The driver lives in the
  private converter repo.
- The same capture records the trunk's dense inputs (`Collector.observeLinear`): every layer's o_proj input
  (`model.layers.{L}.self_attn.o_proj.weight`, [heads x v_head_dim]) in `mimoAttnWith`, and the final normed hidden
  as `lm_head.weight` [hidden] in `forwardMoeWith` (every row, also where a chunk skips the projection); each as
  sum(x²)/rows beside `<name>.rows`. embed_tokens has no input activation and gets no entry.

## Sliding layers: the ring

- **Sliding layers RING** (`ModelConfig.swaRingTokens`, `KVCache.setSwaRing`): they store
  `sliding_window + SWA_RING_SLACK` rows (128 + 512; a compaction copies the retained window when the slack fills),
  never the context.
  A non-zero `max_seq` into `KVCache.update` IS the ring predicate, so `slidingViewFor` may never decline the trim
  on a ringed arch.
- **A ringed entry's `offset` is LOCAL**; absolute = `base + offset` (`absSeqLen`). A clamp or trim below the
  retained window declines by NAME (`SlidingRingRewindPastWindow`) — the hot-cache restore cold-prefills, the SSD
  tier skips such an entry.
- **A hot entry holds a ringed layer's RETAINED ROWS, never the ring's capacity** (`KVCache.snapshotRetained`): the
  buffer is allocated at `ringCap` from token one, so a plain share billed and pinned rows no restore can read.
- Per token: bf16 288 KiB → 22.5 KiB, kv8 153 KiB → 12.0 KiB; ring per slot 122 MiB bf16, 65 MiB kv8.

## Attention kernels

- **Every layer PREFILLS FUSED** (`sushi_attn_pd`, qk 192 / v 128). A sliding layer's learned sink joins the online
  max and sum with no value row (template flag `SINK`; `SINK=0` compiles the global layers' code unchanged, proven
  byte-identical); `slidingPrefillFused` gates the dispatch AND `server.slidingBandScoreBytes`, so the band sheet is
  billed only where it still composes (chunks under 16 rows, `SUSHI_FUSED_256=0`). A quantized cache is read one
  DISPATCH at a time (`fusedSdpaPrefillKv`; `kr` = {begin, end, koff, kL_abs} puts every causal comparison in CACHE
  coordinates), never rebuilt whole; the sliding ring view is window + chunk rows, dequantized per view.
  Landed 2026-09-23 (668278c): 39-layer band attention 39.5 -> 20.3 ms at chunk 512, 603 -> 95.5 at 2048, 2439 -> 181
  at 4096; live prefill (back-to-back pair) 886 -> 1059 tok/s at 4k and 526 -> 603 at 64k, chunk 2048; a 500k prompt
  admits at chunk 2048 (`needed=10262 MB available=17538 MB`); 16x512 KLD 0.07747 vs 0.07761.
- **On M5 both layer kinds prefill on the matrix units** (`sushi_attn_pd_nax`, same carries, bill and slices;
  `SUSHI_ATTN_PD_NAX=0` = the SIMD kernel): global attention ~3x faster per layer, live 64k 597 -> 786 tok/s
  ([engine-kernels](engine-kernels.md#prefill-kernels), [perf-baselines](perf-baselines.md#mimo-decode)).
- **A packed-cache global-layer DECODE reads in place** (`mimoGlobalDecodeArm`): with matrix units (M5) the matmul2d
  `sushi_qkv_mpp` (`qkvMppDecodeServes`, from `QKV_MPP_DECODE_MIN_TK` = 4096 keys); without them (M4) the QSA split-K
  body over the whole causal range (`qkvAttnSplitKKernel`, from `QKV_SPLITK_DECODE_MIN_TK` = 4096 keys; 512 keys per
  split, 64-128 splits, split count a runtime value); else the dense rebuild. `SUSHI_KVQ_FORCE_SPLITK=1` takes
  the split-K arm on an M5 for A/B.
- **The arm is chosen per decode STEP from the cache's current key count**, never from the request's admission-time
  `kv_attn_fused` (`auto` resolves that from the PROMPT, so a short prompt that grew long stayed on the rebuild,
  unbilled, and at 256k drove wired memory to the limit). `--kv-attn-mode` and the per-request field no longer reach
  these layers; `SUSHI_KV_ATTN_FUSED=0` still does, and `kvDequantScratchBytes` then bills the whole-cache rebuild
  (`mimoGlobalDecodeRebuildMaxKeys`). The older SIMD `qkvAttnDecodeKernel` cannot stage gqa 16 x qk 192.
  Attention-only microbench (9 layers, kv8) vs the per-call dequant+SDPA rebuild: split-K 0.62x at 4k, 0.50x at 16k,
  0.42x at 64k, 0.35-0.36x at 512k (M5 proxy for M4; split-K is compute-bound, the rebuild bandwidth-bound);
  matmul2d is ~1.2-1.4x faster than split-K at 16k-512k (cross-run). Live 64k split-K decode not yet measured.
- Before the sliding fusion landed, the composed band+sink sheet was the biggest chunk-dependent bill term (0.17 GB
  at chunk 512, 2.28 GB at 2048), so 500k at chunk 2048 billed ~14.3 GB against ~12.4 GB of headroom. That term is
  now zero wherever the fused arm serves. Global-layer decode no longer rebuilds on M4-class GPUs (split-K, above).

## Decode dispatches

- QKV is ONE FP8 GEMV per layer with three outputs (`fp8_block` `gemv3`); V leaves it already multiplied by
  `attention_value_scale` (`RowSplit.v_scale`, rounded to the output dtype first, as the composed multiply did).
- Every residual add runs in one kernel with the norm that reads its sum (`fusedAddRmsNormUngated`): the
  post-attention norm (`fusedAddRmsNormRouted` also emits the f32 router input), the next layer's input norm and the
  final norm. The router is widened to f32 once at load (source-trunk packs), not per forward.
- All bit-identical to the ops they replaced: greedy text and top-3 logprobs match over 2x160 tokens.
- Count one decode forward's primitives with `SUSHI_DECODE_FWD_GRAPH=<path>` beside `SUSHI_DECODE_FWD_UBENCH`.
  What is left, per token: the kv8 append (2 quantize + 6 slice updates per layer, 384) and the sliding ring's
  dequant (78) are ~40%; a partial rotary copies its input before rotating (96 hidden copies).
- A joined `[Q | K]` GEMV output with one rope over both passed its unit tests but moved live logits by ~0.05
  nats at the first token, cause unfound; parked on branch `joint-rope-parked`.

## Bills (the bill follows the storage in the SAME commit)

- `kvBytesPerToken` counts the 9 global layers per token (spread over `kvPerTokenLayerCount`, never every caching
  layer), `swaRingBytes` the ring once per slot (`server.slotRingBytes`, at `kv_bits`), `swaStreamBytesPerToken` the
  chunk a prefill stages before compaction, for the layers one eval-cadence window lets coexist.
  `server.kvDequantScratchBytes` bills the kv-quant dense rebuild as ONE layer at the rows that layer stores.
- `mimo_source.countResidentBytes` bills each MoE router twice: as stored (bf16) and as the f32 copy the
  transformer loader keeps (~0.2 GB on the Flash pack).
- **The prefill chunk is chosen per request** (`perRequestPrefillChunk` covers a ringed arch): the widest rung up
  to 4096 whose admission bill fits live memory. The ungated load-time pin subtracts the hot-cache ask first and
  pinned 2048 (512 before the fused sliding prefill) at every context.
- **A ringed arch RESERVES its cache capacity up front** (`ModelConfig.reservesKvCapacity`, narrower than
  `longCtxGated`) and bills the reservation headroom and the ring: growing +25% at a time duplicated a global layer
  mid-prefill.
- Admission at kv8 against the ~12.4 GB the weights leave: 64k 2.75 GiB, 128k 3.69, 512k 9.29 at chunk 512
  (10.34 at 1024), 1M 16.76. 1M at kv8 needs 12.83 GB of KV alone and is out on this box; kv4 fits. Auto-context
  advertises about 480k; `--ctx-size 524288` outranks it.

## Evidence

- `tests/dump_mimo_v2_fixtures.py` supplies the independent HF oracle; `MIMO_V2_SOURCE` tests the downloaded Flash
  config/template. Native-byte preservation, forward parity, and live serving are separate gates; a header audit
  proves neither numerical parity nor generation. Test commands: [tests/CLAUDE.md](../tests/CLAUDE.md).
- Quality: MCG K2.5 w12 KLD 0.0776 through EOS on the 16x512 teacher; tables in [quality-kld](quality-kld.md#mimo).
