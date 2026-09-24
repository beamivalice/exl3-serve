# Performance baselines

The recorded speed numbers for the served packs on this box, the roofline they are
judged against, and the levers already ruled out. Before any A/B, find the matching baseline here and INHERIT it
(Team process in CLAUDE.md); every new number lands here, with its commit, binary stamp, QoS and lock, in the same
landing.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [quality-kld](quality-kld.md),
[engine-kernels](engine-kernels.md), [engine-exl3-experts](engine-exl3-experts.md),
[arch-qwen4exp](arch-qwen4exp.md), [arch-mimo-v2](arch-mimo-v2.md), `benchmarks.md` (release columns).

## How to read these tables

- Box: M5 Max 128 GB (`Mac17,6`), macOS 27, AC power; the desktop takes a small share of the GPU under every cell.
- Only same-session, same-methodology cells compare. MTP cells are variance (sample across boots). Absolute numbers
  drift up to 15% between sessions with no code change (MUL1 no-MTP read 58.6 and 54.8 in two sessions).
- Tools: `./tests/bench.sh` / llmprobe `--bench-only --full` (median of 3 per rung); `/bench` skill for methodology.
- Every row names its binary commit. A row without one is history, not a baseline.

## Roofline

Measured peaks (mlx 0.32.2, binary a73713d):

| probe | median | best |
|---|---|---|
| streaming read kernel, 1 GiB | 538 GB/s | 558 GB/s |
| streaming read kernel, 64-256 MiB | 519-528 GB/s | 536-556 GB/s |
| copy (read+write) | 547 GB/s | |
| bf16 GEMV 16x(16384x4096) | 560 GB/s | |
| bf16 GEMM 4096 / 8192 | 45 / 51 TFLOPS | 59 / 62 TFLOPS |

The chip is a 600 GB/s class part that delivers ~530-557 GB/s to a read kernel. Decode ceilings below use 600 GB/s,
so real ceilings are ~10% lower.

| pack | bytes per decoded token | ceiling at 600 GB/s | measured | share |
|---|---|---|---|---|
| Flash-Next MCG K3 w12 | 5.58 GB (trunk 4.01 affine-8, lm_head 0.66, routed 0.91) | 107 tok/s | 61.8 | 58% |
| MiMo MCG K2.5 w12, bf16 trunk | 13.69 GB (trunk 9.47, lm_head 1.25, routed 2.97) | 44 tok/s | ~26 | 59% |
| MiMo, FP8-native trunk (branch) | ~10.6 GB | ~56 tok/s | 39.0 | |
| MiMo, FP8 + o_proj affine-8 (branch) | ~9.0 GB | ~67 tok/s | 43.5 | |

MiMo byte breakdown: qkv 5.74 GB bf16, o_proj 3.22, layer-0 MLP 0.40, router 0.10, routed K2.5 2.97, lm_head 1.25,
sliding ring 0.014.

<a id="exl3"></a>
## EXL3 decode and prefill attribution (Flash-Next)

- EXL3 K4 resident, kv8 (2026-09-22): decode forward 18.35 ms (~56 tok/s live), prefill ~1600 tok/s at 9.6k.
  Decode is ~860 kernels per token with only ~9.8 ms of kernel time; the rest is dispatch gaps (~7 us per boundary).
  Delivered: pair GEMV ~313 GB/s, down fused ~275 GB/s, trunk 8-bit qmv ~600 GB/s.
- Prefill per 2048-token chunk: the three trellis GEMMs ~520 ms (~0.4 TFLOPS, latency-bound serial k-loop, whole
  expert re-decoded per 32-row window); rest of layer ~650-980 ms.
- Landed since: pair prepare fused into pair GEMV + simd tile reduce (decode 18.35 → 17.45 ms); half4 activation
  loads + one window table per layer (prefill 1600 → 1850 tok/s at 9.6k); the n=48 lane funnel for every non-MUL1
  codebook (197395d).
- Ruled out (do not retry without a new reason): 64-row GEMM windows (+10%, registers), k-loop unroll/prefetch, ALU
  trims in the MUL1 decode, more split-K, `MLX_MAX_OPS_PER_BUFFER`, `MLX_METAL_FAST_SYNCH`. Decode loads (630 GB/s
  alone) and decode ALU do not overlap.

## Flash-Next K3, serial (no MTP)

llmprobe `--bench-only --full`, ctx 65536, KV unquantized, MTP verified off (1.01 tok/step), one server at a time.

| pack | binary | decode 192 tok | prefill 2k | 4k | 8k | 16k | 32k | 64k | first token at 64k |
|---|---|---|---|---|---|---|---|---|---|
| affine 4/8 control | a05d15f | 65.7 | 1741 | 57.5 | | 59.4 | | 56.6 | 34.9 s |
| affine 4/8 control | 28d7fab | 63.6 | 1618 | 57.3 | | 59.9 | | 56.4 | 33.9 s |
| MCG K3 w12 (clean paired run) | 28d7fab | 60.0 → 62.4 | 1779 | 57.1 | 56.5 | 54.8 | 56.0 | 55.7 | 34.5 s |
| MUL1 K3 w12 | a05d15f | 54.8 | 1409 | 52.3 | | 51.2 | | 50.7 | 40.4 s |
| affine 4/8, MTP on (depth 6) | a05d15f | 93.4 (3.4 tok/step) | 1707 | 88.6 | | 82.3 | | 79.0 | 32.9 s |

MCG decodes 15-18% faster than MUL1 and prefills 25-32% faster (same bytes, cheaper decode ALU).

<a id="mtp"></a>
## Flash-Next K3 with MTP (MCG vs MUL1)

Binary eb458ad, one session, alternating, live cost table, llmprobe short bench (192-token decode, 2k prefill):

| cell | MCG off | MUL1 off | MCG MTP | MUL1 MTP |
|---|---|---|---|---|
| decode tok/s | 47.2 | 44.6 | 70.2 | 59.1 |
| predictable text | 47.3 | 44.4 | 90.6 | 69.8 |
| novel text | 48.7 | 44.8 | 60.3 | 50.2 |
| tokens per step | 1.02 | 1.02 | 3.69 | 4.00 |
| prefill 2k | 1733 | 1316 | 1577 | 1328 |

Verify is 89-92% of a round's wall; forced-depth round ms MCG/MUL1 on code: depth 1 31.2/34.5, 2 36.0/42.9, 3
42.4/48.6. The absolute off-MTP figures sit below the serial table's: a different session, so only the within-session
pairs compare. MCG verify ms 28.2 / ~33.5 / ~37.5 at 2/3/4 rows: ~4.6 ms per extra row.

## Flash-Next Sushi3bpw, 1M context ladder (725b76ca)

Pack `Qwen3.8-Flash-Next-Sushi3bpw`, `--ctx-size 1048576 --kv-quant 8 --mtp`, llmprobe `--bench-only --rungs
4k,8k,16k,32k,64k,128k,256k,512k,980k`, `taskpolicy -a`, lock `bench-qwen-1m-clean`, quiet box (load average < 1), fans at
max with 4 min idle before boot (the thermal protocol in [process-measurement](process-measurement.md)). Headline cells:
decode 93.8 tok/s, prefill 1906 tok/s at 2k, MTP 3.62 tokens per step (predictable 119.9, novel 82.9), and llmprobe
saw an 11.9% sustained-load slide over the 38 min run.

| context | decode tok/s | prefill tok/s | first token | tokens per step |
|---|---|---|---|---|
| 4k | 96.1 | 1702 | 2.5 s | 3.37 |
| 8k | 92.5 | 1940 | 4.2 s | 3.31 |
| 16k | 93.2 | 1949 | 8.4 s | 3.20 |
| 33k | 96.2 | 1961 | 16.8 s | 3.00 |
| 66k | 80.7 | 1945 | 33.7 s | 2.56 |
| 131k | 80.9 | 1900 | 69.0 s | 2.63 |
| 262k | 73.4 | 1826 | 143.7 s | 3.20 |
| 524k | 55.1 | 1686 | 311.1 s | 3.37 |
| 1004k | 56.6 | 1465 | 685.2 s | 3.31 |

The same ladder on a338ca2, run hot after 2 h of other GPU work with workers computing alongside, read 20-33% lower
prefill at every rung (1358 at 2k, 1127 at 1004k) and 71.8 decode at 4k. Prefill code did not change between the two
binaries, so that gap is the box. Decode also gained from d72178a (the MTP regime gate). Never compare a ladder cell
across a thermal state.

## Upstream comparison (decided: no rebase)

- Rebased onto upstream vs main 6755ff2 on the MCG K3 pack, interleaved: MTP
  decode 88.2/81.9 vs 82.0/83.1, MTP off 62.1 vs 60.4, prefill 1643 vs 1598: neutral. Four streams with MTP: 85
  aggregate both ways (our merged-verify decline past width one holds).
- kv8 A/B on the affine 4/8 pack: upstream's `qkvAttnMppKernel` engaged
  zero times (QSA caps keys at 2048); all differences were run-to-run and spec variance.
- Decision: main stays; upstream's kv8 attention kernel and grouped MTP are cherry-pick candidates later, each with
  its own certification.

<a id="qsa"></a>
## QSA wide verify (b89991a)

Per 12-layer forward on a packed kv8 cache, S=16: 42.2 → 3.8 ms at 8k keys, 13.4 → 4.1 at 64k, 24.0 → 4.1 at 256k.
Prefill gather over the rebuild beats the mask arm at every width (8k keys, S=1024: 227 → 81 ms). Live MCG
Flash-Next kv8: 68k prompt byte-identical, 33k diverges at a 0.125-nat near tie; prefill 1740 → 1763 tok/s (33k).
Declined: packed reads for wide long-context chunks (15-20% slower). Flash-Next attention + indexer is 6-13% of decode
and 4-8% of a depth-3 verify; a matmul2d QSA prototype (branch `qsa-mpp-proto`) cuts the attention part 30-45%
(~2% of a token) and is parked.

<a id="mimo-decode"></a>
## MiMo MCG K2.5 w12

Ladder on binary a73713d, bf16 trunk, kv8 (`mode=fused` =
the matmul2d packed decode, `mode=dense` = the rebuild):

| context | prefill tok/s | decode fused | decode dense |
|---|---|---|---|
| 4k | 700 | 29.7 | 29.2 |
| 16k | 622 | 29.2 | 27.3 |
| 64k | 389 | 26.0 | 15.4-17.3 |
| 128k | 241 | 13.7-14.2 | 7.0-9.6 |
| 256k | 157 | 20.6-20.7 | 10.4 |

The 128k rung reads below the 256k rung in this run; recorded as measured, unexplained (re-measure before quoting it).

Prefill chunk sweep: width 2048 is best (852 tok/s at 4k,
477 at 64k) against 512 (737-817, 418) and 4096/8192 (~725, ~437). That was before the fused sliding prefill, whose
composed band sheet grew with the chunk. After it (binary 7c9a5af, ctx 131072, kv8, `SUSHI_PREFILL_CHUNK`, one
boot per cell, 2048 4096 4096 2048, QoS restored, lock `dispatch-chunk`, 2026-09-24): 2048 → 4096 is 926-1173 vs
1006-1175 tok/s at 4k, 879/889 vs 814/950 at 16k, 501/543 vs 539/562 at 64k; 4096 is never slower on the mean, so
the per-request chooser keeps the 4096 cap. Head ffdfc38 choosing per request (4096 admitted every time): 4k
1106-1122, 16k 888-922, 64k 500 (ctx 528384) / 559 (ctx 131072).

Decode dispatch diet (ffdfc38 family; fwd-ubench, 4096 KV, one boot per arm, A D D A twice, lock `dispatch-chunk`,
QoS restored): one decode forward's primitives 1399 → 1113 non-view
(`SUSHI_DECODE_FWD_GRAPH`); 24.37/24.32/24.23/24.22 → 23.73/23.94/23.83/24.00 ms per forward (-0.41 ms, -1.7%; GPU
eval 23.56 → 23.04 ms, CPU build +0.11 ms). Greedy text and top-3 logprobs identical to the base over 2x160 tokens.
llmprobe `--bench-only` on 7c9a5af (ctx 32768, kv8, no MTP): decode 40.8 tok/s (39.7-45, contended; sustained
40.8 → 44.8), predictable 44.7, novel 44.6 (recorded 43.5, predictable 42.3, on f72f989 without lm_head/embed
affine-8). 16x512 KLD to EOS 0.07700, top-1 92.06%, resident 101.48 GB (+0.2 GB: the f32 router copy).

A bf16 trunk (b2670b6, the lossless-teacher ruling) decodes ~15% slower than an affine-8 trunk (31.1 → ~26 tok/s;
+2.7 GiB read per token). Hence the FP8 work:

Measured on f72f989/3b27c11 (kv8, no MTP, ctx 32768, llmprobe `--bench-only`, A B B A):
decode 32.4/32.5 (bf16 trunk) → 39.0/39.0 (FP8 native) → 43.5/43.6 (+ o_proj
affine-8); affine-8 for the FP8 linears 43.3/43.2 (no faster, lossy, not shipped). + lm_head and embed affine-8:
bill 95.42 → 94.32 GB.
Stored imatrix affine-8 o_proj + lm_head + embed (28d8a4b, overlay pack, kv8, no MTP, ctx 32768, llmprobe
`--bench-only`, `taskpolicy -a`, lock `mimo-trunk-affine`): decode 44.2 tok/s (a
first run read 40.5 with 4.6 GB less free memory and a -12% sustained slide: box interference, discarded). Bill
94.32 GB; boot to `/health` 24.0-25.5 s. FP8 GEMV runs 465-488 GB/s
at one row; o_proj via MLX affine-8 qmv only 363 GB/s (a dedicated kernel could save ~1 ms/token). Sliding-layer fused prefill with sinks, 39-layer ubench: 39.5 → 20.3 ms at chunk 512, 603 → 95.5 at 2048,
2439 → 181 at 4096.

Per-step packed decode (11a0912; MCG K2.5 w12, FP8 trunk, kv8, no MTP, ctx 81920, 2026-09-24): the global-layer arm
is chosen from the cache's CURRENT key count each step (switch logged at `Tk=4096` inside a request admitted at 3.6k
tokens); 16k decode 41.2 tok/s with auto = dense, 64k 36.8 with auto = dense (bf16-trunk ladder: dense 15.4-17.3 vs
packed 26.0); prefill 660 tok/s at 16k, 465 at 64k (chunk auto).

Live decode with PLD off (9c9eb92, kv8, no MTP, `/v1/completions` on code, 192 greedy tokens, `taskpolicy -a`, lock
`longctx-regress`, 2026-09-24): 44.4 tok/s at 17k, 38.7 at 72k, 34.1 at 126k, 25.9-26.4 at 244k. That is 0.1-0.4
ms/token over the forward microbench at 16k-128k keys (22.1 / 25.1 / 29.6 ms, 79a4cb4) and 1.3-2.1 ms at 244k
(37.65 ms at 262k). PLD on the same 244k prompt: 10.3 tok/s before the verify-rows fix, 25.0 after
([arch-mimo-v2](arch-mimo-v2.md#prompt-lookup-decoding)).

Prefill attention on the matrix units (`sushi_attn_pd_nax`, 2026-09-24, binary b33ec32 built 01:30, taskpolicy -a,
lock attnpd-nax; baselines on 7ed9795).
One global layer, H 64 / Hk 4, qL 2048, kv8, ms: kL 2048 8.04 -> 2.90, 4096 22.1 -> 7.18, 16384 113.0 -> 33.0,
65536 448 -> 159, 262144 2012 -> 601 (34-39 TFLOPS; the SIMD kernel 11-12). 39 sliding layers' band call, per call:
qL 512 1.96 -> 0.78, 2048 1.66 -> 0.95-0.97, 4096 2.98 -> 0.89. Live MiMo MCG K2.5 w12, kv8, no MTP, chunk 2048, one
run per cell: 4k 1150 / 1138 -> 1193 / 1170 tok/s, 64k (53882 tokens) 597 -> 786 tok/s. The SIMD kernel's K^T
staging fix on M5: 2048x16384 106.9 -> 99.4 ms, 2048x65536 448-458 -> 415 ms.
16x512 KLD (first EOS) moves ~1% on bf16 store-rounding flips alone. Same binary, deterministic runs: b33ec32 NAX 0.07801
/ SIMD 0.07700; rebased 3b15b2f (current pack layout) NAX 0.07787 / SIMD 0.07793, and the SIMD kernel with its output
scaled by 1 +- 2^-18 / 2^-17 before the store (0.066% / 0.13% of outputs flip one ulp; NAX vs SIMD flips 0.03-0.14%)
reads 0.07772 / 0.07754 / 0.07771 / 0.07836. Per call on real prefills (4 chunks to 6k keys, carries, kv8 slices, band +
sinks) the two arms' error against an f32 reference agrees to 1e-4 relative RMS; a NAX-vs-SIMD KLD delta under ~1% is
noise.

`sushi_attn_pd_nax` speed work (2026-09-24). Harness: a python replica of the dispatch chain, qL 4096, H 64 / Hk 4,
kv8 slices, arms interleaved in one process, `taskpolicy -a`, lock `lever2-attn`. One global layer, ms (this run read
~30% slower in absolute terms than the same arms an hour earlier, on a contended box; the interleaved ratios hold):

| kL | 08cec69 (before f16 P) | 3ba7272 (f16 P) | f16 P + lockstep causal, clamped loads (250M budget) | this kernel (+ 1e9 NAX budget) |
|---|---|---|---|---|
| 4096 | 11.02 | 10.38 | 8.85 | 7.71 (-30%) |
| 16384 | 74.4 | 72.0 | 61.5 | 59.4 (-20%) |
| 65536 | 457 | 421 | 364 | 357 (-22%) |
| 262144 | 2004 | 1930 | 1658 | 1581 (-21%) |

- Live prefill, served pack, kv8, no MTP, ctx 163840 (chunk 2048), same prompts, fans auto, no idle wait, A B A boots:
  main 08cec69 858 / 713 / 512 tok/s at 3.6k / 54.7k / 136.6k tokens, then this kernel 962 / 789 / 596, then main
  again 995 / 772 / 555. The main arm drifted 8-16% between its two boots. Against their mean, this kernel reads
  +4% / +6% / +12%; only the 136.6k cell clears the drift.
- At qL 2048 (the same harness), the lockstep kernel with f16 P is -20% to -23% at every kL. The 1e9 budget adds
  nothing there beyond 4k keys.
- f16 P alone buys little: -3% to -8%. The gain comes when the causal simdgroups also walk in lockstep and loads are
  branch-free. Without f16 P, those two changes gave only -2% to -8%.
- Sliding band call, per call, ms (39 layers, window 128, sinks; band keeps per-simdgroup walks): qL 512
  0.316 -> 0.299, 2048 0.576 -> 0.532, 4096 0.947 -> 0.852.
- Ruled out in the same harness:
  - a strict float P (1.4x slower);
  - an int8 correction term (costs what a bf16 one does);
  - 16x32x32 tiles (<= 2%);
  - `max_total_threads_per_threadgroup` (0);
  - fast exp2 (0);
  - 8 simdgroups (slower);
  - a larger budget on the non-lockstep kernel: slower at long kL (1e9: +3% at 64k, +13% at 256k), because K/V fall
    out of cache.

<a id="mimo-long-decode"></a>
Long-context decode, global-layer attention (2026-09-24, kv8, no MTP, prefix cache off, one boot per cell,
`taskpolicy -a`, lock `lever3-kv`).
- Main = 79a4cb4 (binary 08:09).
- New = e4dc88e (binary 09:19): 79a4cb4 plus the kernel change that landed as a338ca2; the `src/` diff is the same
  85 lines. The change runs `sushi_qkv_mpp` on 4 simdgroups with packed words prefetched in registers, and its
  output is bit-identical.

`SUSHI_DECODE_FWD_UBENCH=128` after a 2048-chunk prefill of that many keys, ms per forward:

| keys | main 79a4cb4 | new | conditions |
|---|---|---|---|
| 16k | 22.10 | | morning, no fan pre-cool |
| 64k | 25.10 | 24.65 | morning, no fan pre-cool, not adjacent |
| 128k | 27.71 | 25.53 | adjacent pair 20:07 / 20:15, fans max + 10 s, die 77.7 / 69.7 °C, load 1.7-1.8 |
| 256k | 37.65 | 33.27 | morning, no fan pre-cool, not adjacent |

Live decode, `--no-pld`, one cold request per boot:
- 244k: new 29.5 tok/s. That run had no fan pre-cool, load 2.15 at start, and a 244,232-token prompt with ctx
  303104. The inherited no-PLD baseline on 79a4cb4 reads 25.9-26.4 tok/s, but it was taken with a hot prefix cache.
- 72k: adjacent fan-gated pair, new 38.9 then main 41.9 tok/s (die 67 / 79 °C). Prefill, which never runs this
  kernel, read 706 vs 817 tok/s in the same boots. The box moved ~15% between the holds, which swamps the
  ~2% this cell expects.

Attention-only µbench (9 dependent layers, us per layer, arms interleaved in one process, two runs): 16k
133-135 -> 136-139, 64k 402 -> 321-323, 256k 1697-1782 -> 1185-1252, 512k 3591-3979 -> 2398-2670. Earlier
same-session runs had main's kernel at 1468-1481 us at 256k (~243 GB/s of a ~540 GB/s read peak).
Split-K on M5 at the same shape: 155 / 528 / 1923 / 3767 us at 16k / 64k / 256k / 512k, so it never beats matmul2d
at 8k keys or more.
Ablations at 256k show where the old kernel's time went:
- dropping both matmuls left the barrier and softmax loop at 545 us with 8 simdgroups, 217 us with 4;
- the rest was the two 16-row matmul2d calls plus loads that the per-page barriers exposed.
What did not help: 64-key pages, separate K/V tiles with two barriers, vector tile stores, transposed QK, V one page
ahead, 256 splits (-3% at 256k, worse at 16k), and one merge kernel (-10 us/layer at 2-4k only, not bit-identical).
From 4k to 16k keys the new kernel costs 3-10 us more per layer, under 0.1 ms per token.
Byte identity, no PLD: greedy serial new == main on 3 prompts (4.9k / 9.8k / 18.6k tokens, 256 generated each).
Forced-depth-3 MTP == serial on the same 3 prompts, on the new kernel rebased onto 36ae6d0.

<a id="mimo-verify-rows"></a>
Verify-row cost (binary 37d5f0d = main 7ed9795 + the MTP branch, pre-00:55 layout of the MCG K2.5 w12 pack with
o_proj/lm_head/embed affine-8, kv8, 4096 KV, `SUSHI_DECODE_FWD_UBENCH=40` with
`_S=1,2,3,4 _ROW_ARMS=1 _PROFILE=1`, one boot, `taskpolicy -a`, lock `mimo-mtp`, 2026-09-24). ms/forward, lm_head in
brackets:

| rows | prefill-shaped (main) | verify rows (decode arithmetic) | expert-grouped reads |
|---|---|---|---|
| 1 | 24.15 (1.13) | | |
| 2 | 34.14 (1.15) | 34.92 (1.36) | 36.39 |
| 3 | 46.46 (1.03) | 44.72 (1.14) | 47.85 |
| 4 | 60.04 | 59.80 (1.08) | 64.64 |

An extra row costs ~10-12 ms, ~45% of a forward (the a73713d bf16-trunk ladder had 26.7 ms: its bf16 projections
went row-serial through `denseMatmul`; the FP8 GEMV and the affine-8 row kernels share weight reads). lm_head stays
~1.1 ms at 1-4 rows. The profiled pass puts ~85% of the extra row in the MLP (+8.5-9 ms/row: each row's own 8
routed experts, ~3 GB of EXL3 bytes) and ~1.5-2 ms in attention and projections. Row-identical verify costs what the
prefill-shaped forward did. Expert-grouped reads (the first slot of an expert running every slot of that expert in
one threadgroup) lost 1.5-5 ms and were dropped.

MTP (same binary and pack, kv8, ctx 32768, `--prefix-cache-entries 0`, one boot per arm, same session;
llmprobe `--bench-only`):

| cell | serial (`--no-mtp`) | MTP auto (`--mtp`) |
|---|---|---|
| decode tok/s (192) | 44.9 | 56.0 |
| predictable / novel | 44.4 / 44.3 | 63.1 / 43.5 |
| context 0.5k / 4k / 8k / 16k | 44.4 / 43.4 / 43.9 / 43.5 | 47.8 / 50.1 / 51.8 / 44.7 (2.0-2.9 tok/step) |
| prefill 2k | 1049 | 1041 |

Same-boot A/B, 256 greedy tokens, serial vs MTP: forced depth 3 code 44.1 -> 56.4, count 44.2 -> 66.9, JSON 44.0 ->
66.5, story 44.1 -> 36.0, explain 43.8 -> 39.4, novel recipe 43.6 -> 47.1; auto (cold-start cap 2, generic EV
costs) code 59.6-60.0, count 64.5-65.1, JSON 63.3-65.3, story 40.0-42.1, explain 45.2-45.4, recipe 47.5-48.8. All 18
pairs byte-identical (plus 6/6 with the MiMo EV surface as `SUSHI_MTP_EV_COSTS=0.04,0.44,0.44,0.02`: code 59.6,
count 65.6, JSON 64.2, story 38.0, explain 44.5, recipe 49.5 — one rep, not separable from run noise). Forced-depth-3 rounds: verify ~56 ms at 4 rows, three drafts ~2.5-3 ms (pre-drafted), per-index
acceptance code 1.00/0.91/0.81, count/JSON 1.00/1.00/1.00, story 0.56-0.59/0.31-0.41/0.13-0.16, explain
0.59-0.84/0.28-0.44/0.16-0.31.

With the MiMo EV surface (binary add003d, same pack and flags, `--mtp` auto): llmprobe decode 51.0,
predictable / novel 63.8 / 43.3, context 47.7 / 49.2 / 41.3 / 46.5; 256-token A/B serial 42.8-43.2 vs MTP code 57.9,
count 62.4, JSON 62.7, story 41.2, explain 42.4, recipe 46.3, 6/6 byte-identical. That boot's box ran ~3% slower
serial and read prefill 880 with an unchanged prefill path and cold TTFT (1670 vs 1618 ms). Rebased on 9dbe85e
(forced depth 3): 6/6 byte-identical, seeded sampled requests stream == non-stream on both arms.

On main 9942e8e (head 83b564a, binary built 04:41, the served pack as stored, same flags, `taskpolicy -a`, lock
`mimo-mtp`): forced depth 3 code / count / story byte-identical, serial 52.4 -> MTP 66.6 / 84.3 /
46.2, verify 44.6 ms at 4 rows. One auto boot, llmprobe `--bench-only` MTP direct vs serial through an
`enable_mtp: false` proxy: decode 59.1 vs 49.6, predictable / novel 71.1 / 50.5 vs 49.6 / 49.9, context 0.5k / 4k / 8k
/ 16k 55.3 / 60.3 / 60.3 / 58.9 vs 49.7 / 49.4 / 48.9 / 47.6, prefill 2k 925 vs 1007 (one reading per arm: noise, see
below); auto A/B 3/3 byte-identical. That boot's serial ran 49.6, below the first boot's 52.4.

<a id="mimo-verify-attribution"></a>
Where a verify row goes on main 94f0e4b (binary built 06:52, served pack, kv8, 4096 KV prefilled,
`SUSHI_DECODE_FWD_UBENCH=40`, `taskpolicy -a`, lock `mimo-verify`, 2026-09-24). Ladder, one boot, ms/forward:
verify rows 1 / 2 / 3 / 4 = 20.62 / 28.56 / 37.16 / 49.65 (prefill-shaped 28.68 / 38.22 / 49.07); a second boot of the
same binary read 43.41 at 4 rows, so compare forward arms inside one boot only. Per kernel family from a Metal System
Trace of 1-row and 4-row forwards (per-call time x calls per forward):

| family | 1 row ms | 4 rows ms | delta | achieved |
|---|---|---|---|---|
| EXL3 pair GEMV (gate+up) | 4.52 | 16.42 | +11.90 | 437 -> 481 GB/s of slot bytes |
| EXL3 prepared down | 2.22 | 7.45 | +5.23 | 444 -> 529 GB/s |
| EXL3 mid + reduce | 0.21 | 0.28 | +0.07 | |
| router (fused kernel + per-row f32 GEMV) | 0.61 | 0.80 | +0.19 | |
| FP8 QKV GEMV (39 sliding + 9 global) | 5.03 | 5.13 | +0.10 | ~570 GB/s, flat in rows |
| affine-8 o_proj (MLX qmv -> row kernel) | ~2.7 | 3.90 | ~+1.2 | |
| attention (sdpa per row, qkv_mpp per row, merges, rope) | ~0.72 | ~2.5 | ~+1.8 | |
| KV append, norms, copies | ~0.65 | ~0.84 | ~+0.2 | |
| lm_head | 1.16 | ~1.1 | 0 | |
| GPU idle between dispatches | ~2.4 | ~4.0 | ~+1.6 | 1279 -> 1835 non-view primitives |

The extra rows are ~75% EXL3: each row's own 2.96 GB of experts at ~516 GB/s, 96% of the 538 GB/s streaming
peak, so no multi-row EXL3 efficiency lever remains. On real text 4 verify rows share experts (live forced depth 3,
per layer): 22.6 / 21.4 / 24.2 / 22.0 unique of 32 slots on code / story / count / explain, an expert in all four
rows in 36-65% of layer calls. Deduplicating them pays little because a decode GEMV slot is bound by its own FMA and
input path: a second slot in the same threadgroup costs 0.52 of a slot, dropping the MCG decode saves only 6-9%.
Grouped decode GEMVs (two slots per threadgroup, bit-identical per row), interleaved kernel bench at the served
geometry with ~22 unique of 32 slots, net us per call base -> grouped: 4 rows pair 352.5 -> 333.1, down 160.8 ->
149.0; 3 rows 267.1 -> 255.9, 122.4 -> 116.4; 2 rows 183.6 -> 183.8, 84.7 -> 84.3; rows sharing nothing 358.3 ->
370.6. Four slots per threadgroup ran 2x slower (spilled accumulators); Flash-Next's pair lost 18% at 2 rows.
Same boot, whole forward, grouping off/on alternated (6 pairs at 4 rows, 4 at 3; 20 forwards per arm; same flags,
lock, QoS): 4 rows 47.5-51.7 -> 46.8-49.3 ms, every pair faster, median -1.9 ms; 3 rows 40.7-41.0 -> 39.8-40.1,
median -0.95. Live forced depth 3 (main vs grouped, boots A B B A, two sets) moved verify by -1.4 ms on the mean with
~7 ms of boot-to-boot drift, and MTP tok/s by +2-5% on code / count / JSON / story; 16/16 greedy pairs byte-identical
to serial.
The earlier expert-grouped reads ran a leader's slots back to back: every weight re-decoded per slot, the slots'
work serialized in one threadgroup, and the bytes saved were never the bound.

<a id="mimo-mtp-prefill"></a>
MTP does not slow MiMo's prefill (7ce480f, binary 05:55, the served pack renamed `MiMo-V2.6-Flash-Sushi2.5bpw`, kv8,
`--mtp --no-pld --prefix-cache-entries 0`, ctx 81920, `taskpolicy -a`, lock `mimo-mtp-prefill`). One boot per row,
MTP on vs `enable_mtp: false` alternated ABBA, 8 tokens, median streamed TTFT:

| prompt | pairs | TTFT on / off (ms) | paired diff on - off | MTP-only prefill work (eval, on - off) |
|---|---|---|---|---|
| 2.0k (1 chunk) | 10 | 1904 / 1914 | -2.5 ms (se 9.2) | +4 ms |
| 17.0k (5 chunks of 4096) | 10 | 17285 / 17286 | +108 ms (se 119) | +16 ms |
| 71.4k (18 chunks) | 2 | 90561 / 89958 | +604 ms (se 1305) | +116 ms |

The MTP-only work is the three heads' last-window forward on every chunk (~6 ms per 4096-row chunk, <= 0.2%); the
trunk chunk with its full-hidden capture reads the same as without. Within one arm at 2.4k an earlier boot spread
1862-2248 ms, so a one-reading gap says nothing.

MiMo EXL3 kernel history (n=40 readers, codebook-generic): the n=40 prefill reader took the synthetic MoE layer from
12.45 to 8.48 ms at 512 rows; the prefill scatter fused into the finish reduce added +5-9%; the n=40 decode lane
funnel cut the decode chain 20% at one row and 36% at seven; the prepared-mid dispatch took rows-1 from 0.524 to
0.485 ms. A 16-lane affine down with f32 scores was slower at every width and is parked.

<a id="mimo-prefill-gemm"></a>
## MiMo prefill: the EXL3 expert GEMMs (79a4cb4)

Live prefill (served pack, kv8, no MTP, prefix cache off, chunk 2048, `taskpolicy -a`, lock `lever1-gemm`,
2026-09-24): 3.5k 1055 tok/s, 13k 1002, 53k 820. Share of prefill wall in the three EXL3 GEMMs
(`SUSHI_EXL3_LAYER_UBENCH=1`, ~6% overhead): 64% at 3.5k (~29 TFLOPS), 62% at 13.8k, 50% at 52.5k; prepare, mid
and reduce ~3-4%; trunk, attention and router the rest.

Kernel microbench at MiMo geometry (E256, 4096x2048, n40, MCG w12, uniform top-8, R=2048, window fill 0.815, arms
interleaved in one process, median of 7): gate/up base 7.52 ms (36.6 TFLOPS); decode ALU removed 5.87 (-22%); MMA
removed 6.78 (-10%); MMA plus x loads only 5.47 (50 TFLOPS); MLX dense f16 at the same FLOPs 4.60 (60 TFLOPS).
Decode ALU is ~22% of the GEMM, weight reads ~5%, and the device-read-x 16x32x16 MMA structure caps it near 50
TFLOPS. Flash-Next shows the same ratios. A zero-cost decode would make a 4k prefill ~13% faster; no design found
reaches any of it (ruled out in [engine-exl3-experts](engine-exl3-experts.md)).

The FP8 trunk's prefill dequant-to-bf16-scratch costs ~0.27 ms per qkv layer at M=2048 (4.36 vs 4.10 ms
pre-dequantized), ~13 ms per 2048-row chunk over 48 layers: ~0.6% of prefill. There is no FP8 MMA to read into.

<a id="exl3-decode-layout"></a>
## EXL3 decode GEMV layout (two tiles per threadgroup)

The lane-funnel decode GEMVs (n40 MiMo, n48 Flash-Next) take two output tiles per threadgroup, two k-tiles per
iteration with both loads issued first, and pointer bumps; outputs bit-identical (see
[engine-exl3-experts](engine-exl3-experts.md#kernels)). Kernel microbench: 47 chained dispatches per round, arms
interleaved, median net of a null chain, `taskpolicy -a`, lock `exl3-decode-layout`; base = the served kernels,
recorded in the research run, new arm with sources read verbatim from the commit.

| geometry, kernel | rows 1 | rows 2 | rows 4 | rows 8 |
|---|---|---|---|---|
| MiMo pair (gate+up) | 138.6 → 101.1 us | 259.9 → 188.4 | 500.4 → 359.5 | 1123.5 → 698.7 |
| MiMo prepared-mid down | 63.2 → 47.9 | 117.6 → 87.8 | 228.7 → 167.4 | 471.1 → 322.4 |
| Flash-Next pair (E=512, in-process old arm) | 43.9 → 36.6 | | 133.0 → 98.7 | |
| Flash-Next fused-mid down (in-process old arm) | 59.2 → 44.2 | | 95.4 → 65.9 | |

One tile per threadgroup with the unroll and pointer bumps reads the same as two on the Flash-Next pair at one row
(35.7 us) but loses at four rows (107.2) and on the fused-mid down (54.1 / 86.0), whose SwiGLU prepare two tiles
share: one policy, two tiles.

Live, llmprobe `--bench-only`, no MTP, one boot per arm, `taskpolicy -a`, lock `exl3-decode-layout`, greedy
200-token chat completion byte-identical between the arms of each pair:

| pack, flags | base | new | decode | prefill 2k |
|---|---|---|---|---|
| MiMo MCG K2.5 w12, stored-affine trunk (post-03:47 layout), kv8, ctx 32768 | 2161e18 | this change on 2161e18 (bc27a4e) | 44.9 → 52.0 | 916 → 1068 |
| Flash-Next MCG K3 w12 plugged, kv off, ctx 65536 | 61.8 recorded (28d7fab, `--full`) | this change on c4f3f7a (aff4f85) | 61.8 → 66.2 | 1763 → 1845 |

The MiMo prefill gain (+17%) is not attributed: prefill rows take `moePrefill`, which this change
does not touch; re-measure before quoting it.
