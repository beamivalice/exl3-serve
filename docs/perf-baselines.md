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

Same boot, ctx 8192, kv8, no MTP, 3x200-token samples (a05d15f): MUL1 → TINY (retired; the same instruction
stream as MCG) on the same K3 w12 experts: decode 37.27 → 38.17 tok/s, prefill 1k 1493 → 1766, 2k 1573 → 1825. Decode moves little because a
Flash-Next step is mostly trunk kernels and dispatch gaps; prefill moves because trellis GEMMs are a large share.

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

## Upstream comparison (decided: no rebase)

- Rebased onto upstream (branch `upstream-rebase`, 8f64acd) vs main 6755ff2 on the MCG K3 pack, interleaved: MTP
  decode 88.2/81.9 vs 82.0/83.1, MTP off 62.1 vs 60.4, prefill 1643 vs 1598: neutral. Four streams with MTP: 85
  aggregate both ways (our merged-verify decline past width one holds).
- kv8 A/B on the affine 4/8 pack: upstream's `qkvAttnMppKernel` engaged
  zero times (QSA caps keys at 2048); all differences were run-to-run and spec variance.
- Decision: main stays; upstream's kv8 attention kernel and grouped MTP are cherry-pick candidates later, each with
  its own certification. The `upstream-rebase` branch is kept as the reference.

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

The bf16 trunk (b2670b6, the lossless-teacher ruling, which the served pack shares) cost the MCG/TINY pack ~15% of
decode against the affine-8 trunk it replaced (31.1 → ~26 tok/s; +2.7 GiB read per token). Hence the FP8 work:

Landed 4cb68cc..a1fb67f (measured on f72f989/3b27c11, kv8, no MTP, ctx 32768, llmprobe `--bench-only`, A B B A):
decode 32.4/32.5 (bf16 trunk) → 39.0/39.0 (FP8 native) → 43.5/43.6 (+ o_proj
affine-8); affine-8 for the FP8 linears 43.3/43.2 (no faster, lossy, not shipped). + lm_head and embed affine-8: load
bill 95.42 → 94.32 GB, decode not yet measured on a quiet box (microbench predicts ~45.5).
Stored imatrix affine-8 o_proj + lm_head + embed (28d8a4b, overlay pack, kv8, no MTP, ctx 32768, llmprobe
`--bench-only`, `taskpolicy -a`, lock `mimo-trunk-affine`): decode 44.2 tok/s (a
first run read 40.5 with 4.6 GB less free memory and a -12% sustained slide: box interference, discarded); the same
format packed at load by main (4c8367f, taken once because that product had no quiet number) 44.0. Bill 94.32 GB both;
boot to `/health` 24.0-25.5 s stored vs 25.1 s load-time: the load-time packing was not a measurable cost. FP8 GEMV runs 465-488 GB/s
at one row; o_proj via MLX affine-8 qmv only 363 GB/s (a dedicated kernel could save ~1 ms/token). Sliding-layer fused prefill with sinks, 39-layer ubench: 39.5 → 20.3 ms at chunk 512, 603 → 95.5 at 2048,
2439 → 181 at 4096.

Per-step packed decode (11a0912; MCG K2.5 w12, FP8 trunk, kv8, no MTP, ctx 81920, 2026-09-24): the global-layer arm
is chosen from the cache's CURRENT key count each step (switch logged at `Tk=4096` inside a request admitted at 3.6k
tokens); 16k decode 41.2 tok/s with auto = dense, 64k 36.8 with auto = dense (bf16-trunk ladder: dense 15.4-17.3 vs
packed 26.0); prefill 660 tok/s at 16k, 465 at 64k (chunk auto).

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

<a id="mimo-verify-rows"></a>
Verify-row cost (binary 37d5f0d = main 7ed9795 + the MTP branch, pre-00:55 layout of the MCG K2.5 w12 pack with
`trunk_quant` o_proj/lm_head/embed affine-8, kv8, 4096 KV, `SUSHI_DECODE_FWD_UBENCH=40` with
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
| MiMo MCG K2.5 w12, load-time affine trunk (pre-03:47 layout), same flags | c4f3f7a | this change on c4f3f7a (aff4f85) | 44.0 → 50.4 | 964 → 1069 |
| Flash-Next MCG K3 w12 plugged, kv off, ctx 65536 | 61.8 recorded (28d7fab, `--full`) | this change on c4f3f7a (aff4f85) | 61.8 → 66.2 | 1763 → 1845 |

The MiMo prefill gain (+11-17%, both pairs) is not attributed: prefill rows take `moePrefill`, which this change
does not touch; re-measure before quoting it.
