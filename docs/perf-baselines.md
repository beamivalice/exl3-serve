# Performance baselines

The recorded speed numbers for the served packs on this box, where each one's raw files live, the roofline they are
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

Measured peaks (`/Users/beam/claude-tmp/mimo-roofline/peak.json`, mlx 0.32.2, binary a73713d):

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

MiMo byte breakdown: `/Users/beam/claude-tmp/mimo-roofline/bytes.json` (qkv 5.74 GB bf16, o_proj 3.22, layer-0 MLP
0.40, router 0.10, routed K2.5 2.97, lm_head 1.25, sliding ring 0.014).

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
Raw files: `/Users/beam/claude-tmp/bench-tiny-vs/` (`fw-*`, `flash-*`, each with a `.binary.txt` stamp).

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
- kv8 A/B on the affine 4/8 pack (`/Users/beam/claude-tmp/bench-rebase-ab/`): upstream's `qkvAttnMppKernel` engaged
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

Ladder on binary a73713d, bf16 trunk, kv8 (`/Users/beam/claude-tmp/mimo-roofline/ladderB/ladder.out`; `mode=fused` =
the matmul2d packed decode, `mode=dense` = the rebuild):

| context | prefill tok/s | decode fused | decode dense |
|---|---|---|---|
| 4k | 700 | 29.7 | 29.2 |
| 16k | 622 | 29.2 | 27.3 |
| 64k | 389 | 26.0 | 15.4-17.3 |
| 128k | 241 | 13.7-14.2 | 7.0-9.6 |
| 256k | 157 | 20.6-20.7 | 10.4 |

The 128k rung reads below the 256k rung in this run; recorded as measured, unexplained (re-measure before quoting it).

Prefill chunk sweep (`/Users/beam/claude-tmp/mimo-roofline/sweep/sweep.out`): width 2048 is best (852 tok/s at 4k,
477 at 64k) against 512 (737-817, 418) and 4096/8192 (~725, ~437). That was before the fused sliding prefill, whose
composed band sheet grew with the chunk. After it (binary 7c9a5af, ctx 131072, kv8, `SUSHI_PREFILL_CHUNK`, one
boot per cell, 2048 4096 4096 2048, QoS restored, lock `dispatch-chunk`, 2026-09-24): 2048 → 4096 is 926-1173 vs
1006-1175 tok/s at 4k, 879/889 vs 814/950 at 16k, 501/543 vs 539/562 at 64k; 4096 is never slower on the mean, so
the per-request chooser keeps the 4096 cap. Head ffdfc38 choosing per request (4096 admitted every time): 4k
1106-1122, 16k 888-922, 64k 500 (ctx 528384) / 559 (ctx 131072). Raw: `scratchpad/dispatch/runs/cw*`, `pfG_*`.

Decode dispatch diet (ffdfc38 family; fwd-ubench, 4096 KV, one boot per arm, A D D A twice, lock `dispatch-chunk`,
QoS restored; raw `scratchpad/dispatch/runs/ub*`): one decode forward's primitives 1399 → 1113 non-view
(`SUSHI_DECODE_FWD_GRAPH`); 24.37/24.32/24.23/24.22 → 23.73/23.94/23.83/24.00 ms per forward (-0.41 ms, -1.7%; GPU
eval 23.56 → 23.04 ms, CPU build +0.11 ms). Greedy text and top-3 logprobs identical to the base over 2x160 tokens.
llmprobe `--bench-only` on 7c9a5af (ctx 32768, kv8, no MTP): decode 40.8 tok/s (39.7-45, contended; sustained
40.8 → 44.8), predictable 44.7, novel 44.6 (recorded 43.5, predictable 42.3, on f72f989 without lm_head/embed
affine-8). 16x512 KLD to EOS 0.07700, top-1 92.06%, resident 101.48 GB (+0.2 GB: the f32 router copy).

The bf16 trunk (b2670b6, the lossless-teacher ruling, which the served pack shares) cost the MCG/TINY pack ~15% of
decode against the affine-8 trunk it replaced (31.1 → ~26 tok/s; +2.7 GiB read per token). Hence the FP8 work:

Landed 4cb68cc..a1fb67f (measured on f72f989/3b27c11, kv8, no MTP, ctx 32768, llmprobe `--bench-only`, A B B A;
raw files `scratchpad/fp8/live/`): decode 32.4/32.5 (bf16 trunk) → 39.0/39.0 (FP8 native) → 43.5/43.6 (+ o_proj
affine-8); affine-8 for the FP8 linears 43.3/43.2 (no faster, lossy, not shipped). + lm_head and embed affine-8: load
bill 95.42 → 94.32 GB, decode not yet measured on a quiet box (microbench predicts ~45.5).
Stored imatrix affine-8 o_proj + lm_head + embed (28d8a4b, overlay pack, kv8, no MTP, ctx 32768, llmprobe
`--bench-only`, `taskpolicy -a`, lock `mimo-trunk-affine`; raw `scratchpad/trunkq/live/`): decode 44.2 tok/s (a
first run read 40.5 with 4.6 GB less free memory and a -12% sustained slide: box interference, discarded); the same
format packed at load by main (4c8367f, taken once because that product had no quiet number) 44.0. Bill 94.32 GB both;
boot to `/health` 24.0-25.5 s stored vs 25.1 s load-time: the load-time packing was not a measurable cost. FP8 GEMV runs 465-488 GB/s
at one row; o_proj via MLX affine-8 qmv only 363 GB/s (a dedicated kernel could save ~1 ms/token). Sliding-layer fused prefill with sinks, 39-layer ubench: 39.5 → 20.3 ms at chunk 512, 603 → 95.5 at 2048,
2439 → 181 at 4096.

Per-step packed decode (11a0912; MCG K2.5 w12, FP8 trunk, kv8, no MTP, ctx 81920, 2026-09-24): the global-layer arm
is chosen from the cache's CURRENT key count each step (switch logged at `Tk=4096` inside a request admitted at 3.6k
tokens); 16k decode 41.2 tok/s with auto = dense, 64k 36.8 with auto = dense (bf16-trunk ladder: dense 15.4-17.3 vs
packed 26.0); prefill 660 tok/s at 16k, 465 at 64k (chunk auto). Raw: session scratchpad `live/out.jsonl`, `server.log`.

MiMo EXL3 kernel history (n=40 readers, codebook-generic): the n=40 prefill reader took the synthetic MoE layer from
12.45 to 8.48 ms at 512 rows; the prefill scatter fused into the finish reduce added +5-9%; the n=40 decode lane
funnel cut the decode chain 20% at one row and 36% at seven; the prepared-mid dispatch took rows-1 from 0.524 to
0.485 ms. A 16-lane affine down with f32 scores was slower at every width and is parked.
