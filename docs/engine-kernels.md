# Engine: custom Metal kernels

The rules every custom kernel in this engine follows: which decode, MoE, prefill and verify kernels exist and when
each engages, how a kernel is proven correct, and the Metal/NAX pitfalls that cost days. Read this before writing or
changing any `mlx_fast_metal_kernel` source or an eligibility predicate.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-exl3-experts](engine-exl3-experts.md),
[engine-mlx-gotchas](engine-mlx-gotchas.md), [engine-qsa-long-context](engine-qsa-long-context.md),
[engine-kv-cache](engine-kv-cache.md), [perf-baselines](perf-baselines.md).

## Decode kernels

- A custom kernel can be LATENCY-bound rather than op-bound (qwen4 fused hc read, `MLX_SERVE_HC_FUSED=0`; `hcWrite`
  DEFERS into the next read).
- A kernel keyed on `batch*seq == 1` declines every verify row AND batched slot, so the grid carries the rows
  (`HC_FUSED_MAX_ROWS`/`GDN_FUSED_MAX_ROWS` 16).
- GDN decode = three fused dispatches (`MLX_SERVE_GDN_DECODE_FUSED=0`; S 1..9 bit-identity is SAMPLING).
- A dependent-kernel cut that REDISTRIBUTES a reduction into every threadgroup loses; a routing-independent chain
  the GPU already OVERLAPS is not a dispatch to fuse. Meter: `MLX_SERVE_DECODE_FWD_UBENCH`.
- Decode on this box is dispatch-gap bound: ~860 kernels per Flash-Next token, kernel time ~9.8 of ~18 ms, ~7 us
  per boundary. `MLX_MAX_OPS_PER_BUFFER` and `MLX_METAL_FAST_SYNCH` gave nothing; decode wins come from fewer,
  denser kernels ([perf-baselines](perf-baselines.md#exl3)).

## MoE kernels (affine)

- Fused gate+up made `gatherQmv` the decode default (eligibility = the kernel's OWN conditions,
  `useGatherQmvDecode`, never a model_type list).
- Down+reduce is ONE dispatch splitting each row over 8 lanes with packs hoisted (bar = fp32-truth RMS no worse than
  the composed chain). A 16-lane variant with f32 scores passes the bar but is slower at every width on MiMo; parked.
- 3-bit is a BYTE TRIPLE (`mlxserve_qpack`). MoE PREFILL uses `_gather_sort`.
- The grouped-expert NAX tile at verify widths is a measured LOSS (parked upstream).
- EXL3 expert kernels: [engine-exl3-experts](engine-exl3-experts.md).

## Prefill kernels

- `msv_attn_pd` at (qk,v) 256/256 and 192/128 — the widths MLX's steel kernel lacks (band always fused; q_len < 16
  declined); a width `prefillHeadDimFused` lists owes a dispatch at EVERY prefill site scoring at it.
- On NAX the stock sdpa is the hd-256 kernel (`naxSdpaPreferred`, `MLX_SERVE_NAX_SDPA=0|1`).
- MLX sdpa has a WIDTH WALL at hd 256 (dense causal q 6..9 ride `splitCausalSdpa`); `use_fallback` has NO fused arm
  for an hd-256 ARRAY mask (`splitMaskedSdpa256`).
- Qwen4 HC + GDN prefill fusions take the chunk WIDTH as a scalar INPUT (`MLX_SERVE_HC_PREFILL=0` /
  `MLX_SERVE_GDN_PREFILL_FUSED=0`).

## Verify lanes

- `vqmmLaneFor`: split-K M 2–7 / wide tile N≥100K / NAX m16 M 8–16; parity = fp32-dequant per width, never vs
  stock's worst element (`VerifyQmmParity`); a verify lane is never byte-identical to stock. Sub-4-bit weights fall
  outside it (a 4/5/6-bit specialisation).
- `--decode-attn-quant` (default ON, LOSSY) requants dense attention at decode AND verify.

## NAX and Metal pitfalls

- **A cooperative-tensor template arg is `metal::remove_addrspace_t<decltype(t)>`**, never `decltype(t)` (the
  macOS 27 MPP header rejects the `thread` qualifier).
- Metal JIT-compiles at first EVAL, not at apply, so an optional NAX arm is PROBED on a one-tile problem before it
  is trusted (`buildNaxGemmKernel`); a failed probe declines by name and the sorted arm serves.
- Every matmul2d tile in the engine is f16/bf16 (EXL3 expert GEMM, QSA gather, attention); int8 NAX would need int8
  activations (W8A8) and is not planned.

## Proving a kernel

- GPU parity = no-worse-than fp32 ground truth, never kernel-vs-kernel; a parity loop asserts FINITENESS before it
  diffs; every shape an eligibility predicate adopts gets its own A/B.
- Drive a parity case at the model's REAL activation magnitudes, not synthetic unit-scale inputs, and score it
  against a TRUE f32 oracle (an oracle that mirrors the kernel's own f16 stores cannot see a saturation).
- A `metal_kernel` config is cached by FULL SHAPE (`ShapeKey`); a per-token-varying TEMPLATE value is a fresh JIT
  per value — ramping values ride INPUTS.
- Threadgroup memory is an OCCUPANCY decision (≤ ~10 KiB).
- JIT vs metallib transcendentals disagree (a 16-bit domain is swept ENTIRELY, `swigluSigTable`).
- A custom kernel's signature comes from each input's ACTUAL dtype; <8-element arrays land in `constant`; every new
  kernel ships a one-shot "engaged" log + parity on the LIVE dtype.

## Reproducing MLX

- Reproducing an MLX op means reproducing its REDUCTION TREE and ACCUMULATOR; `mlx_compile` on the same math is NOT
  output-preserving; a weight-layout fusion changes which KERNEL runs; a fusion pays only if it shortens the
  DEPENDENCY CHAIN.
- A lever that pays in another harness may pay for a constraint we don't have — a DEFAULT belongs to the engine that
  MEASURED it.

## Timing a kernel

- Interleave A/B kernels in ONE process (separate runs drift 15%); one-shot per-kernel ubench timings are not
  evidence (codebook-free kernels swung 20-40% between arms from clock ramp).
- A Metal System Trace: `xcrun xctrace record --template 'Metal System Trace' --instrument 'Metal GPU Counters'
  --attach <pid>`, then export `metal-shader-profiler-intervals` (the profiler under-samples short kernels).
- Every timing run takes the GPU lock and restores QoS ([CLAUDE.md, Team process](../CLAUDE.md#team-process)).
