# Engine: MTP speculative decoding (native heads, opt-in `--mtp`)

How the native MTP heads draft and verify: Qwen3.8-Flash-Next's one head and MiMo-V2.6's three, their inputs, the
spec-verify invariant, draft re-scoring, the measured round-cost table, head KV and norms. Read this before touching
`src/mtp.zig`, `src/mtp_*.zig`, `src/mimo_mtp.zig`, `src/round_cost.zig` or the MTP orchestration in
`src/generate.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [arch-qwen4exp](arch-qwen4exp.md),
[engine-qsa-long-context](engine-qsa-long-context.md), [engine-prefix-cache](engine-prefix-cache.md#spec-state),
[perf-baselines](perf-baselines.md#mtp).

## Code map

| File | Role |
|---|---|
| `src/mtp.zig` | MTP head (in-checkpoint `mtp.*` or sidecar via `resolveMtpSource`) |
| `src/mtp_acceptance.zig` | acceptance modes `exact|typical|tokenv3` |
| `src/mtp_group_planner.zig` / `src/mtp_group_cost.zig` | grouped verify planner |
| `src/mtp_qmv.zig` | M=1-exact qmv rows |
| `src/mimo_mtp.zig` | MiMo's three trained heads (`model.mtp.layers.{0,1,2}`), per-request row state |
| `src/round_cost.zig` | Measured per-model/width/KV-bucket spec round-cost table (`Transformer.round_cost`) |
| `src/generate.zig` | MTP orchestration, `commitForcedTokens` |

## The head

- The head = the checkpoint's own QSA+MoE layer over the PRE-mixer stream (`Qwen4Mtp`, `MtpHeadRef.qwen4`):
  `capture_hidden(_all)` = `[B,L,hc*hidden]`, never the mixed 2560; head row r = (stream at r, token r+1) at query
  position r+1, so QSA takes a `pos_base`.
- Per-request state is a `Qwen4MtpState` swapped onto the module (`qwen4MtpActivate` before EVERY head touch);
  nothing is module-owned, so MTP slots are not exclusive.
- `--no-mtp` gates the IN-CHECKPOINT head too (`entry.mtp` reads `mtpChoiceFor`, logged `[mtp] on|off (<source>)`);
  an explicit `--mtp`/`--no-mtp` beats `model-settings.json` `mtp`. MTP is refused while
  streaming ([engine-expert-streaming](engine-expert-streaming.md)).

<a id="mimo"></a>
## MiMo's three heads

- **What is generic and what is qwen4's.** The controller is head-agnostic (`MtpHeadRef` switches five
  operations): the round phases, verify invariant, draft rerank, acceptance modes, EV planner, round-cost table,
  depth caps and EV seed serve any head. qwen4-only: the pre-mixer hyper-connection stream as the head input, the
  mixer output as the lm_head input, QSA `pos_base`, the deferred PLE leaf, `forwardQwen4VerifyRows`, head
  persistence, the G17 cost profile and merged multi-slot verify.
- **Semantics (SGLang's multi-layer EAGLE for MiMo; vLLM runs layer 0 only).** Head k's row p =
  `eh_proj(cat[enorm(embed(x_{p+k+1})), hnorm(h_p)])` at rope position p, `h_p` the trunk's FINAL-NORMED hidden
  (`capture_hidden_all`), predicting x_{p+k+2}. Every head reads the target's hidden, never the previous head's
  output, so a round drafts d1..d3 by running head i at the round's last committed position q; head i's rows past
  q-i carry drafts and are truncated at the next round (`mimo_mtp.State.truncate`).
- Each head is a sliding (128) layer with sinks: FP8 qkv (rank-local, tp 4 solved from the 116 scale rows) + bf16
  o_proj, FP8 dense SwiGLU 16384, own `final_layernorm`, the trunk's embedding and lm_head. Its K/V live in a
  per-request `RowCache` holding the window, never a `KVCache`; the prompt appends only its last window per head.
- The `.mimo` arm maps the generic stash + merged first step onto head 0 and each later step onto head i
  (`draftStep`); the step index rides `hidden_next` (a scalar), host token ids ride `host_ids`. Depth and the free
  EV cap clamp to the head count; rounds stay solo (`mtpRoundsStaySolo`); no prefix-cache persistence (the head
  rebuilds from the prompt's last window).
- **Verify rows keep decode arithmetic** (`ForwardCtx.verify_rows`, up to `MIMO_VERIFY_ROWS_MAX` = 4 rows, the FP8
  GEMV's direct-row limit): every row's attention runs through `mimoDecodeAttn` on the keys its own decode tick saw
  (`mimoVerifyRowsAttn`), the rest of the forward is row-identical already (FP8 GEMV <= 4 rows, `mtp_qmv` affine-8,
  serial router rows, the EXL3 decode chain). A partial accept truncates the cache (attention-only trunk).
- Oracle: `tests/dump_mimo_v2_mtp_fixtures.py` renders the heads from the HF reference's own modules on the tiny
  fixture model; `mimo mtp heads track the torch rendering…` replays history, rounds, wrong drafts and rollbacks.
- **A MiMo verify row is ~45% of a forward** (~10-12 ms of 24; its own 8 routed experts), so depth pays only on
  predictable text: forced depth 3 is +27-52% on code/lists/JSON and -10 to -18% on prose; per-index acceptance on
  code 1.00/0.91/0.81 confirms the non-chained semantics. Greedy MTP is byte-identical to serial (18/18 pairs at 256
  tokens, forced and auto). Numbers: [perf-baselines](perf-baselines.md#mimo-verify-rows).
- **A MiMo verify row costs ~8-12 ms of a ~20.6 ms forward, ~75% of it its own experts** streaming at 96% of the
  read peak; attention per row, the o_proj row kernel and ~550 extra dispatches make most of the rest. Real-text rows
  share ~30% of their expert slots; grouping them (the grouped decode GEMVs) saves ~1.9 ms at 4 rows and ~1 ms at 3.
  What is left is small: one sdpa for all rows of a sliding layer (~0.7 ms at 4 rows; the global layers' split-K
  follows each row's own key count, so batching them is not bit-identical), the o_proj row kernel and the per-row
  router GEMV ([perf-baselines](perf-baselines.md#mimo-verify-attribution)).
- **MTP costs MiMo's prefill nothing measurable**: each chunk's head catch-up (three heads x the 128-row window) is
  ~6 ms per 4096-row chunk, and same-boot TTFT on vs off stays within noise from 2k to 71k
  ([perf-baselines](perf-baselines.md#mimo-mtp-prefill)). Compare prefill arms interleaved in one boot, never one
  reading per arm.
- The EV planner prices a MiMo EXL3 round with its own surface (`.mimo_exl3`, `MTP_EV_MIMO_EXL3_COSTS`: draft
  .04, verify row .44 of a forward, flat to depth 3); the generic surface prices a row at .20 and over-drafts prose.

## Spec verify invariant

- `cache.step = prompt_len + emitted`, t1 NOT in cache on entry, verify input `[t1, draft…]`, partial-accept
  correction from ORIGINAL `verify_logits[accepted]`.
- A block decoder checks its ENTRY token before drafting (`generate.tokenStops`); the token budget is a PRE-COMMIT
  invariant; a committed argmax is a `CommittedArgmax` (only `verifyArgmax` builds one, masking reserved ids).
- logprobs>0 + grammar disable spec.

## Cost and acceptance

- **A qwen4 verify row is BYTES, not dispatches**: a second row's own experts are read, so a depth-2 round ≈ 2
  serial forwards and prose accepts ~1.0. On the MCG K3 pack this no longer holds cleanly: verify rows cost
  ~4.6 ms each (28.2 / ~33.5 / ~37.5 ms at 2/3/4 rows) because routed experts are a minority of the bytes; measure
  before relying on either reading.
- Acceptance is a PROMPT-TYPE property (code ≫ prose; `SUSHI_MTP_FORCE_DEPTH=n` + `acc_idx=` on `[mtp-trace]`),
  so MTP stays opt-in.
- Rounds stay solo; two interleave, three or more go plain (`mtpRoundsStaySolo`; `SUSHI_MTP_BATCHED_QWEN4` opts
  in; `mergedVerifyDeclineReason` names the decline). Four MTP streams on MCG K3 aggregate ~85 tok/s today; a linear
  model of the measured verify-row cost predicts ~95-125 with merged verify at depth 2-3. Measure before any code.
- **Drafts shortlist on a coarse lm_head copy and re-score exactly** from the MIXER output
  (`buildRerankCoarse`/`rerankShortlist`/`fullReadoutArgmax`, `StepWant.mixed`; `SUSHI_MTP_DRAFT_RERANK=0`
  restores the full readout). A greedy target drafts the argmax (byte-identity contract); a sampled target draws
  from the re-scored top-32 (`mtpDraftStepPath`); draft temperature is per family.

## Round cost table

- **Round cost is MEASURED** per model/width/KV bucket from live single-chunk rounds (`round_cost.zig`;
  `SUSHI_MTP_COST_TABLE=0` = prior only); width trials m_lo then m_lo+1 never m_lo−1; the silicon depth row is a
  COLD-START cap.
- **The regime gate** compares the two round SHAPES at one base depth: two-chunk (draft m_lo, sync on the chain's
  confidence, maybe extend to m_hi) against single-chunk at m_lo, each as round wall over tokens. A round emits 1..m+1
  tokens, so each shape is judged on the running mean of `MTP_REGIME_MIN_SAMPLES` rounds or more; a verdict on one
  round per shape judged acceptance luck (the 40-70 ms/tok first verdicts on Flash-Next, then 128 rounds throttled).
- Persistence is OPT-IN (`SUSHI_ROUND_COST_PERSIST=1`); an A/B with the table live measures the TABLE, so set
  `=0` on BOTH arms. A round's wall is between round ENDS, so an interleaved prefill chunk drops the round clock too.
- **The EV seed lives on `Qwen4Mtp`** (`ev_seed_accept`/`ev_seed_m_lo`), per loaded model; publish AND consume
  decline under `SUSHI_MTP_FORCE_DEPTH`. `MtpCostProfile` comes from the runtime fingerprint
  (`g17_nax_qwen4_q4_gs64`; `SUSHI_MTP_QWEN4_PROFILE=0` revokes it); unmeasured = generic/cap-6.
- EXL3 packs take the chip's generic depth row (6 on M5 Max). A deeper round pays only on predictable text: at the
  MCG K3 round costs (31.2 / 36.0 / 42.4 ms at depth 1 / 2 / 3, ~5 ms per row after) and a ~20 ms serial token,
  depth 2 breaks even at ~0.53 per-draft acceptance, depth 3 at ~0.60, depth 4 at ~0.63; depth 4-6 wins only above
  ~0.85, where the model says cap 2 leaves 25-35% (computed from the recorded costs, not yet measured live).
- **Adaptive serial** (qwen4, kv >= 32k): the plan's base width is voted against the bucket's measured serial token
  (table AND this request's 16-round window must both lose by 5%, three rounds running). A serial request re-enters
  MTP when its OWN KV bucket changes: the read bucket maps a never-measured bucket back onto the switch's, so a
  request that went serial at 33k stayed serial to 91.8k (main 36ae6d0, Sushi3bpw, kv8, temp 1 thinking).

## Reproducibility

- **Auto-mode MTP output is NOT byte-reproducible** (round times pick depths → widths → kernels → greedy near-tie
  flips); byte bar = `SUSHI_MTP_FORCE_DEPTH`.
- `test_mtp_equivalence.sh` acquits divergences at serial top-2 gap ≤ 0.15 nats and boots
  `--prefix-cache-entries 0`.
- Forced-depth outputs are byte-equal to the pack's own no-MTP greedy (48/48 on MCG and MUL1 K3).

## Head KV and norms

- **Head KV**: dense by default; `--mtp-head-kv-quant` opts it into `--kv-quant` (billed at its effective width
  either way, `mtpHeadKvBytesPerToken`); a spec sidecar under another scheme is declined at restore and rewritten
  on the next commit. Head persistence with its QSA half: `tests/test_qwen4_mtp_head_persist.sh`.
- Acceptance modes `exact|typical|tokenv3` (`mtp_acceptance.zig`, per-model `mtp_acceptance`).
- **Norms**: delta-encoded head norms AUTO-FOLD at load (raw-HF heads get the `+1` repair, `mtpNormNeedsRepair` reads
  the norm's OWN negative fraction, whole-head 5% bar); publish packs FOLDED (the converter folds them).
  Quant re-solved PER WEIGHT; a sidecar's mode is solved from GEOMETRY (`quantParamsFromGeometry`); dense bf16 head
  trunks requantize at load (`SUSHI_MTP_HEAD_QUANT_BITS` 4/g64).

<a id="ple-defer"></a>
## The deferred PLE leaf

The deferred PLE leaf is filled before anything evaluates the build (`ForwardCtx.ple_defer` + `flushDeferredPle`,
set + flushed by BOTH `lazyForward` and the MTP verify build; `pleClaimSpecCapture` claims the spec slot at BUILD
time): a host token read inside the graph build serialized the build with the GPU, and a capture evaluated before
the fill saw a zero PLE.
