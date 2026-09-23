# Engine: MTP speculative decoding (native head, opt-in `--mtp`)

How the Qwen3.8-Flash-Next native MTP head drafts and verifies: the head's inputs, the spec-verify invariant, draft
re-scoring, the measured round-cost table, head KV and norms. Read this before touching `src/mtp.zig`,
`src/mtp_*.zig`, `src/round_cost.zig` or the MTP orchestration in `src/generate.zig`.

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
  streaming ([engine-expert-streaming](engine-expert-streaming.md)). MiMo's 3 MTP layers are not loaded.

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
- Persistence is OPT-IN (`SUSHI_ROUND_COST_PERSIST=1`); an A/B with the table live measures the TABLE, so set
  `=0` on BOTH arms. A round's wall is between round ENDS, so an interleaved prefill chunk drops the round clock too.
- **The EV seed lives on `Qwen4Mtp`** (`ev_seed_accept`/`ev_seed_m_lo`), per loaded model; publish AND consume
  decline under `SUSHI_MTP_FORCE_DEPTH`. `MtpCostProfile` comes from the runtime fingerprint
  (`g17_nax_qwen4_q4_gs64`; `SUSHI_MTP_QWEN4_PROFILE=0` revokes it); unmeasured = generic/cap-6.
- EXL3 cold-start depth cap 2 on M5 Max binds the auto path only.

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
