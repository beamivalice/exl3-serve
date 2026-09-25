# MiMo-V2.6-Flash (`mimo_v2`) inference path — bug hunt

Branch `audit/mimo-inference-bug-hunt`, audited and fixed against `f22a3831`.

Scope: the served MiMo text path — the layer forward in `src/transformer.zig`, the FP8 trunk
(`src/fp8_block.zig`, `src/mimo_source.zig`), routing and the MTP head, and the sliding ring,
prefix cache and memory bills. Read `CLAUDE.md` and `docs/arch-mimo-v2.md` first; this report assumes both.

Method: four independent passes over disjoint areas (trunk + packed-cache decode arms; routing, MoE and
MTP; ring, prefix cache and bills; the layer-forward assembly), then every claim re-verified against the
tree before anything was changed. Two claims did not survive verification and are recorded as refuted
rather than fixed — a plausible bug report is not a bug.

## What was NOT run, and why

No KLD, no benchmark, no `tests/*.sh` that boots a server. Verification was `zig build test
-Doptimize=ReleaseFast` (hermetic, tiny fixtures) plus `zig build -Doptimize=ReleaseFast`.

**A live MiMo run was attempted and could not complete on this box.** The served resident EXL3 pack needs
`loadRequirementBytes(97.23 GB)` = weights + `min(weights/8, 6 GB)` + 1 GB = **104.2 GB**, against
**95.7 GB** reported available, so the load preflight refuses it. Retried under the GPU lock with
`--skip-mem-preflight --ctx-size 8192` — the override the guard's own comment documents as the sanctioned
path for a flagship MoE on 128 GB hardware — with a monitor aborting if reclaimable RAM fell below 6 GB.
It **aborted at 5.8 GB** (pages free + inactive) with the weights still loading, so the shortfall is
physical on this machine, not the guard's margin alone. The box recovered to 95% free and the lock was
released. Nothing was forced past the abort.

**So `tests/test_mimo_ring_reuse.sh`, `tests/test_mimo_streaming.sh`, the mimo smoke cell, and the MTP
byte-identity A/B all remain unrun.** Every live MiMo gate needs this same load, so closing the desktop
apps (or raising `iogpu.wired_limit_mb` above 104 GB) is the unblocking step. None of the four fixes
touch the ring or the hot cache; they are a dispatch decision, two loader key classifications and a
routing parity test. The rebase onto current `origin/main` was clean and the suite is green on it.

Final state on the rebased tree: 2702 tests, 2610 passed, 92 skipped, **0 failed**.

## Fixed

### 1. A verify wider than the decode arithmetic was silently re-armed (primary)

`MIMO_VERIFY_ROWS_MAX` (4) is a contract boundary, not a tuning knob. It is the widest row count for
which a verify keeps BOTH its decode-tick attention AND the FP8 trunk GEMV's direct-row arm. Past it
the forward silently took the causal prefill-shaped attention and the trunk GEMV reassociated onto the
staged-x arm, so greedy MTP output stopped being byte-identical to serial decode — no crash, no log.

Two leaks, both found independently by more than one pass:

- `Generator.mtpRoundPlanInner` returned the forced depth before the cap was computed, so
  `SUSHI_MTP_FORCE_DEPTH=5..8` planned a 6-9 row verify. `--mtp-depth` did not leak: the load-time clamp
  in `src/scheduler.zig` already bounded it to the head count.
- `mimoAttnWith`'s gate filtered on `seq_len <= MIMO_VERIFY_ROWS_MAX` and fell through to the prefill
  arm above it, which also made `mimoVerifyRowsAttn`'s own `MimoVerifyRowsTooWide` guard unreachable.

The rule, now in one place: a MiMo verify is decode-shaped at every width it serves, and a width it
cannot serve is refused by name, never silently re-armed. `mimoAttnArm` is that one place, returning a
named error instead of a different arm; the load-time clamp now also honours the verify row budget, which
is the identity for the shipped 3-head config but closes the same hole for a 4+ head pack
(`MTP_ADAPTIVE_DEFAULT_CAP` is 6, so a 4-head MiMo would have planned 7 rows).

Reachability was checked at all four `ctx.verify_rows` writers, not assumed: the PLD path bounds the
verify by the constant *and* by the match length, the MTP round is bounded by the load clamp, the
forward microbench only sets the flag inside the boundary, and the one test helper loops `{2, 4}`. So the
new refusal cannot fire on a live request.

The forced path is clamped to the verify budget rather than to the launch depth — a deliberate
departure from the original brief. Three in-tree consumers require a forced round to exceed the launch
depth: a live A/B script pins `SUSHI_MTP_FORCE_DEPTH=4` on an arch whose default depth is 3, the cost
table's own width trial plans above it, and `docs/engine-mtp.md` names the lever the byte bar for depth
measurements. Clamping to the launch depth would have silently shrunk all three.

### 2. The MiMo source loader hardcoded the dense prefix to layer 0

`mimo_source` classified the dense prefix's FP8 MLP only at `ref.layer == 0` (four sites) and refused
anything else, while `first_k_dense_replace` is config-driven everywhere else — `validateExl3Expert` in
the same file already generalizes on it. A checkpoint config with a 2+ layer dense prefix is accepted by
the parser and by the transformer, then refused at load. The failure was loud, not silent, and the
shipped Flash config has a dense prefix of 1, so this was latent. Now read from the config, as the
sibling validator does.

### 3. The verify row budget now follows the FP8 GEMV's direct-row arm

`MIMO_VERIFY_ROWS_MAX` (4) and `fp8_block.gemv_direct_max_rows` (4) were equal **by coincidence**, with
the relationship held only by a doc comment. `gemv_direct_max_rows` is a `pub var` the
`SUSHI_FP8_UBENCH=1` microbench sets to 0. Lower its default to 3 and a 4-row verify keeps the
decode-shaped attention while the trunk GEMV silently moves to the staged-x arm that reassociates the f32
sum — the MTP/PLD invariant "an accepted row is the serial row bit for bit" breaks with no crash and no
log line, which is the same failure fix 1 just removed from the attention side.

The budget is now read from the kernel (`mimoVerifyRowsBudget`) instead of restating it beside it, so
the two cannot drift apart. The test lowers the kernel's arm and pins that a width it can no longer
serve row-for-row is refused rather than re-armed.

### 4. The shipped routing combination had no bit-parity test (test-only)

The fused router kernel's comment claims its tests assert bit equality with the fallback chain, but the
one pairing MiMo ships — ungrouped `.sigmoid_bias` with an f32 output — was exercised only inside a
`MIMO_V2_MODEL`-gated fixture test. hy3's arm proves the same kernel at a bf16 output, which absorbs a
last-ulp difference between the kernel's `w / (tot + 1e-20f)` and MLX's divide chain; MiMo keeps the
routing weights in f32 into the expert sum, so nothing absorbs it.

Added, and it **passes**: bit-equal on all 2048 weights against `mimoRoutingChain`, ids equal to the f64
host reference. The fused arm is the chain's arithmetic, not a cleaner one (max deviation from the f64
reference is identical for both). The assertion was mutation-checked — perturbing the chain's scale makes
it fail — so it has teeth. No bug in the shipped routing path.

## Coverage gap found while validating (not fixed here)

- **`tests/test_mtp_equivalence.sh` cannot see a sharded pack.** It probes a monolithic
  `model.safetensors` header for exact marker keys (`mtp.fc.weight`, `language_model.mtp.fc.weight`,
  `mtp.eh_proj.weight`) and never reads `model.safetensors.index.json`. The MiMo resident pack is sharded
  and namespaces its heads `model.mtp.layers.{k}.*` (48 keys), so the probe misses and the script exits
  `SKIP`. The consequence is real: **the MiMo resident pack's MTP path — the flagship feature, and the
  one whose verify invariant fixes 1 and 3 harden — has no live byte-equivalence gate at all.** The
  check is cheap (read the index when there is no monolithic file) and belongs in a follow-up with its
  own byte bar. A manual `--mtp` vs `--no-mtp` A/B was the substitute, and it could not complete on this
  box for the memory reason above.

## Refuted (reported by a pass, did not survive verification)

- **"The decode arm can see several query rows and attend unmasked over keys ahead of each row."**
  Impossible by construction: `forwardMoeWith` defines `is_prefill = seq_len > 1` and passes that same
  value into `mimoAttnWith`, so `!is_prefill` implies `seq_len == 1`. The other two call sites are
  single-row too. The premise treated `!is_prefill` as independent of `seq_len`.
- **"The resident and streamed arms feed the router different precision."** They feed it identical
  values. The fused norm kernel's f32 output is `float()` of the already-bf16-rounded value, i.e. exactly
  the `astype(f32)` the streamed arm performs. The f32 copy saves a dispatch, not precision. Confirmed
  twice, independently.

## Recorded, fix declined

- **A hot entry pins and bills the 9 global layers' reserved buffer slack** (~106 MB at kv8: the
  reservation is `prompt + RESERVE_GEN_HEADROOM + chunk`, and `snapshotRetained` trims only ringed
  layers). Real, and the same class the ringed case fixed. **The obvious fix is worse**: trimming a
  global layer means copying it, because a ring holds 384 rows while a global layer holds the whole
  prompt — ~392 MB copied per turn to reclaim ~106 MB, on the critical path. The real fix is to not
  over-reserve for a ringed arch; that is a different change in a different place and was not attempted.
  Direction is over-bill, so there is no OOM exposure — the cost is premature LRU eviction and pinned
  waste.
- **`kvDequantScratchBytes` prices the ringed layers' rebuild at rows stored** (640) while the fused arm
  materializes a `window + chunk` view. This contradicts the term's own "one layer at the rows that layer
  stores" wording, but the aggregate bill covers it (`swaStreamBytesPerToken` is >= 4 layers x
  (window+chunk) for chunk >= 512, and the 640-row floor covers below that). A decomposition that reads
  wrong, not a hole. The project's invariant is about *under*-billing.
- **`ringEntryServes` is one row conservative** — it requires `base <= len - window` where
  `base <= len - window + 1` would serve. Refuses one available reuse, in the safe direction.

## Open question

- **Kernel-config lifetime.** The FP8/QSA/MPP config caches free a `mlx_fast_metal_kernel_config` on
  key change while MLX is lazy. This is settled for the common case (a step's logits are evaluated
  before the next step changes the key) and for the probe paths (which eval before their `defer`), but
  the one narrow window is a verify whose row range crosses a 512-key split boundary while
  `32768 <= t_k < 65536` — above 65536 the split count is clamped at 128 and stops changing. Whether
  `mlx_fast_metal_kernel_apply` copies the config at call time could not be determined here: the header
  is an opaque handle with no lifetime contract and the pinned `mlx-c` submodule is not checked out.
  Worth settling before someone caches more configs.

## Also worth knowing

- **The FP8 GEMV's three width arms reassociate the f32 sum** — direct (<=4 rows) strides each row in
  16-byte chunks per lane, staged x (5-16) gives each lane one 4-column group per 128-column tile, the
  wide arm (>=17) dequantizes weights to bf16. So a row is byte-identical to a decode tick only at or
  below `MIMO_VERIFY_ROWS_MAX`. That coupling was previously held by a comment next to two constants;
  fix 1 makes it enforced, and `docs/arch-mimo-v2.md` now records why. The arms were deliberately NOT
  made bit-identical: ~1e-7 is one more instance of the reassociation the engine already accepts between
  its prefill and decode arms, and the wide arm's bf16 dequant is documented and KLD-gated.
- **Doc freshness checked, not assumed.** The MiMo-ViT landing (`641e0c86`) does update
  `docs/arch-mimo-v2.md` from "text only" to "text + the vision tower", with `has_vision` now reading
  `c.mimo_vision`. No staleness.
- **A stale tree nearly invalidated this audit, twice.** The worktree's `main` was 16 commits behind
  `origin/main`, and those commits land in the ring, prefix-cache and vision code; the audit
  fast-forwarded before reading any code. `origin/main` then moved again (5 more commits, including an
  EXL3 prefill kernel change that touches `transformer.zig`), so the branch was rebased onto it before
  the final verification. Every line reference is against the post-rebase tree.
- The full-suite runner *does* collect `test` blocks nested inside a container body — the four in
  `ModelRegistry` and `Transformer` were each run and pass. A report claimed otherwise; it did not
  survive checking, and no dead tests exist.
