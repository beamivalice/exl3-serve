# Engine: prefix cache and SSD-first

How prompt-prefix KV reuse works: the hot RAM cache, hybrid (GDN/QSA) restore points, the SSD tier and SSD-first
mode, checkouts and donations, and spec state riding the cache. Read this before touching `src/prefix_cache.zig`,
`src/kv_disk_cache.zig` or `src/kv_disk_writer.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-kv-cache](engine-kv-cache.md),
[engine-memory-admission](engine-memory-admission.md), [engine-mtp](engine-mtp.md),
[arch-mimo-v2](arch-mimo-v2.md#sliding-layers-the-ring).

## Code map

| File | Role |
|---|---|
| `src/prefix_cache.zig` | Hot prefix cache (`--prefix-cache-entries`, `--prefix-cache-mem`) |
| `src/kv_disk_cache.zig` | SSD tier (`--prefix-cache-disk`) |
| `src/kv_disk_writer.zig` | SSD-first mode's background writer thread |
| `src/restore_dump.zig` | Prefix-cache restore diagnostics (`tests/diff_restore_dump.py`) |

## Basics

- KV reuse via prompt-prefix matching; invalidated after tool calls + pad-only gens (`commitDeclinesPadOnly`: only an
  ALL-pad generation declines); hot cache spills to SSD; RAM invalidation propagates to disk.
- Restore ALWAYS clamps (`truncate(final_len)`); a failed restore hands back an EMPTY cache; every eviction loop has
  a no-progress exit (checked-out entries are unevictable).
- **Media keys are a CHAIN** (`MediaSpan`: per block, a hash of its pixels, its position and every block before it;
  the entry key is the last). An entry keyed by a request's block k restores up to block k+1, so a turn that appends
  a screenshot reuses everything before it; any other key mismatch shares only the text before the first media row
  (`crossKeyBoundary`). The splice resumes at the placeholder count inside the restored prefix.
- **A restore is not bit-identical on a HYBRID** (≤ 0.047 nats; the chunking class ~0.3 nats top-5 for QSA state) ⇒
  byte-stable greedy needs `--prefix-cache-entries 0`. A hybrid cache hit moves the top logprob ~0.2 nats, so any
  scorer boots with the cache off.
- The always-on SSM snapshot sits 30 tokens BEFORE prompt end; a restored tail inside that window forwards as ONE
  span (`ssmSnapshotBackoff`). Guard: `tests/test_hybrid_reuse_equivalence.sh`.
- **A ringed (sliding-window) entry restores at its end or at one of its ring checkpoints**
  (`KVCache.ringCheckpoint`, `Entry.ring_cps`, the highest `RING_CHECKPOINT_MAX` = 4 kept): each ringed layer's
  window + 30 rows at a position; its own is the prompt end, taken right after prefill. A reply longer than the
  ring's slack compacts it past where the next turn diverges (the previous reply re-renders); the checkpoint's rows
  go under the ringed layers (`restoreRing`) and the usual clamp follows.
  A checkpoint restore of fewer than `RING_RESTORE_MIN_TOKENS` (64) cold-prefills: on MiMo kv8 one costs +6 to
  +41 ms over the cold prefill at 16-32 tokens, breaks even at 64, and saves ~180 ms at 256.
  Below both, `SlidingRingRewindPastWindow` → cold prefill. The SSD tier skips ringed entries; persisting the global
  prefix plus the checkpoint would lift that ([arch-mimo-v2](arch-mimo-v2.md#sliding-layers-the-ring)).
- **A commit that forked off another entry inherits that entry's ring checkpoints below the fork** (`bestRingDonor`,
  refcount-shared and billed per entry like SSM checkpoints): a request appending to the conversation (a client's
  side request: the chat + a reminder) otherwise holds only its own prompt end, and once the count cap evicts the
  main entry the next main turn, diverging where the reminder was appended, cold-prefilled every turn.

## Candidate ranking and trimming

- **Hybrid candidates rank by RESTORABLE checkpoint position, not raw match** (`findBestRestorableMatch` RAM,
  `bestHybridMatch` disk). Ringed candidates rank by `ringRestore`; an un-restorable one stays eligible at 0, so a
  lookup with nothing better still declines by name.
- Checkpoint retention thins the INTERIOR with a dense newest quarter (`spanPreservingDropIndex`, `ThinPolicy`).
- An oversized candidate is TRIMMED to the longest restorable prefix that fits (`trimLenForBudget`,
  `KVCacheSnapshot.trimmedCopy` is a REAL copy); a QSA trim bills the bank on the final retained checkpoint.
- A decline carries its `TrimDecline` reason; a RAM-budget decline spills to SSD (`spillDeclinedToDisk`).

## Budget

- A commit declines and frees its incoming snapshot when checked-out residents prevent satisfying either the entry-count or byte cap; the request continues and one `[hot-cache]` line names the limiting cap.

- **The hot-cache budget is CLAMPED at load** to what the weights leave under the GPU ceiling and is a HARD cap; it
  FOLLOWS residency (`reviseHotCacheBudgets` after every load/unload, repeated for 10 s because the OS returns pages
  lazily).
- Eviction is WORKLOAD-fair (`cache_key`: `prompt_cache_key` > `metadata.user_id` > system-prompt hash;
  `lruIndexExcluding`).

## SSD-first

- Disk fingerprints include the model path, config size/mtime and overrides, plus sorted indexed weight-shard (or unindexed safetensors) names and size/mtime and `ngram_table.bin` size/mtime; payloads are statted through symlinks, never content-hashed.

- `prefix_cache.ssdFirstActive` = capable arch AND a disk tier, mirrored onto `HotPrefixCache.ssd_first` +
  `DiskTier.ssd_first`: RAM floors at ONE session, `--prefix-cache-mem` = the IDLE allowance.
- Spill and EVICT are two decisions (`PersistOutcome`: only `.persisted` + an agreeing index + landed files license
  discarding RAM); writes ride `kv_disk_writer.zig` (FIFO, `meta.json` last, epoch fence at the ONE removal site);
  per-chunk write-through; a diverging turn hard-links the donor's LANDED chunks; a full-prefix hit CHECKS the entry
  OUT so the first append donates.
- **A checkout is a PROMISE until the append DONATES** (`donateCheckout` right before `Generator.initWithOptions`,
  below every refusal; `releaseCheckout` hands an undonated entry back intact). Disk checkpoints come off the TOP of
  the flush budget; the disk tier serves the pre-media text prefix only.
- **"Free disk" is what the OS will GRANT** (`sushi_volume_free_for_use`, statfs fallback): purgeable space is released
  on demand. The `volumeSpace` test must not race the OS's purgeable answer.

<a id="spec-state"></a>
## Spec state rides the cache

`Entry.mtp` + `restoreSpecSnap`, adopt only on `base + step == matched`; MTP trims to `mtpCommittedLen`; survives the
SSD tier (`spec.safetensors`). An adopted spec cache has ONE owner at a time (`runPrefill` clears its locals BEFORE
`initWithOptions`).

## Guards

`tests/test_prefix_cache_*.sh` (budget revisit, disk, hot, mem, workloads), `tests/test_hybrid_reuse_equivalence.sh`,
`tests/test_mimo_ring_reuse.sh`, `tests/test_qwen4_mtp_head_persist.sh`. Grep the log for `[cache]`, `[hot-cache]`, `[disk-cache]`.
