# Benchmarks — sushi decode by release

**Update rules — read before editing:**
- Results go into the tables ONLY. No text, no commentary, no per-release notes — commit messages carry the stories.
- **Apple M5 Max 128 GB ONLY.** Numbers across hardware are not comparable and one mixed column poisons the history.
- A cell is `./tests/bench.sh` decode tok/s (llmprobe `--bench-only`: warmup discarded, median of 3, its own code-completion prompt), sushi ReleaseFast at its FASTEST config, with the speculative mode that engaged named beside the number. `·` = not measured that release.
- The perf gate is one pack, Qwen3.8-Flash-Next-Sushi-4bpw. `speedup` = first measured column vs the latest.

## Decode tok/s by release

| Model | v1.0.0 | speedup |
|---|---|---|
| Qwen3.8-Flash-Next-Sushi-4bpw (MTP) | · | · |
