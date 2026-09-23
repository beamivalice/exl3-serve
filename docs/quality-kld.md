# Quality: KLD against the teacher

How a pack's quality is measured: the `kld` subcommand, the teacher fixtures on this box, the one reading the owner
uses, the rule that the teacher path is lossless, and the recorded KLD of every served pack. Every new KLD lands here
with its binary commit and fixture.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [perf-baselines](perf-baselines.md),
[pack-format](pack-format.md#quality-bar), [engine-exl3-experts](engine-exl3-experts.md#parity-bars),
[arch-mimo-v2](arch-mimo-v2.md).

## The tool

- `src/kld.zig`: `sushi kld capture|compare`. `capture` writes a teacher fixture from the bf16 path; `compare`
  scores a pack against it. KLD is scored to the teacher's first end-of-turn token.
- `compare` prints two lines: "to-first-EOS" and all-positions. **Quote the first-EOS number** (the owner's tables);
  give the all-positions number beside it when comparing with older records.
- `kld` takes the SERVED weight loader (`model.loadWeightsForConfig`); a second loader once bound a MiMo pack's raw FP8
  QKV and made every pack score the same.
- Teacher captures run the KV cache dense (`--kv-quant off`); students are scored at kv8 unless the row says so.

## The standard reading

- **16 prompts x 512 tokens, scored to the first EOS, for every model** (Flash-Next's 16 wikitext prompts, raw text,
  no template). 60x64 is a short-context screen only, never a verdict.
- Differences under ~3% of mean KLD are inside conversion noise (a quantizer's seed alone moves it that much); need
  several seeds per arm or a larger margin before ranking.
- Differences under ~1% on ONE pack are inside the ROUNDING-FLIP floor (measured 2026-09-24 on the MiMo MCG pack:
  flipping 0.07-0.13% of attention outputs by one bf16 ulp, no precision loss, moved 16x512 KLD -0.5% .. +0.55%). A
  kernel or storage change that flips bits reads as a KLD change of that size with no quality meaning; each flip
  pattern is deterministic. Compare arms on the same binary; say the floor beside the number.
- **NAX vs SIMD is accepted hardware noise (owner policy).** A NAX (tensor-op) arm and its SIMD twin accumulate in a
  different order, so they score a slightly different KLD. The engine takes the faster arm knowingly: such a delta is
  recorded, never treated as a regression or a reason to hold a NAX kernel back, and never "fixed" toward SIMD.
  A NAX arm still has to pass its fp32 parity test; this policy covers only the model-level KLD difference.
- No static weight metric (per-module error, weighted error, tail quantiles) predicts model-level KLD rank; only a real
  pack and this reading decide.

## Teacher fixtures on this box

| fixture | model | notes |
|---|---|---|
| `/Users/beam/llm/models/kld-teacher/mlx-serve-bf16-16x512-raw` | Flash-Next | bf16 stream, 16x512, raw; the standard |
| `/Users/beam/llm/models/kld-teacher/mlx-serve-bf16-f32stream-16x512-raw` | Flash-Next | f32 residual stream variant |
| `/Users/beam/llm/models/kld-teacher/mlx-serve-bf16-60x64` | Flash-Next | the 60x64 screen |
| `/Users/beam/llm/models/kld-teacher/mimo-bf16trunk-16x512-raw` | MiMo | original checkpoint, FP8 trunk dequantized to bf16, dense KV; mean strict NLL 0.278; 741 s capture |

Commands (MiMo; Flash-Next drops `--ssd-budget-gb` when the source fits):

```sh
sushi kld capture --model /Users/beam/llm/models/MiMo-V2.6-Flash-RL \
  --prompts /Users/beam/llm/models/kld-teacher/mlx-serve-bf16-16x512-raw --out <teacher dir> \
  --tokens 512 --top-k 10 --label <label> --no-template --kv-quant off --ctx-size 8192 --ssd-budget-gb 94
sushi kld compare --model <pack> --fixture <teacher dir> --label <label> \
  --kv-quant 8 --tokens 512 --top-k 10 --ctx-size 8192 --json <out>.json
```

Both are heavy GPU jobs: take the lock per run (CLAUDE.md, Team process).

<a id="teacher-path"></a>
## The teacher path is lossless

- The reference forward may add NO quantization of its own. MiMo has no bf16 release (FP8 e4m3 trunk, MXFP4
  experts): use the FP8 as FP8 or dequantize it to bf16/f16, never requantize to affine-8; capture with `--kv-quant`
  off. Before any capture, read the loader path the original checkpoint takes and list every dtype change; each must
  be exact.
- Say "original checkpoint through path X", never "bf16 teacher", unless the checkpoint is bf16.
- History: a MiMo teacher captured through an affine-8 trunk and a kv8 cache differed from the lossless one by 0.0076
  nats (the whole engine-to-engine gap mlx-lm had measured); no pack number moved, but a biased reference is refused
  regardless of size. A pack's stored-affine trunk (served packs only) leaves the teacher untouched.
- `SUSHI_NGRAM_BF16_DIR=<hf checkpoint>` serves a Flash-Next pack with the original bf16 n-gram table to isolate
  the PLE table's cost.

## Cross-engine check

mlx-lm's MiMo support (upstream PR 1219, router patched to f32), streamed one layer at a time, against our MiMo
teacher: the original checkpoint scores 0.0077 nats / 95.8% top-1 (the engine-to-engine floor; every flip sits at a
teacher top-2 gap ≤ 0.5 nats, flat across the context); the affine iq2.7 pack scores 0.249 / 80.9% against our 0.247 /
81.5%. The teacher and the tool are validated by an implementation that shares no code with ours.

## Flash-Next (16x512, first EOS, 7186 positions, kv8)

| pack | KLD | top-1 | cosine loss | all positions |
|---|---|---|---|---|
| affine 4-bit gs64 / 8-bit (control) | 0.0818 | 91.39% | 2.63% | 0.0752 |
| turboderp K3, MUL1 w16 | 0.0946 | 90.79% | 2.88% | 0.0866 |
| MUL1 K3 w12 (in-house experts, turboderp dense) | 0.1031 | 90.44% | 3.15% | 0.0951 |
| MCG K3 w12 (served; in-house experts, turboderp dense) | 0.1041 | 90.22% | 3.21% | 0.0958 |
| MCG K3 w15 (in-house experts, turboderp dense; sashimi eebb3e9, pre-#17 skew rule; binary 7ed9795) | 0.1012 | 90.26% | 3.14% | 0.0931 |

w12 -> w15 bought 2.8% of KLD on MCG; the remaining gap to turboderp's MUL1 w16 (0.0946) is not mostly the window.
Raw: scratchpad `qwen_w15_kld.json` (session 3ede61a6); pack `/Users/beam/llm/models/Qwen3.8-Flash-Next-EXL3-K3-w15-mcg-plugged`.

Raw: `/Users/beam/claude-tmp/bench-tiny-vs/kld16x512_*.json`. Binaries a05d15f / 28d7fab (the KLD tool is unchanged
between them).

EXL3 K4 (turboderp), 60x64 screen: the f32 SwiGLU widening moved mean KLD 0.01872 → 0.01816 and top-1 96.20% →
96.07% (a wash; the widening stands on MiMo's magnitudes).

<a id="mimo"></a>
## MiMo (16x512, first EOS, 8037 positions, student kv8)

| pack | expert bpw | KLD | top-1 | cosine loss | all positions | binary |
|---|---|---|---|---|---|---|
| MCG K2.5 w12 (served) | 2.5 | 0.0776 | 92.0% | 1.95% | 0.0792 | a916af3 |
| affine iq2.7 (benchmark reference) | 2.70 | 0.1641 | 88.3% | 3.37% | 0.1682 | a916af3 |
| MCG K2.5 w12, FP8-native trunk (branch) | 2.5 | 0.0776 | | | | f72f989 |
| MCG K2.5 w12, FP8 + o_proj affine-8 (branch) | 2.5 | 0.0774 | | | | f72f989 |
| MCG K2.5 w12, fused sliding prefill (branch) | 2.5 | 0.0775 | | | | |
| MCG K2.5 w12, load-time affine-8 o_proj + lm_head + embed | 2.5 | 0.07761 | 92.09% | 1.96% | 0.07884 | 3b27c11 |
| MCG K2.5 w12, stored imatrix affine-8 o_proj + lm_head + embed | 2.5 | 0.07793 | 92.14% | 1.97% | 0.07944 | 28d8a4b |

Stored imatrix-weighted affine-8 trunk vs the load-time MLX packer: the weighted weight error of the three tensors is
~45% lower (most of it from the error-minimizing search with scale/bias rounded to bf16 before the codes, which the
same search unweighted also gets; the imatrix weighting adds 4-7%), yet 16x512
KLD moves +0.0003 (NLL 0.3480 -> 0.3462, top-1 +0.05 pt): at 8 bits these tensors sit below the pack's noise floor,
which the K2.5 experts set. Raw: `scratchpad/trunkq/live/kld_new.{json,log}`.

The FP8-native teacher against the bf16-rounded teacher: 0.0034 nats. The affine 2.70 bpw MiMo pack is coherent but
degenerates after a few chat turns in use; its KLD had foreshadowed it.
