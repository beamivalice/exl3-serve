# Qwen3.8 Flash Next EXL3 K3 cosine measurements

The completed four-pack RAM/logit-cosine table and fresh affine3 conversion
instructions are in [comparison.md](comparison.md).

Target: `<models>/Qwen3.8-Flash-Next-EXL3-K3`.
Source: `<models>/Qwen/Qwen3.8-Flash-Next`.
Engine base revision: `8d04cedcc5b9d8039e0fd5cb8ffe13f8f1c66010`, plus the
cosine-scoring changes accompanying this report.

Cosine loss means `1 - dot(reference, candidate) / (norm(reference) *
norm(candidate))`. Multiply by 100 for percent. This is not a percentage
reduction in model capability.

## Reference checks

The source and target have identical tokenizer and generation-config hashes:

| File | SHA-256 |
| --- | --- |
| `tokenizer.json` | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `generation_config.json` | `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e` |

Target `config.json` SHA-256:
`3845514255e0c3aff7af1c6d236164a9c762242ab615c1f43416bf280362d2a7`.
Source `config.json` SHA-256:
`889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b`.

The target declares K3/MUL1 routed experts, affine trunk weights and a raw BF16
n-gram table. Config hashes identify configuration, not the full tensor payload.

The reference-control fixture is
`<models>/kld-teacher/mlx-serve-bf16-60x64`: 60 WikiText-2
excerpts rendered with the chat template, with 64 greedy teacher tokens each.
This is teacher-generated continuation scoring, not perplexity on held-out
WikiText ground-truth continuations.

Before comparing K3, the current engine reproduced all 64 rows of the first
teacher prompt with exactly zero KL divergence and 64/64 top-1 agreement:
see `bf16-control.json` and `bf16-control.log`.

The new K3 logit measurement instead uses
`<models>/kld-teacher/mlx-serve-bf16-16x512-raw`, matching the
recovered historical oq8e comparison below: 16 raw excerpts, 512 teacher-generated
tokens each. Both fixtures were captured from the same BF16-source model.

```sh
./zig-out/bin/mlx-serve kld compare \
  --model <models>/Qwen/Qwen3.8-Flash-Next \
  --fixture <models>/kld-teacher/mlx-serve-bf16-60x64 \
  --limit 1 --ssd-budget-gb 60 --kv-quant off \
  --label bf16-reference-control \
  --json measurements/qwen38-exl3-k3/bf16-control.json
```

**Reference qualification:** this is the BF16 source through the served engine,
not an all-BF16 Hugging Face forward. Its normal decode attention requantization
is active (INT8 group 32, with the NVFP4 tail from layer 38). The control log
records those engaged paths. KV is BF16 and MTP is off. The historical fixture
is checked on one prompt, not recaptured in full.

## Recovered historical KLD: oq8e + BF16 n-gram

These are prior measurements, not the new cosine run. All three use
`mlx-serve-bf16-16x512-raw`, BF16 KV, 16 prompts × 512 teacher tokens, and
7,186 positions through the first EOS (8,192 including post-EOS positions).
Each candidate is compared to the BF16-source teacher, not to another quant.
The corresponding logs explicitly confirm a 16-bit n-gram table.

| Historical candidate | KLD through EOS | Top-1 agreement through EOS | KLD, all positions |
| --- | ---: | ---: | ---: |
| `exl3k3-oq8e` | 0.09282674020388251 | 0.9059281937099917 | 0.08521007443529839 |
| `exl3k4-oq8e` | 0.06076958555902735 | 0.9295853047592542 | 0.05683902929988502 |
| `mixed-4-8bit-oq8e` | 0.07690571991846509 | 0.9153910381296966 | 0.07156979722083645 |

Original JSON files (each has an adjacent `.log`), kept outside the repo:

- `exl3-dense8/K3-new2.kld.json`
- `exl3-dense8/K4-new2.kld.json`
- `dense8b/kld-bf16ng.json`

The JSON model paths use the historical
`<models>/Qwen3.8-Flash-Next-MLX-Serve-` prefix plus the candidate
suffixes above. These artifacts alone do not prove tensor identity with the
current renamed/component-packed directories.

The earlier rounded commit summary (K3 0.098, K4 0.068, affine 0.081) is not the
specific oq8e + BF16 n-gram comparison above.

## Current pack: teacher-forced logit cosine

| Metric | Through first EOS | All 512 positions per prompt |
| --- | ---: | ---: |
| Positions | 7,186 | 8,192 |
| Mean cosine similarity | 0.9713559870674179 | 0.9689973385533918 |
| Mean cosine loss (%) | **2.8644012932581942** | 3.1002661446608234 |
| Worst single-position cosine loss (%) | 29.72355967523207 | 29.72355967523207 |
| Mean KLD (nats) | 0.09229618417723492 | 0.08473202061886717 |
| Top-1 agreement (%) | 90.7598107431116 | 91.61376953125 |

Each position compares the complete **248,320-element raw logit vectors**
(including padding rows), with float64 dot products and norms. There is no
centering, softmax, temperature adjustment, or top-k restriction. Means weight
each position equally, not each prompt equally. The primary result stops after
the teacher's first EOS, inclusive; post-EOS continuation is secondary.
Worst loss occurs in the `hed-pe` prompt. Per-position cosine values were not
retained, so no token-level percentile is reported.

This measures the **current deployed pack configuration**, not expert-only
quantization loss: affine trunk quantization and the target's YaRN factor 4
(262,144 → 1,048,576, mscale 1.138629) also participate. The reference caveat
above applies. Raw-logit cosine is sensitive to additive logit shifts even
when the softmax distribution is unchanged.

```sh
./zig-out/bin/mlx-serve kld compare \
  --model <models>/Qwen3.8-Flash-Next-EXL3-K3 \
  --fixture <models>/kld-teacher/mlx-serve-bf16-16x512-raw \
  --kv-quant off --label K3-cosine-raw16x512 \
  --json measurements/qwen38-exl3-k3/logits.json
```

Raw results and engagement log: `logits.json`, `logits.log`.

### Scorer validation

- Red-first tests failed on the missing cosine fields; the completed scorer's
  five row tests pass (identical, scaled, orthogonal, opposite, nonfinite and
  zero-norm cases, plus existing KLD checks).
- `zig build -Doptimize=ReleaseFast` passed.
- `zig build test -Doptimize=ReleaseFast -Dtest-filter='kld:'` passed.
- `zig build test -Doptimize=ReleaseFast` passed.
- Independently recomputing position-weighted means from the emitted prompt
  records agrees with every reported KLD, NLL and cosine mean within `1e-12`.

## K3 sampled reconstructed expert weights

The CPU-only pass covers all 48 trunk layers, expert IDs `[63, 191, 319, 447]`
in each layer, and gate/up/down: **576 full matrices, 943,718,400 weights**.
This is a fixed stratified sample of 0.78125% of routed trunk expert matrices,
not an exhaustive scan. MTP, shared experts and other trunk weights are excluded.

| Statistic | Cosine loss (%) |
| --- | ---: |
| Global cosine from summed dot/norm moments | **0.983836** |
| Mean per-matrix loss | 0.983221 |
| P95 per-matrix loss | 1.034278 |
| Worst sampled matrix | 1.172235 |

The global similarity is **0.9901616405853412**. The worst matrix is layer 0,
expert 447, down projection. Global loss by projection is 1.001869% gate,
1.002092% up and 0.946143% down.

The script reads original BF16 source experts, splits gate/up, transposes to
the public `[in, out]` layout, and compares against the private converter's CPU reconstruction
including H128 transforms, input/output scales and final FP16 rounding.
Dot products and norm sums use float64. Raw tensor reads avoid loading full
expert banks or invoking GPU computation.

The command is kept with the private converter (sashimi).

The six hermetic Python tests passed, and the real-header audit validated all
48 layers × 3 matrix layouts. The JSON contains exact sample IDs, moments,
config/index hashes and method metadata. Runtime: 948.2 seconds.
