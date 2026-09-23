# Pack format — the consumer contract

What this engine reads out of an EXL3 pack. The conversion side lives in
PonyExl3 (`python -m ponyexl3.serve_convert`); this document is the interface
between them, and a converter change that changes any line here is a format
change.

Readers in this repo: `src/expert_quant.zig` (layout + `expert_quant` parse),
`src/expert_exl3.zig` (decode), `src/mimo_source.zig` (`validateShardStamps`),
`src/model.zig` (weight loading).

## Routed-expert tensors

Routed experts are stacked per layer and projection, expert `e` on axis 0, so a
gather kernel indexes it directly. Per MoE layer `L` and projection `P` in
`{gate_proj, up_proj, down_proj}`:

| tensor | dtype | shape |
|---|---|---|
| `<prefix>.layers.{L}.mlp.switch_mlp.{P}.trellis` | `U16` | `[E, in/16, out/16, n]` |
| `<prefix>.layers.{L}.mlp.switch_mlp.{P}.suh` | `F16` | `[E, in]` |
| `<prefix>.layers.{L}.mlp.switch_mlp.{P}.svh` | `F16` | `[E, out]` |

`<prefix>` is the arch's own nesting: `language_model.model` for `qwen4_exp`,
`model` for `mimo_v2`. Every other module keeps the affine pack's names and
layout; only the routed banks are EXL3.

`n` is the packed halfwords per 256-weight tile and it, not an integer K, is
what every reader keys on: weight `t`'s codeword is the 16-bit window ending at
`((t+1)*n)>>4`. Even `n` in `[32, 64]` is admitted — 40 = K2.5, 48 = K3,
64 = K4. `k` printed anywhere reads 2.5, never 40. The per-tensor rate is read
from the trellis shape, so a shard at or below the config's rate is over-billed
rather than refused; a wider one refuses.

`in` and `out` are the tile grid times 16, and the Hadamard block is 128
(`H128`), so both must be multiples of it.

### MTP bank

The MTP head's own MoE layer is the same three tensors under the head's prefix
(`language_model.mtp.layers.0.mlp.switch_mlp.{P}.{trellis,suh,svh}`) and is read
by the same decoder. It shares the trunk's geometry, which is why a converter's
resume names its shards `mtp-…`: geometry alone cannot tell the two apart.

Its rows ride the decode chain and refuse wider
(`Exl3MtpRowsExceedDecode`).

### suh / svh, and the g-scale

`suh` scales the input before the Hadamard and `svh` scales the output after it
(`expert_exl3.prepareInput` / `finishOutput`), both through f16 rounding. The
converter's per-expert global codebook scale is **folded into `suh`** — it is
divided out there and never stored as a separate field, so the engine applies
one scale vector per side and nothing else. A pack converted without the g-scale
search and one converted with it differ only in these `suh` values.

## `config.json`

```json
"expert_quant": { "format": "exl3", "k": 2.5, "codebook": "tiny", "window": 12 }
```

- `format` — must be the string `exl3`; anything else is `ExpertLayoutUnsupported`.
- `k` — the rate, a JSON number and possibly fractional. It names the WIDEST
  rate a layer packs and is what the engine bills.
- `codebook` — `mul1`, `tiny` or `mcg`. One per process: the codebook is bound
  at load (`expert_exl3_kernels.setCodebook`) and every weight kernel inlines
  its `exl3_pairh`.
- `window` — the codeword width the search hashed, 8..16. **Absent means 16.**
  The same bitstream decodes to different weights at each width, so a window
  this build cannot decode is `Exl3WindowUnsupported`, never a fallback to 16.

`num_experts_per_tok` must be ≤ 32 (the decode reduce bank,
`Exl3TopKExceedsReduceBank`).

## The shard stamp

Each written shard carries a safetensors `__metadata__` map — every value a
string, because that is all safetensors stores:

| key | value |
|---|---|
| `format` | `exl3` |
| `k` | the rate as written, e.g. `2.5` or `4` |
| `codebook` | `mul1` \| `tiny` \| `mcg` |
| `window` | the codeword width, e.g. `12` |
| `quantizer` | which search wrote it (`ldlq` / `direct`) |
| `g_scale` | the global-scale mode, e.g. `gss` |
| `imatrix_sha256` | the imatrix's digest, or `none` |
| `converter` | the calling converter's own `CONVERTER_VERSION` |

Two rules run off it:

**Load (`mimo_source.validateShardStamps`).** Before any bytes are uploaded, a
stamped shard's `codebook`, `window` and `k` are checked against `expert_quant`.
A disagreement is `Exl3ShardStampMismatch` — a named refusal, never garbage
weights. An **unstamped** shard is legacy and is admitted. `k` is compared as
halfwords: at or below the config's rate passes, wider refuses.

**Resume (converter side).** A converter adopts an existing shard only when its
**whole** stamp matches, `converter` and `imatrix_sha256` included. Geometry
alone cannot tell a K3/w8/LDLQ shard from a K3/w16/direct one.

These strings are load-bearing across repos: packs already on disk carry
`converter: qwen4-exl3-1-rotated-gss` and
`converter: mimo-exl3-3-calibrated-regularize`, and changing either string
invalidates every resume and every verify against those packs.

## Component packs

The flat component layout keeps embeddings, the output head, two layer-aligned
trunk shards, one routed-expert shard per layer (gate/up/down together), MTP and
vision in separate files. `model.safetensors.index.json` maps the **unchanged**
tensor names onto them; `ngram_table.bin` stays outside safetensors. Legacy
shard names still load. Payload bytes are identical either way — repacking never
decodes or requantizes. See README.md.

## Fixtures

`src/fixtures/exl3_*_linear.safetensors` are committed and `@embedFile`d by
`src/expert_exl3.zig` and `src/transformer.zig`. They are produced by
`python -m ponyexl3.serve_convert linear-fixture`; regenerate one only to change
the format, and keep the one searched and decoded at window 12
(`exl3_k2p5_tiny_w12_linear.safetensors`), which is what certifies a narrowed
window against PonyExl3 rather than against our own masking of a w16 bitstream.

## Quality bar

A pack is judged by KLD against the bf16 teacher (`mlx-serve kld capture` /
`kld compare`), never by bytes against an affine pack: the EXL3 kernel arms round
once and are not byte-identical to any composite.
