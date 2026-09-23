# Pack format — the consumer contract

What this engine reads out of an EXL3 pack. The conversion side lives in the
private converter repo; this document is the interface between them, and a
converter change that changes any line here is a format change.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related:
[engine-exl3-experts](engine-exl3-experts.md), [quality-kld](quality-kld.md),
[arch-qwen4exp](arch-qwen4exp.md), [arch-mimo-v2](arch-mimo-v2.md).

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
one scale vector per side and nothing else. The search runs on `inner * g`, so a pack converted with the g-scale search
and one without it differ in their trellis codewords as well as in `suh`.

## `config.json`

```json
"expert_quant": { "format": "exl3", "k": 2.5, "codebook": "mcg", "window": 12 }
```

- `format` — must be the string `exl3`; anything else is `ExpertLayoutUnsupported`.
- `k` — the rate, a JSON number and possibly fractional. It names the WIDEST
  rate a layer packs and is what the engine bills.
- `codebook` — `mul1` or `mcg`. MCG is the codebook for new packs; MUL1 serves
  turboderp's packs; the retired `tiny` is refused by name
  (`Exl3CodebookUnsupported`, in `expert_quant` or a shard stamp alike). The codebook and window follow the MODEL: `moeExl3` sets them
  (`expert_exl3_kernels.setDecodeParams`) before every dispatch, so packs with
  different codebooks can be resident together, and every weight kernel
  inlines its `exl3_pairh`.
- `window` — the codeword width the search hashed, 8..16. **Absent means 16.**
  The same bitstream decodes to different weights at each width, so a window
  this build cannot decode is `Exl3WindowUnsupported`, never a fallback to 16.

`num_experts_per_tok` must be ≤ 32 (the decode reduce bank,
`Exl3TopKExceedsReduceBank`).

### `trunk_quant` (`mimo_v2`)

```json
"trunk_quant": {
  "o_proj":       { "mode": "affine", "bits": 8, "group_size": 64 },
  "lm_head":      { "mode": "affine", "bits": 8, "group_size": 64 },
  "embed_tokens": { "mode": "affine", "bits": 8, "group_size": 64 }
}
```

A pack field, never the source checkpoint's. At load the engine requantizes
each named bf16 tensor with MLX's own affine packer (deterministic) and bills
it at the packed bytes: `o_proj` = every layer's `self_attn.o_proj.weight` and
`lm_head` = `lm_head.weight`, both served through `quantized_matmul`;
`embed_tokens` = `model.embed_tokens.weight`, served by the quantized row
gather. The original checkpoint `kld capture` reads never carries the field,
so the teacher keeps all three as stored.

- Keys are `o_proj`, `lm_head` and `embed_tokens`, each optional; `mode` must
  be `affine`, `bits` one of 2, 3, 4, 5, 6, 8 and `group_size` one of 32, 64,
  128. Anything else is `UnsupportedTrunkQuant`, never a silent bf16 fallback.
- The pack's bytes stay the source's bf16; only the served copy is packed.

## The shard stamp

Each written shard carries a safetensors `__metadata__` map — every value a
string, because that is all safetensors stores:

| key | value |
|---|---|
| `format` | `exl3` |
| `k` | the rate as written, e.g. `2.5` or `4` |
| `codebook` | `mul1` \| `mcg` (`tiny` refused) |
| `window` | the codeword width, e.g. `12` |
| `quantizer` | which search wrote it (`ldlq-rotated` / `direct`) |
| `g_scale` | the global-scale mode, e.g. `gss` |
| `imatrix_sha256` | the imatrix's digest, or `none` |
| `converter` | the calling converter's own `CONVERTER_VERSION` |
| `source_sha256` | digest of the source checkpoint's index (new shards; older ones lack it) |
| `seed_scheme` | which per-expert seed formula produced suh/svh signs (new shards) |
| `out_scales_mode` | the output-scale rule, `auto` \| `always` \| `never` (new shards) |

Two rules run off it:

**Load (`mimo_source.validateShardStamps`, the MiMo loader only — Flash-Next packs are not stamp-checked today).**
Before any bytes are uploaded, a stamped shard's `codebook`, `window` and `k` are checked against `expert_quant`.
A disagreement is `Exl3ShardStampMismatch` — a named refusal, never garbage
weights. An **unstamped** shard is legacy and is admitted. `k` is compared as
halfwords: at or below the config's rate passes, wider refuses.

**Resume (converter side).** A converter adopts an existing shard only when its
**whole** stamp matches, `converter` and `imatrix_sha256` included. Geometry
alone cannot tell a K3/w8/LDLQ shard from a K3/w16/direct one.

These strings are load-bearing across repos: packs already on disk carry
`converter: qwen4-exl3-1-rotated-gss` and
`converter: mimo-exl3-3-calibrated-regularize` (the pre-2026-09-24 output-scale rule); packs converted after the
upstream skew fix carry `qwen4-exl3-2-upstream-skew` / `mimo-exl3-4-upstream-skew`. A converter also writes
`exl3-prior-experts.json` beside the shards, naming every expert that fell back to the prior (no imatrix rows).

## Component packs

The flat component layout keeps embeddings, the output head, two layer-aligned
trunk shards, one routed-expert shard per layer (gate/up/down together), MTP and
vision in separate files. `model.safetensors.index.json` maps the **unchanged**
tensor names onto them; `ngram_table.bin` stays outside safetensors. Legacy
shard names still load. Payload bytes are identical either way — repacking never
decodes or requantizes. See README.md.

## Fixtures

`src/fixtures/exl3_*_linear.safetensors` are committed and `@embedFile`d by
`src/expert_exl3.zig` and `src/transformer.zig`. They are produced by the
private converter; regenerate one only to change the format, and keep the one
searched and decoded at window 12 (`exl3_k2p5_mcg_w12_linear.safetensors`),
which is what certifies a narrowed window against the converter's own decode
rather than against our own masking of a w16 bitstream.

## Quality bar

A pack is judged by KLD against the bf16 teacher (`sushi kld capture` /
`kld compare`, see [quality-kld](quality-kld.md)), never by bytes against an
affine pack: the EXL3 kernel arms round once and are not byte-identical to any
composite.

## Loader rules the engine applies to every pack

- **Expert layout is solved from PACKED shapes** (`expert_quant.zig`): affine
  (bits, group_size) from `w_cols*32 / in_dim`; EXL3 K from the trellis shape;
  `expert_layout` decides `moeExl3` vs the affine kernels and the streaming byte
  plan (`exl3ExpertBytes`). No literal quant width at any
  `mlx_quantized_matmul`/`mlx_dequantize` site — `affineParamsFromGeometry`.
  Affine bits outside {2,3,4,5,6,8} reject at PARSE.
- A trellis whose packed shape does not match the config's rate or expert
  geometry is refused by name at load; a config `k` narrower than a shard would
  under-bill, which on this engine is a Metal OOM rather than an error.
- **Quant modes resolve PER WEIGHT** (`computeQuantParams`; scales dtype decides
  fp8 vs affine; `.biases` mandatory under affine, optional in `loadLinear`;
  `qLinearFwd` passes `mode.cstr()`). A layer-init path that DEMANDS `.scales`
  can't load a DENSE checkpoint (`getLayerScaleOpt`; every dense contracted
  weight owes `maybeTransposeForBf16`).
- **A gather-read table may be quantized only where the READER has a
  quantized-gather path**: LM `embed_tokens` via `gatherQuantizedRows` is a
  SIZE decision (our packs quantize it).
- A component pack's shared files are immutable: a converter replaces a
  hard-linked file, never modifies it in place.
