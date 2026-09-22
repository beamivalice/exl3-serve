#!/usr/bin/env python3
"""MiMo-V2.6 (model_type `mimo_v2`) HF -> mlx-serve pack converter.

Source: the XiaomiMiMo MiMo-V2.6-Flash-RL layout (48 layers, hidden 4096,
vocab 152576 untied, 256 experts top-8, layer 0 dense, layers 1..47 MoE, 3
nextn MTP layers + a dflash draft sidecar). One fused qkv row block per
attention layer: nq*192 | nkv*192 | nkv*128 (qk head dim 192, v head dim 128);
SWA layers 64q/8kv, GA layers (hybrid_layer_pattern[i] == 0) 64q/4kv.

Layout / renames:
    model.layers.N.mlp.experts.E.{gate,up,down}_proj.{weight,weight_scale}
        -> per-layer STACKED banks [E, ...] at
           model.layers.N.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales}
           (the engine's expert-store convention: gather kernels index expert e
           on axis 0). The mxfp4 e8m0 block-32 bytes are kept NATIVE and
           BYTE-IDENTICAL: HF's `weight` U8 [out, in/2] packs two e2m1 nibbles
           per byte (low nibble = the even element), which is exactly MLX's
           packed-u32 layout (element i at bit offset 4i) read as U32 [out, in/8]
           -- a pure view relabel, no decode. `weight_scale` U8 e8m0 (2^(b-127))
           becomes MLX's bias-less `scales` verbatim. NO mxfp4 -> f32 -> requant
           happens anywhere in this converter.
    model.layers.N.self_attn.qkv_proj.{weight,weight_scale_inv}
        -> fp8 e4m3 dequant (128x128 F32 `weight_scale_inv` blocks) ->
           SPLIT into self_attn.{q,k,v}_proj.* -> affine 8-bit gs64
           ({weight,scales,biases}). Split at conversion because the loader's
           splitFusedQkvRows demands rows == q_rows + 2*kv_rows from one
           head_dim, which nq*192 | nkv*192 | nkv*128 can never satisfy.
    model.layers.0.mlp.{gate,up,down}_proj + model.mtp.layers.* fp8 linears
        -> same fp8 dequant -> affine 8-bit gs64.
    model.mtp.layers.{0,1,2}.*  -> mtp/weights.safetensors keys mtp.layers.*
        (the loader's native `mtp.*` sidecar convention; MTP norms are plain
        RMSNorm -- nothing to fold).
    o_proj / embed_tokens / lm_head / norms / router gate /
    e_score_correction_bias / attention_sink_bias -> BF16 (F32) verbatim.
    visual.* / audio_encoder.* / speech_embeddings.* -> DROPPED in v1
        (vision + audio come later; the pack README carries the manifest note).
    dflash/ -> hard-linked verbatim (self-contained draft model).

Every other 2-D fp8 linear keeps the 8-bit affine pack convention of
convert_qwen38_flash_next.py (weight U32 packed + BF16 scales + BF16 biases,
group 64). config.json carries every arch field plus
`quantization = {bits: 4, group_size: 32, mode: "mxfp4"}` in the style of the
gpt-oss mxfp4 packs (per-weight geometry resolves the affine pieces).

  python3 tests/convert_mimo_v2.py --self-test          # hermetic, synthetic
  python3 tests/convert_mimo_v2.py --src <hf dir> --dst <pack dir> --dry-run
  python3 tests/convert_mimo_v2.py --src /Users/beam/llm/models/MiMo-V2.6-Flash-RL \
      --dst ~/.mlx-serve/models/ddalcu/MiMo-V2.6-Flash-MLX-Serve-mxfp4

--dry-run is a full name/shape audit of the source index (shard HEADERS only,
no tensor data) that prints the tensor-name mapping plan and the streamed
expert byte plan, then exits without writing anything. It does not validate
payload bytes or numerical fp8 dequantization.
"""

import argparse
import json
import os
import re
import shutil
import struct
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with warnings.catch_warnings():
    # convert_dsv4_weights' e8m0 LUT build overflows exp2 at code 254+ on import.
    warnings.simplefilter("ignore", RuntimeWarning)
    from convert_dsv4_weights import (E2M1_TABLE, E4M3_LUT, E8M0_LUT,  # noqa: E402
                                      mlx_affine_quant, write_safetensors_raw)

COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt", "chat_template.jinja", "generation_config.json",
              "preprocessor_config.json", "LICENSE")
DROP_PREFIXES = ("visual.", "audio_encoder.", "speech_embeddings.")
EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate|up|down)_proj\.(weight|weight_scale)$")
MTP_RE = re.compile(r"^model\.mtp\.layers\.(\d+)\.(.+)$")
FP8_TEXT_BASE_RE = re.compile(
    r"^(?:model\.layers\.\d+\.self_attn\.qkv_proj"
    r"|model\.layers\.0\.mlp\.(?:gate|up|down)_proj)$")
FP8_MTP_BASE_RE = re.compile(
    r"^mtp\.layers\.\d+\.(?:self_attn\.qkv_proj|mlp\.(?:gate|up|down)_proj)$")
VERBATIM_RE = tuple(re.compile(p) for p in (
    r"^model\.embed_tokens\.weight$",
    r"^lm_head\.weight$",
    r"^model\.norm\.weight$",
    r"^model\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm)\.weight$",
    r"^model\.layers\.\d+\.self_attn\.o_proj\.weight$",
    r"^model\.layers\.\d+\.self_attn\.attention_sink_bias$",
    r"^model\.layers\.\d+\.mlp\.gate\.weight$",
    r"^model\.layers\.\d+\.mlp\.gate\.e_score_correction_bias$",
))

class Dims:
    """Architecture dims the shape math is solved against (from config.json)."""

    def __init__(self, *, layers, hidden, intermediate, moe_intermediate,
                 experts, nq, kv_ga, kv_swa, qk_dim, v_dim, ga_layers=(),
                 vocab=None, mtp_layers=0, swa_sink=False, full_sink=False):
        self.layers = layers
        self.hidden = hidden
        self.intermediate = intermediate
        self.moe_intermediate = moe_intermediate
        self.experts = experts
        self.nq = nq
        self.kv_ga = kv_ga
        self.kv_swa = kv_swa
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self._ga_layers = frozenset(ga_layers)
        self.vocab = vocab
        self.mtp_layers = mtp_layers
        self.swa_sink = swa_sink
        self.full_sink = full_sink

    @classmethod
    def from_config(cls, cfg):
        return cls(
            layers=cfg["num_hidden_layers"], hidden=cfg["hidden_size"],
            intermediate=cfg["intermediate_size"],
            moe_intermediate=cfg["moe_intermediate_size"],
            experts=cfg["n_routed_experts"],
            nq=cfg["num_attention_heads"],
            kv_ga=cfg["num_key_value_heads"],
            kv_swa=cfg["swa_num_key_value_heads"],
            qk_dim=cfg["head_dim"], v_dim=cfg["v_head_dim"],
            ga_layers=[i for i, p in enumerate(cfg["hybrid_layer_pattern"])
                       if p == 0], vocab=cfg.get("vocab_size"),
            mtp_layers=cfg.get("num_nextn_predict_layers", 0),
            swa_sink=cfg.get("add_swa_attention_sink_bias", False),
            full_sink=cfg.get("add_full_attention_sink_bias", False))

    def kv_heads(self, layer):
        """GA layers (hybrid_layer_pattern[layer] == 0) carry kv_ga heads."""
        return self.kv_ga if layer in self._ga_layers else self.kv_swa


def qkv_row_layout(dims, layer):
    """(q_rows, k_rows, v_rows) of the fused qkv projection for one layer."""
    nkv = dims.kv_heads(layer)
    return (dims.nq * dims.qk_dim, nkv * dims.qk_dim, nkv * dims.v_dim)


def fp8_out_bases(base):
    """Output bases for one fp8 tensor: qkv splits into q/k/v, mlp stays."""
    if base.endswith(".qkv_proj"):
        stem = base[:-len(".qkv_proj")]
        return tuple(stem + f".{p}_proj" for p in ("q", "k", "v"))
    return (base,)


def plan_tensor(key, meta, dims):
    """Map one HF tensor name to its pack action.

    Returns (action, payload): action in {verbatim, bank_slice, fp8_affine,
    mtp_verbatim, mtp_fp8_affine, drop}. Raises `ValueError` naming the tensor
    when it belongs to no known category (a fail-loud audit, never a silent
    skip)."""
    for prefix in DROP_PREFIXES:
        if key.startswith(prefix):
            return "drop", prefix
    m = EXPERT_RE.match(key)
    if m:
        layer, expert = int(m.group(1)), int(m.group(2))
        if not 1 <= layer < dims.layers:
            raise ValueError(f"{key}: MoE bank layer outside 1..{dims.layers - 1}")
        if not 0 <= expert < dims.experts:
            raise ValueError(f"{key}: expert outside 0..{dims.experts - 1}")
        part = "weight" if m.group(4) == "weight" else "scales"
        return "bank_slice", (layer, m.group(3), expert, part)
    m = MTP_RE.match(key)
    if m:
        out = mtp_out_name(key)
        for suffix in (".weight_scale_inv", ".weight_scale", ".weight"):
            if out.endswith(suffix):
                base = out[:-len(suffix)]
                if FP8_MTP_BASE_RE.match(base):
                    return "mtp_fp8_affine", fp8_out_bases(base)
                if suffix != ".weight":
                    raise ValueError(f"stray scale tensor on a bf16 module: {key}")
                break
        return "mtp_verbatim", out
    for suffix in (".weight_scale_inv", ".weight_scale", ".weight"):
        if key.endswith(suffix):
            base = key[:-len(suffix)]
            if FP8_TEXT_BASE_RE.match(base):
                return "fp8_affine", fp8_out_bases(base)
            if suffix != ".weight":
                raise ValueError(f"stray scale tensor outside the fp8/bank sets: {key}")
            break
    for pat in VERBATIM_RE:
        if pat.match(key):
            return "verbatim", rename_text(key)
    raise ValueError(f"unclassified tensor: {key}")


def check_meta(key, action, meta, dims):
    """Audit: raise when a planned tensor's dtype/shape contradicts config dims."""
    shape, dt = tuple(meta["shape"]), meta["dtype"]
    if action == "bank_slice":
        m = EXPERT_RE.match(key)
        out_dim, in_dim = bank_dims(dims, m.group(3))
        part_w = m.group(4) == "weight"
        want = ("U8", (out_dim, in_dim // (2 if part_w else 32)))
    elif action in ("fp8_affine", "mtp_fp8_affine"):
        rows, cols = fp8_dims(key, dims)
        if key.endswith(".weight_scale_inv"):
            if not (dt == "F32" and shape[1] == cols // 128
                    and shape[0] >= -(-rows // 128)):
                raise ValueError(f"{key}: found {dt} {list(shape)}, expected F32 "
                                 f"[>={-(-rows // 128)}, {cols // 128}]")
            return
        want = ("F8_E4M3", (rows, cols))
    else:
        want = verbatim_shape(key, dims)
    if want is not None and (dt, shape) != want:
        raise ValueError(f"{key}: found {dt} {list(shape)}, expected "
                         f"{want[0]} {list(want[1])}")


def fp8_dims(key, dims):
    """(rows, cols) of one fp8 source weight from the config dims."""
    m = re.match(r"^(?:model|model\.mtp|mtp)\.layers\.(\d+)\.(.+?)"
                 r"\.weight(?:_scale_inv)?$", key)
    layer, stem = int(m.group(1)), m.group(2)
    if stem == "self_attn.qkv_proj":
        nkv = dims.kv_swa if key.startswith("model.mtp.") else dims.kv_heads(layer)
        return (dims.nq * dims.qk_dim + nkv * dims.qk_dim + nkv * dims.v_dim,
                dims.hidden)
    proj = stem.split(".", 1)[1]
    if proj == "down_proj":
        return dims.hidden, dims.intermediate
    return dims.intermediate, dims.hidden


def verbatim_shape(key, dims):
    """(dtype, shape) a verbatim tensor must match, or None to skip."""
    hidden, nq = dims.hidden, dims.nq
    o_shape = (hidden, nq * dims.v_dim)
    if key in ("model.embed_tokens.weight", "lm_head.weight"):
        return None if dims.vocab is None else ("BF16", (dims.vocab, hidden))
    if key == "model.norm.weight":
        return ("BF16", (hidden,))
    rules = (
        (r"^model\.layers\.\d+\.(?:input_layernorm|post_attention_layernorm)\.weight$",
         ("BF16", (hidden,))),
        (r"^model\.layers\.\d+\.self_attn\.o_proj\.weight$", ("BF16", o_shape)),
        (r"^model\.layers\.\d+\.self_attn\.attention_sink_bias$", ("BF16", (nq,))),
        (r"^model\.layers\.\d+\.mlp\.gate\.weight$",
         ("BF16", (dims.experts, hidden))),
        (r"^model\.layers\.\d+\.mlp\.gate\.e_score_correction_bias$",
         ("F32", (dims.experts,))),
        (r"^model\.mtp\.layers\.\d+\.eh_proj\.weight$",
         ("BF16", (hidden, 2 * hidden))),
        (r"^model\.mtp\.layers\.\d+\.(?:input_layernorm|pre_mlp_layernorm|enorm|hnorm|final_layernorm)\.weight$",
         ("BF16", (hidden,))),
        (r"^model\.mtp\.layers\.\d+\.self_attn\.o_proj\.weight$",
         ("BF16", o_shape)),
        (r"^model\.mtp\.layers\.\d+\.self_attn\.attention_sink_bias$",
         ("BF16", (nq,))),
    )
    for pat, want in rules:
        if re.match(pat, key):
            return want
    return None


def rename_text(key):
    """Pack name for a verbatim text tensor (root names preserved)."""
    return key


def mtp_out_name(key):
    """model.mtp.layers.{i}.rest -> mtp.layers.{i}.rest (loader's mtp.* prefix)."""
    m = MTP_RE.match(key)
    if not m:
        raise ValueError(f"not an MTP tensor: {key}")
    return f"mtp.layers.{m.group(1)}.{m.group(2)}"


def dequant_fp8_blocks(w_u8, s_f32, block=128):
    """fp8 e4m3 [out, in] with F32 scale_inv per [block, block] -> f32.

    Tiles start at row zero of ONE matrix. Rank-sharded fused QKV must use
    dequant_fp8_qkv so each rank's final partial tile stays local."""
    rows, cols = w_u8.shape
    s_rows, s_cols = s_f32.shape
    if cols % block or s_cols != cols // block or s_rows * block < rows:
        raise ValueError(f"scale grid {(s_rows, s_cols)} does not tile "
                         f"{(rows, cols)} at block {block}")
    w = E4M3_LUT[np.asarray(w_u8, dtype=np.uint8)]
    s = np.repeat(np.repeat(np.asarray(s_f32, np.float32), block, 0), block, 1)
    return w * s[:rows, :cols]


def dequant_fp8_qkv(raw, scales, q_rows, k_rows, v_rows):
    """Decode rank-local Q|K|V blocks, then assemble global Q, K and V.

    Layout follows llama.cpp conversion/mimo.py (_tp_aware_qkv_dequant).
    FP8 scale tiles restart at each rank, including partial final tiles.
    """
    sizes = (q_rows, k_rows, v_rows)
    total = sum(sizes)
    if raw.shape[0] != total:
        raise ValueError("fused QKV row count disagrees with head geometry")
    tp = next((n for n in (8, 4)
               if all(size % n == 0 for size in sizes)
               and scales.shape[0] == n * ((total // n + 127) // 128)), None)
    if tp is None:
        raise ValueError(f"cannot resolve QKV rank layout from {raw.shape}, {scales.shape}")
    rows_per_rank = total // tp
    blocks_per_rank = (rows_per_rank + 127) // 128
    q_per, k_per, _ = (size // tp for size in sizes)
    groups = ([], [], [])
    for rank in range(tp):
        w = raw[rank * rows_per_rank:(rank + 1) * rows_per_rank]
        s = scales[rank * blocks_per_rank:(rank + 1) * blocks_per_rank]
        decoded = dequant_fp8_blocks(w, s)
        groups[0].append(decoded[:q_per])
        groups[1].append(decoded[q_per:q_per + k_per])
        groups[2].append(decoded[q_per + k_per:])
    return tuple(np.ascontiguousarray(np.concatenate(group)) for group in groups)


def split_qkv(w, nq, nkv, qk_dim, v_dim):
    """Split a dequantized fused qkv matrix's rows into (q, k, v) f32."""
    q_rows, k_rows, v_rows = nq * qk_dim, nkv * qk_dim, nkv * v_dim
    if w.shape[0] != q_rows + k_rows + v_rows:
        raise ValueError(f"qkv rows {w.shape[0]} != {q_rows} | {k_rows} | {v_rows}")
    bounds = ((0, q_rows), (q_rows, q_rows + k_rows),
              (q_rows + k_rows, q_rows + k_rows + v_rows))
    return tuple(np.ascontiguousarray(w[a:b]) for a, b in bounds)


def qkv_geometry_for_source(key, dims):
    """Resolve source-prefix MTP geometry.

    MTP tensors are named ``model.mtp.layers.*`` in the HF checkpoint but
    always use the SWA head counts.  The emitted pack prefix is unrelated to
    this choice and must not be inspected here.
    """
    m = re.match(
        r"^model\.(mtp\.)?layers\.(\d+)\.self_attn\.qkv_proj\.weight$",
        key)
    if not m:
        raise ValueError(f"not a source qkv weight: {key}")
    layer = int(m.group(2))
    nkv = dims.kv_swa if m.group(1) else dims.kv_heads(layer)
    return dims.nq, nkv, dims.qk_dim, dims.v_dim


def split_qkv_for_source(key, w, dims):
    return split_qkv(w, *qkv_geometry_for_source(key, dims))


def bank_dims(dims, proj):
    """(out_dim, in_dim) of one expert's projection matrix."""
    if proj in ("gate", "up"):
        return dims.moe_intermediate, dims.hidden
    if proj == "down":
        return dims.hidden, dims.moe_intermediate
    raise ValueError(f"unknown expert projection: {proj}")


def bank_shapes(dims, proj):
    """((w_dtype, w_shape), (s_dtype, s_shape)) of one stacked mxfp4 bank."""
    out_dim, in_dim = bank_dims(dims, proj)
    if in_dim % 8:
        raise ValueError(f"{proj}: packed-u32 rows need in % 8 == 0, got {in_dim}")
    if in_dim % 32:
        raise ValueError(f"{proj}: e8m0 block 32 needs in % 32 == 0, got {in_dim}")
    e = dims.experts
    return (("U32", (e, out_dim, in_dim // 8)),
            ("U8", (e, out_dim, in_dim // 32)))


def stack_expert_bank(slices_w, slices_s, experts, out_dim, in_dim):
    """Per-expert raw slices -> the two stacked-bank triples.

    Each input list is `experts` raw byte slices in expert-index order; the
    output weight/scales bytes are the plain concatenation (byte identity).
    HF's packed-U8 mxfp4 rows ARE MLX's packed-U32 rows byte for byte
    (element i at bit offset 4i, low nibble = the even element), so stacking
    never touches a value."""
    if len(slices_w) != experts or len(slices_s) != experts:
        raise ValueError(f"need {experts} expert slices, got "
                         f"{len(slices_w)} weight / {len(slices_s)} scales")
    w_bytes, s_bytes = out_dim * in_dim // 2, out_dim * in_dim // 32
    for e, (w, s) in enumerate(zip(slices_w, slices_s)):
        if len(w) != w_bytes or len(s) != s_bytes:
            raise ValueError(f"expert {e}: slice sizes {len(w)}/{len(s)} != "
                             f"{w_bytes}/{s_bytes}")
    return {
        "weight": ("U32", (experts, out_dim, in_dim // 8), b"".join(slices_w)),
        "scales": ("U8", (experts, out_dim, in_dim // 32), b"".join(slices_s)),
    }


def expert_slot_bytes(dims, proj):
    """Per-expert bytes of one projection's streamed slice (from its shapes)."""
    (_, ws), (_, ss) = bank_shapes(dims, proj)
    return ws[1] * ws[2] * 4 + ss[1] * ss[2]


def build_config(hf_cfg):
    """Source config.json -> the pack's config.json content (a JSON object)."""
    cfg = dict(hf_cfg)
    for k in ("vision_config", "audio_config", "processor_config", "auto_map",
              "quantization_config", "quantization", "transformers_version"):
        cfg.pop(k, None)
    # the converter splits the fused qkv into q/k/v; the pack is text-only v1
    cfg["attention_projection_layout"] = "split"
    if cfg.get("routed_scaling_factor") is None:
        cfg["routed_scaling_factor"] = 1.0
    cfg["language_model_only"] = True
    cfg["quantization"] = {"bits": 4, "group_size": 32, "mode": "mxfp4"}
    cfg["quantization_config"] = cfg["quantization"]
    return cfg


SHARD_BYTES = 2 * 1024 ** 3

README = """\
# {repo_name}

mlx-serve pack of MiMo-V2.6-Flash-RL (`model_type: mimo_v2`), built by
`tests/convert_mimo_v2.py`. {total_gb:.1f} GB text model.

## Widths

| tensors | width |
|---|---|
| routed experts (256 x 47 layers) | native mxfp4, e8m0 block 32 — byte-identical to the HF checkpoint |
| attention q/k/v (split from the fused qkv), layer-0 MLP, MTP linears | 8-bit affine, group 64 (re-quantized from fp8 e4m3 + 128x128 `weight_scale_inv`) |
| o_proj, eh_proj, embed_tokens, lm_head, router gate, norms, attention sinks | bf16 |
| `e_score_correction_bias` | f32 (router selection only) |

The routed experts are per-layer stacked `[E, ...]` banks at
`model.layers.N.mlp.switch_mlp.{{gate,up,down}}_proj.{{weight,scales}}` (the
engine's expert-store convention). The source's packed-U8 mxfp4 rows ARE MLX's
packed-U32 layout (element i at bit offset 4i, low nibble = the even element),
so every bank byte is the plain concatenation of the per-expert slices — no
mxfp4 decode or requantization happens anywhere in the conversion.

The fused `self_attn.qkv_proj` (nq*192 | nkv*192 | nkv*128 rows) is split into
`self_attn.{{q,k,v}}_proj` at conversion: the engine's fused-qkv splitter assumes
one head dim and cannot represent v at 128.

## Sidecars

- `mtp/weights.safetensors` — the 3-layer nextn MTP head under the loader's
  `mtp.*` prefix (`mtp.layers.{{0,1,2}}.*`; source `model.mtp.layers.*`, qkv
  split and fp8 linears re-quantized like the trunk). Its norms are plain
  RMSNorm — nothing to fold.
- `dflash/` — the dflash draft model, copied verbatim.

## Dropped in v1 (manifest)

Vision and audio are NOT in this pack yet. These source tensors were dropped at
conversion and land in a later pass:

| source prefix | tensors |
|---|---|
| `visual.*` | {n_visual} |
| `audio_encoder.*` | {n_audio} |
| `speech_embeddings.*` | {n_speech} |

## Serving

```bash
mlx-serve --model <org>/{repo_name} --serve
```
"""


def read_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    return header, 8 + hlen


def read_bytes(shard, data_off, meta):
    b, e = meta["data_offsets"]
    with open(shard, "rb") as f:
        f.seek(data_off + b)
        raw = f.read(e - b)
    if len(raw) != e - b:
        raise ValueError(f"{shard}: short read")
    return raw


def read_raw(shard, data_off, meta):
    np_dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32,
             "U32": np.uint32, "I32": np.int32, "I64": np.int64, "U8": np.uint8,
             "F8_E4M3": np.uint8, "F8_E8M0": np.uint8}[meta["dtype"]]
    return np.frombuffer(read_bytes(shard, data_off, meta), np_dt).reshape(meta["shape"])


def required_source_tensors(dims):
    """Names that must be present before a pack conversion can start.

    Pair completeness is explicit here rather than inferred from whichever
    member happened to be named by the source index.  This keeps a truncated
    index from looking like a valid partial pack.
    """
    required = {"model.embed_tokens.weight", "lm_head.weight",
                "model.norm.weight"}

    def add_fp8(base):
        required.update((base + ".weight", base + ".weight_scale_inv"))

    def add_expert(base):
        required.update((base + ".weight", base + ".weight_scale"))

    for layer in range(dims.layers):
        stem = f"model.layers.{layer}"
        required.update((f"{stem}.input_layernorm.weight",
                         f"{stem}.post_attention_layernorm.weight",
                         f"{stem}.self_attn.qkv_proj.weight",
                         f"{stem}.self_attn.qkv_proj.weight_scale_inv",
                         f"{stem}.self_attn.o_proj.weight"))
        if layer in dims._ga_layers:
            if dims.full_sink:
                required.add(f"{stem}.self_attn.attention_sink_bias")
        elif dims.swa_sink:
            required.add(f"{stem}.self_attn.attention_sink_bias")
        if layer == 0:
            for proj in ("gate", "up", "down"):
                add_fp8(f"{stem}.mlp.{proj}_proj")
        else:
            required.update((f"{stem}.mlp.gate.weight",
                             f"{stem}.mlp.gate.e_score_correction_bias"))
            for expert in range(dims.experts):
                for proj in ("gate", "up", "down"):
                    add_expert(f"{stem}.mlp.experts.{expert}.{proj}_proj")

    for layer in range(dims.mtp_layers):
        stem = f"model.mtp.layers.{layer}"
        required.update((
            f"{stem}.eh_proj.weight",
            f"{stem}.enorm.weight",
            f"{stem}.final_layernorm.weight",
            f"{stem}.hnorm.weight",
            f"{stem}.input_layernorm.weight",
            f"{stem}.pre_mlp_layernorm.weight",
            f"{stem}.self_attn.o_proj.weight",
            f"{stem}.self_attn.qkv_proj.weight",
            f"{stem}.self_attn.qkv_proj.weight_scale_inv",
        ))
        if dims.swa_sink:
            required.add(f"{stem}.self_attn.attention_sink_bias")
        for proj in ("gate", "up", "down"):
            add_fp8(f"{stem}.mlp.{proj}_proj")
    return tuple(sorted(required))


def missing_required_tensors(index_names, dims):
    """Return required source names absent from the index weight map."""
    return sorted(set(required_source_tensors(dims)) - set(index_names))


def collect_plan(src, dims):
    """Plan + audit every index name from the shard HEADERS (no data reads)."""
    index = json.loads((src / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    errors = [f"missing required tensor: {key}"
              for key in missing_required_tensors(wm, dims)]
    shards = {}
    for fname in sorted(set(wm.values())):
        path = src / fname
        if not path.exists():
            raise SystemExit(f"missing shard {fname} (incomplete download)")
        shards[fname] = read_header(path)
    plan, dropped = {}, {}
    for key in sorted(wm):
        header, _ = shards[wm[key]]
        meta = header.get(key)
        if meta is None:
            errors.append(f"{key}: named in the index but absent from {wm[key]}")
            continue
        try:
            action, payload = plan_tensor(key, meta, dims)
            check_meta(key, action, meta, dims)
        except ValueError as e:
            errors.append(str(e))
            continue
        plan[key] = (action, payload)
        if action == "drop":
            dropped[payload] = dropped.get(payload, 0) + 1
    cover = {}
    fp8_bases = {}
    for key, (action, payload) in plan.items():
        if action == "bank_slice":
            k = (payload[0], payload[1])
            cover[k] = cover.get(k, 0) + 1
        elif action in ("fp8_affine", "mtp_fp8_affine"):
            base = (key[:-len(".weight_scale_inv")]
                    if key.endswith(".weight_scale_inv") else key[:-len(".weight")])
            fp8_bases.setdefault(base, []).append(key)
    for layer in range(1, dims.layers):
        for proj in ("gate", "up", "down"):
            n = cover.get((layer, proj), 0)
            if n != 2 * dims.experts:
                errors.append(f"model.layers.{layer}.mlp.switch_mlp.{proj}: "
                              f"{n} source tensors, expected {2 * dims.experts}")
    for base, keys in sorted(fp8_bases.items()):
        if sorted(keys) != sorted([base + ".weight", base + ".weight_scale_inv"]):
            errors.append(f"{base}: incomplete fp8 pair ({len(keys)} tensors)")
    total = index.get("metadata", {}).get("total_size", 0)
    return wm, shards, plan, dropped, errors, total


def print_report(dims, n_src, plan, dropped, total):
    counts = {}
    for action, _ in plan.values():
        counts[action] = counts.get(action, 0) + 1
    print(f"mimo_v2 pack plan: {n_src} tensors, {total / 1e9:.1f} GB source")
    for action in ("bank_slice", "fp8_affine", "verbatim",
                   "mtp_fp8_affine", "mtp_verbatim", "drop"):
        if action in counts:
            print(f"  {action:15s} {counts[action]}")
    for prefix in DROP_PREFIXES:
        if prefix in dropped:
            print(f"    dropped {prefix}*: {dropped[prefix]}")
    slot = {p: expert_slot_bytes(dims, p) for p in ("gate", "up", "down")}
    per_expert = sum(slot.values())
    print("streamed expert byte plan (per-expert spans = tensor_bytes / E):")
    for p in ("gate", "up", "down"):
        print(f"  {p:5s} slot {slot[p] / 1e6:.2f} MB")
    print(f"  expert total {per_expert / 1e6:.2f} MB x {dims.experts} experts x "
          f"{dims.layers - 1} MoE layers = "
          f"{per_expert * dims.experts * (dims.layers - 1) / 1e9:.1f} GB")
    print("tensor-name mapping:")
    print("  model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight U8 "
          "-> model.layers.N.mlp.switch_mlp.{gate,up,down}_proj.weight U32 [E, out, in/8]")
    print("  model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight_scale U8 e8m0 "
          "-> model.layers.N.mlp.switch_mlp.{gate,up,down}_proj.scales U8 [E, out, in/32]")
    print("  model.layers.N.self_attn.qkv_proj.{weight,weight_scale_inv} fp8 "
          "-> self_attn.{q,k,v}_proj.{weight,scales,biases} 8-bit affine gs64")
    print("  model.layers.0.mlp.{gate,up,down}_proj.{weight,weight_scale_inv} fp8 "
          "-> same base .{weight,scales,biases} 8-bit affine gs64")
    print("  model.mtp.layers.{0,1,2}.* -> mtp/weights.safetensors mtp.layers.{0,1,2}.*")
    print("  o_proj / embed_tokens / lm_head / norms / gate / e_score_correction_bias "
          "/ attention_sink_bias -> verbatim bf16 (f32)")
    print("  visual.* / audio_encoder.* / speech_embeddings.* -> DROPPED (v1)")
    print("  dflash/ -> hard-linked verbatim")


def _paths_overlap(a, b):
    return a == b or a in b.parents or b in a.parents


def _symlink_component(path):
    """Return a symlink in a destination path, including an existing parent."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    probe = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        probe /= part
        if probe.is_symlink():
            return probe
    return None


def validate_conversion_paths(src, dst):
    """Validate read-only source and fresh, disjoint destination before reads."""
    src = Path(src).expanduser()
    dst = Path(dst).expanduser()
    try:
        src_real = src.resolve(strict=True)
    except (OSError, RuntimeError) as e:
        raise SystemExit(f"refusing conversion: source path is not usable: {src} ({e})")
    if not src_real.is_dir():
        raise SystemExit(f"refusing conversion: source is not a directory: {src}")

    link = _symlink_component(dst)
    if link is not None:
        raise SystemExit(f"refusing conversion: destination contains symlink {link}")
    if os.path.lexists(os.fspath(dst)):
        if not dst.is_dir():
            raise SystemExit(f"refusing conversion: destination is an existing file: {dst}")
        try:
            nonempty = next(dst.iterdir(), None) is not None
        except OSError as e:
            raise SystemExit(
                f"refusing conversion: cannot inspect destination {dst} ({e})")
        if nonempty:
            raise SystemExit(
                f"refusing conversion: destination is not empty: {dst}")

    try:
        dst_real = dst.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise SystemExit(
            f"refusing conversion: destination path is not usable: {dst} ({e})")
    if _paths_overlap(src_real, dst_real):
        raise SystemExit(
            f"refusing conversion: source and destination overlap "
            f"({src_real} and {dst_real})")
    return src_real, dst


def preflight_full_conversion():
    """Require a working MLX quantizer before the first output bank is built."""
    try:
        triples = mlx_affine_quant(
            np.arange(64, dtype=np.float32).reshape(1, 64),
            8, group_size=64)
        if not isinstance(triples, (tuple, list)) or len(triples) != 3:
            raise RuntimeError("mlx_affine_quant returned an invalid triple set")
        for triple in triples:
            if len(triple) != 3 or not triple[1] or not triple[2]:
                raise RuntimeError("mlx_affine_quant returned an empty tensor")
    except Exception as e:
        raise SystemExit(
            f"refusing conversion: MLX quantization preflight failed: {e}")


def convert(src, dst, dry_run=False):
    src, dst = Path(src).expanduser(), Path(dst).expanduser()
    if not dry_run:
        src, dst = validate_conversion_paths(src, dst)
    hf_cfg = json.loads((src / "config.json").read_text())
    dims = Dims.from_config(hf_cfg)
    wm, shards, plan, dropped, errors, total = collect_plan(src, dims)
    if errors:
        for e in errors:
            print(f"  FAIL {e}", file=sys.stderr)
        raise SystemExit(f"{len(errors)} plan errors — nothing written")
    print_report(dims, len(wm), plan, dropped, total)
    if dry_run:
        return 0

    preflight_full_conversion()
    dst.mkdir(parents=True, exist_ok=True)
    state_path = dst / ".convert_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "banks": [], "texts": [], "mtp": False,
        "out_idx": 0, "out_map": {}, "total": 0}
    for f in COPY_FILES:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    if (src / "dflash").is_dir():
        (dst / "dflash").mkdir(exist_ok=True)
        for f in sorted((src / "dflash").iterdir()):
            target = dst / "dflash" / f.name
            if not target.exists():
                try:
                    os.link(f, target)
                except OSError:
                    shutil.copy2(f, target)

    def write_out(tensors, note):
        state["out_idx"] += 1
        fname = f"model-{state['out_idx']:05d}.safetensors"
        write_safetensors_raw(str(dst / fname), tensors)
        n = sum(len(t[2]) for t in tensors.values())
        for k in tensors:
            state["out_map"][k] = fname
        state["total"] += n
        state_path.write_text(json.dumps(state))
        print(f"  wrote {fname} {n / 1e9:.2f} GB ({len(tensors)} tensors) {note}",
              flush=True)

    # Expert banks: one output file per (layer, proj). Bank bytes are the plain
    # concatenation of per-expert slices — native mxfp4, byte identity holds.
    for layer in range(1, dims.layers):
        for proj in ("gate", "up", "down"):
            unit = f"bank:{layer}:{proj}"
            if unit in state["banks"]:
                continue
            out_dim, in_dim = bank_dims(dims, proj)
            slices = {"weight": [], "scales": []}
            for expert in range(dims.experts):
                for part, src_part in (("weight", "weight"),
                                       ("scales", "weight_scale")):
                    key = (f"model.layers.{layer}.mlp.experts.{expert}."
                           f"{proj}_proj.{src_part}")
                    header, data_off = shards[wm[key]]
                    slices[part].append(
                        read_bytes(src / wm[key], data_off, header[key]))
            bank = stack_expert_bank(slices["weight"], slices["scales"],
                                     dims.experts, out_dim, in_dim)
            base = f"model.layers.{layer}.mlp.switch_mlp.{proj}_proj"
            write_out({base + ".weight": bank["weight"],
                       base + ".scales": bank["scales"]}, f"({unit})")
            state["banks"].append(unit)
            state_path.write_text(json.dumps(state))
            del bank, slices

    def convert_unit(key, action, payload):
        """One source tensor -> {pack name: triple} (fp8 pairs via .weight)."""
        fname = wm[key]
        header, data_off = shards[fname]
        meta = header[key]
        if action in ("verbatim", "mtp_verbatim"):
            return {payload: (meta["dtype"], tuple(meta["shape"]),
                              read_bytes(src / fname, data_off, meta))}
        if key.endswith(".weight_scale_inv"):
            return {}  # consumed with its .weight partner
        base = key[:-len(".weight")]
        s_key = base + ".weight_scale_inv"
        s_header, s_off = shards[wm[s_key]]
        raw_w = read_raw(src / fname, data_off, meta)
        raw_s = read_raw(src / wm[s_key], s_off, s_header[s_key])
        if len(payload) == 3:  # fused qkv -> q/k/v
            nq, nkv, hd, vd = qkv_geometry_for_source(key, dims)
            parts = dequant_fp8_qkv(raw_w, raw_s, nq * hd, nkv * hd, nkv * vd)
        else:
            parts = (dequant_fp8_blocks(raw_w, raw_s),)
        out = {}
        for ob, part in zip(payload, parts):
            triples = mlx_affine_quant(part, 8, group_size=64)
            out[ob + ".weight"], out[ob + ".scales"], out[ob + ".biases"] = triples
        return out

    # Trunk text tensors (flushed into ~2 GiB shards), then the MTP sidecar.
    done_texts = set(state["texts"])
    out, out_bytes, pending = {}, 0, []
    for key in sorted(plan):
        action, payload = plan[key]
        if action not in ("verbatim", "fp8_affine") or key in done_texts:
            continue
        for name, triple in convert_unit(key, action, payload).items():
            out[name] = triple
            out_bytes += len(triple[2])
        pending.append(key)
        if out_bytes >= SHARD_BYTES:
            state["texts"].extend(pending)
            write_out(dict(out), "")
            out, out_bytes, pending = {}, 0, []
    if out:
        state["texts"].extend(pending)
        write_out(dict(out), "")

    if not state["mtp"]:
        mtp_out = {}
        for key in sorted(plan):
            action, payload = plan[key]
            if action in ("mtp_verbatim", "mtp_fp8_affine"):
                mtp_out.update(convert_unit(key, action, payload))
        (dst / "mtp").mkdir(exist_ok=True)
        write_safetensors_raw(str(dst / "mtp" / "weights.safetensors"), mtp_out)
        state["mtp"] = True
        state_path.write_text(json.dumps(state))
        n = sum(len(t[2]) for t in mtp_out.values())
        print(f"  wrote mtp/weights.safetensors {n / 1e6:.0f} MB "
              f"({len(mtp_out)} tensors)", flush=True)

    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": state["total"]},
         "weight_map": state["out_map"]}, indent=2))
    (dst / "config.json").write_text(json.dumps(build_config(hf_cfg), indent=2))
    (dst / "README.md").write_text(README.format(
        repo_name=dst.name, total_gb=state["total"] / 1e9,
        n_visual=dropped.get("visual.", 0),
        n_audio=dropped.get("audio_encoder.", 0),
        n_speech=dropped.get("speech_embeddings.", 0)))
    state_path.unlink(missing_ok=True)
    print(f"done: {state['total'] / 1e9:.1f} GB across {state['out_idx']} shards "
          f"+ mtp/weights.safetensors; dropped (v1): {sum(dropped.values())} "
          f"vision/audio/speech tensors")
    return 0


# ── self-test (hermetic: synthetic tensors only, no ckpt/mlx/torch) ─────────

def mk_dims():
    """Tiny but geometrically valid dims (mxfp4 blocks of 32 divide every `in`)."""
    return Dims(layers=3, hidden=96, intermediate=128, moe_intermediate=64,
                experts=3, nq=4, kv_ga=1, kv_swa=2, qk_dim=6, v_dim=4,
                ga_layers=(0,))


def mxfp4_decode_u8(raw, out_dim, in_dim):
    """Reference decode of HF's packed U8 rows (low nibble = the even element)."""
    b = np.frombuffer(raw, np.uint8).reshape(out_dim, in_dim // 2)
    lo = E2M1_TABLE[b & 0x0F]
    hi = E2M1_TABLE[(b >> 4) & 0x0F]
    return np.stack([lo, hi], -1).reshape(out_dim, in_dim)


def mxfp4_decode_u32(raw, out_dim, in_dim):
    """Reference decode of MLX's packed-U32 rows (element i at bit offset 4i)."""
    w = np.frombuffer(raw, np.uint32).reshape(out_dim, in_dim // 8)
    shifts = (4 * np.arange(8, dtype=np.uint32)).astype(np.uint32)
    codes = ((w[..., None] >> shifts) & np.uint32(0x0F)).astype(np.int64)
    return E2M1_TABLE[codes].reshape(out_dim, in_dim)


class Mxfp4DecodeTests(unittest.TestCase):
    def test_u8_view_and_u32_relabel_decode_identically(self):
        rng = np.random.default_rng(0)
        out_dim, in_dim = 3, 96
        raw = rng.integers(0, 256, size=out_dim * in_dim // 2,
                           dtype=np.uint8).tobytes()
        np.testing.assert_array_equal(mxfp4_decode_u32(raw, out_dim, in_dim),
                                      mxfp4_decode_u8(raw, out_dim, in_dim))
        # spot values: byte 0x30 = codes (0, 3) = (0.0, 1.5) low-nibble-first
        raw = (b"\x30" + b"\x00" * (out_dim * in_dim // 2 - 1))
        got = mxfp4_decode_u32(raw, out_dim, in_dim)
        self.assertEqual(got[0, 0], 0.0)
        self.assertEqual(got[0, 1], 1.5)

    def test_odd_rows_and_word_counts_stay_exact(self):
        rng = np.random.default_rng(1)
        for out_dim, in_dim in ((3, 40), (1, 8), (5, 96)):
            raw = rng.integers(0, 256, size=out_dim * in_dim // 2,
                               dtype=np.uint8).tobytes()
            np.testing.assert_array_equal(mxfp4_decode_u32(raw, out_dim, in_dim),
                                          mxfp4_decode_u8(raw, out_dim, in_dim))

    def test_e8m0_scale_is_a_power_of_two(self):
        self.assertEqual(E8M0_LUT[127], 1.0)
        self.assertEqual(E8M0_LUT[129], 4.0)
        self.assertEqual(E8M0_LUT[125], 0.25)
        # full mxfp4 cell: e2m1 value times the block's e8m0 scale
        raw = b"\x70" + b"\x00" * 7  # codes (0, 7) = (0.0, 6.0), in=16
        val = mxfp4_decode_u32(raw, 1, 16)[0, 1] * E8M0_LUT[129]
        self.assertEqual(val, 24.0)


class ExpertStackTests(unittest.TestCase):
    def test_stacking_is_byte_identity_with_expert_leading_index(self):
        rng = np.random.default_rng(2)
        experts, out_dim, in_dim = 3, 5, 96
        slices_w = [rng.integers(0, 256, size=out_dim * in_dim // 2,
                                 dtype=np.uint8).tobytes()
                    for _ in range(experts)]
        slices_s = [rng.integers(0, 256, size=out_dim * in_dim // 32,
                                 dtype=np.uint8).tobytes()
                    for _ in range(experts)]
        bank = stack_expert_bank(slices_w, slices_s, experts, out_dim, in_dim)
        wdt, wshape, wraw = bank["weight"]
        sdt, sshape, sraw = bank["scales"]
        self.assertEqual(wdt, "U32")
        self.assertEqual(wshape, (experts, out_dim, in_dim // 8))
        self.assertEqual(sdt, "U8")
        self.assertEqual(sshape, (experts, out_dim, in_dim // 32))
        self.assertEqual(wraw, b"".join(slices_w))  # byte identity
        self.assertEqual(sraw, b"".join(slices_s))
        # per-expert span (tensor_bytes / E) lands on the slice boundaries
        slot = len(wraw) // experts
        for e in range(experts):
            self.assertEqual(wraw[e * slot:(e + 1) * slot], slices_w[e])

    def test_byte_plan_is_computable_from_shapes(self):
        d = mk_dims()
        for proj, out_dim, in_dim in (("gate", 64, 96), ("up", 64, 96),
                                      ("down", 96, 64)):
            (wdt, wshape), (sdt, sshape) = bank_shapes(d, proj)
            self.assertEqual(wshape, (d.experts, out_dim, in_dim // 8))
            self.assertEqual(sshape, (d.experts, out_dim, in_dim // 32))
            slot = out_dim * in_dim // 8 * 4 + out_dim * in_dim // 32
            self.assertEqual(expert_slot_bytes(d, proj), slot)
        # native mxfp4 footprint: packed weight + one e8m0 per 32 values
        self.assertEqual(expert_slot_bytes(d, "gate"),
                         64 * 96 // 2 + 64 * 96 // 32)


class Fp8DequantTests(unittest.TestCase):
    def test_tp_qkv_scale_alignment_and_reassembly(self):
        # Q128|K96|V64 per rank: K's final tile shares 32 rows with V.
        for tp in (4, 8):
            raw = np.full((tp * 288, 128), 0x38, np.uint8)  # fp8 1.0
            scales = np.arange(1, 3 * tp + 1, dtype=np.float32).reshape(3 * tp, 1)
            q, k, v = dequant_fp8_qkv(raw, scales, tp * 128, tp * 96, tp * 64)
            for rank in range(tp):
                np.testing.assert_array_equal(q[rank * 128:(rank + 1) * 128], 3 * rank + 1)
                np.testing.assert_array_equal(k[rank * 96:(rank + 1) * 96], 3 * rank + 2)
                np.testing.assert_array_equal(v[rank * 64:rank * 64 + 32], 3 * rank + 2)
                np.testing.assert_array_equal(v[rank * 64 + 32:(rank + 1) * 64], 3 * rank + 3)

    def test_128x128_scale_grid_dequant(self):
        block = 128
        rng = np.random.default_rng(3)
        rows, cols = 2 * block, 3 * block
        w_u8 = rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)
        s = (rng.random((2, 3)).astype(np.float32) + 0.5) * 3.0
        got = dequant_fp8_blocks(w_u8, s)
        ref = np.empty((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                ref[i, j] = E4M3_LUT[w_u8[i, j]] * s[i // block, j // block]
        np.testing.assert_array_equal(got, ref)
        # trailing scale rows are padding, not error: a fused qkv quantized with
        # its v block at the qk width keeps unused grid rows (GA 108 > 106)
        # The real GA qkv has 13568 rows (106 row blocks) but a 108x32 scale
        # grid.  Nonzero trailing rows are exporter padding, not weight data.
        padding = np.full((2, 3), 7.0, dtype=np.float32)
        padded = np.concatenate([s, padding], 0)
        np.testing.assert_array_equal(dequant_fp8_blocks(w_u8, padded), ref)

    def test_real_ga_qkv_row_count_ignores_two_nonzero_scale_rows(self):
        rows, cols, block = 13568, 128, 128
        w_u8 = np.ones((rows, cols), dtype=np.uint8)
        scales = np.ones((108, 1), dtype=np.float32)
        scales[106:] = 7.0
        got = dequant_fp8_blocks(w_u8, scales, block)
        np.testing.assert_array_equal(got, E4M3_LUT[w_u8])

    def test_qkv_split_row_boundaries(self):
        nq, nkv, qk, v = 4, 2, 6, 4
        rows = nq * qk + nkv * qk + nkv * v
        w = np.arange(rows * 5, dtype=np.float32).reshape(rows, 5)
        q, k, vv = split_qkv(w, nq, nkv, qk, v)
        np.testing.assert_array_equal(q, w[:nq * qk])
        np.testing.assert_array_equal(k, w[nq * qk:nq * qk + nkv * qk])
        np.testing.assert_array_equal(vv, w[nq * qk + nkv * qk:])
        # real dims: split points are 128-row aligned (fp8 blocks cut cleanly)
        d = Dims(layers=48, hidden=4096, intermediate=16384,
                 moe_intermediate=2048, experts=256, nq=64, kv_ga=4, kv_swa=8,
                 qk_dim=192, v_dim=128, ga_layers=(0, 5, 11))
        self.assertEqual(qkv_row_layout(d, 0), (12288, 768, 512))
        self.assertEqual(qkv_row_layout(d, 1), (12288, 1536, 1024))
        for layer in (0, 1):
            for r in qkv_row_layout(d, layer):
                self.assertEqual(r % 128, 0, (layer, r))


class ConfigTests(unittest.TestCase):
    def test_config_emission_round_trip(self):
        hf = {
            "model_type": "mimo_v2", "hidden_size": 4096,
            "num_hidden_layers": 48, "vocab_size": 152576,
            "intermediate_size": 16384, "moe_intermediate_size": 2048,
            "n_routed_experts": 256, "num_experts_per_tok": 8,
            "n_group": 1, "topk_group": 1, "topk_method": "noaux_tc",
            "scoring_func": "sigmoid", "norm_topk_prob": True,
            "moe_router_dtype": "bfloat16", "routed_scaling_factor": None,
            "moe_layer_freq": [0, 1, 1],
            "num_attention_heads": 64, "num_key_value_heads": 4,
            "head_dim": 192, "v_head_dim": 128,
            "swa_num_attention_heads": 64, "swa_num_key_value_heads": 8,
            "swa_head_dim": 192, "swa_v_head_dim": 128,
            "swa_rope_theta": 10000.0, "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.334,
            "attention_value_scale": 0.707,
            "add_swa_attention_sink_bias": True,
            "add_full_attention_sink_bias": False,
            "hybrid_layer_pattern": [0, 1, 1], "hybrid_block_size": None,
            "sliding_window": 128, "attention_chunk_size": 128,
            "layernorm_epsilon": 1e-6, "max_position_embeddings": 1048576,
            "num_nextn_predict_layers": 3,
            "eos_token_id": 151645, "bos_token_id": None,
            "pad_token_id": 151643, "tie_word_embeddings": False,
            "attention_projection_layout": "fused_qkv",
            "quantization_config": {"quant_method": "fp8", "fmt": "e4m3",
                                    "store_dtype": "mxfp4"},
            "vision_config": {"depth": 28}, "audio_config": {"group_size": 4},
            "processor_config": {"fps": 1.0},
            "auto_map": {"AutoConfig": "x"},
        }
        cfg = json.loads(json.dumps(build_config(hf)))  # round trip
        self.assertEqual(cfg["model_type"], "mimo_v2")
        self.assertEqual(cfg["hybrid_layer_pattern"], [0, 1, 1])
        self.assertEqual(cfg["head_dim"], 192)
        self.assertEqual(cfg["v_head_dim"], 128)
        self.assertEqual(cfg["swa_num_key_value_heads"], 8)
        self.assertEqual(cfg["swa_rope_theta"], 10000.0)
        self.assertEqual(cfg["partial_rotary_factor"], 0.334)
        self.assertEqual(cfg["attention_value_scale"], 0.707)
        self.assertTrue(cfg["add_swa_attention_sink_bias"])
        self.assertFalse(cfg["add_full_attention_sink_bias"])
        self.assertEqual(cfg["n_routed_experts"], 256)
        self.assertEqual(cfg["num_experts_per_tok"], 8)
        self.assertEqual(cfg["moe_intermediate_size"], 2048)
        self.assertEqual(cfg["scoring_func"], "sigmoid")
        self.assertTrue(cfg["norm_topk_prob"])
        self.assertEqual(cfg["routed_scaling_factor"], 1.0)  # null normalized
        self.assertEqual(cfg["eos_token_id"], 151645)
        self.assertEqual(cfg["pad_token_id"], 151643)
        self.assertEqual(cfg["attention_projection_layout"], "split")
        self.assertEqual(cfg["quantization"],
                         {"bits": 4, "group_size": 32, "mode": "mxfp4"})
        self.assertEqual(cfg["quantization_config"], cfg["quantization"])
        for gone in ("vision_config", "audio_config", "processor_config",
                     "auto_map"):
            self.assertNotIn(gone, cfg)


class MtpContractTests(unittest.TestCase):
    def test_mtp_qkv_split_uses_source_prefix_and_swa_heads(self):
        d = mk_dims()
        # MTP qkv is always SWA-shaped.  Its source name still carries the
        # model.mtp prefix while the emitted names use mtp.layers.
        rows = d.nq * d.qk_dim + d.kv_swa * d.qk_dim + d.kv_swa * d.v_dim
        fused = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
        parts = split_qkv_for_source(
            "model.mtp.layers.0.self_attn.qkv_proj.weight", fused, d)
        self.assertEqual([p.shape for p in parts], [(24, 3), (12, 3), (8, 3)])
        np.testing.assert_array_equal(parts[1], fused[24:36])
        np.testing.assert_array_equal(parts[2], fused[36:44])
        trunk_rows = d.nq * d.qk_dim + d.kv_ga * d.qk_dim + d.kv_ga * d.v_dim
        trunk = np.arange(trunk_rows * 3, dtype=np.float32).reshape(trunk_rows, 3)
        trunk_parts = split_qkv_for_source(
            "model.layers.0.self_attn.qkv_proj.weight", trunk, d)
        self.assertEqual([p.shape for p in trunk_parts],
                         [(24, 3), (6, 3), (4, 3)])

    def test_mtp_verbatim_shape_audit_uses_source_prefix(self):
        d = mk_dims()
        key = "model.mtp.layers.0.eh_proj.weight"
        with self.assertRaises(ValueError):
            check_meta(key, "mtp_verbatim",
                       {"dtype": "U8", "shape": [7]}, d)
        check_meta(key, "mtp_verbatim",
                   {"dtype": "BF16", "shape": [d.hidden, 2 * d.hidden]}, d)


class DestinationSafetyTests(unittest.TestCase):
    @staticmethod
    def _source(root):
        src = root / "source"
        src.mkdir()
        (src / "source-sentinel").write_bytes(b"source is unchanged")
        return src

    @staticmethod
    def _assert_refused(test, src, dst):
        with test.assertRaises(SystemExit) as cm:
            convert(src, dst)
        test.assertIn("refusing", str(cm.exception).lower())

    def test_nonempty_target_is_refused_before_source_read(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            root = Path(td)
            src = self._source(root)
            dst = root / "target"
            dst.mkdir()
            target_marker = dst / "target-sentinel"
            target_marker.write_bytes(b"target is unchanged")
            self._assert_refused(self, src, dst)
            self.assertEqual((src / "source-sentinel").read_bytes(),
                             b"source is unchanged")
            self.assertEqual(target_marker.read_bytes(), b"target is unchanged")

    def test_ancestor_descendant_and_symlink_overlap_are_refused_early(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            root = Path(td)
            ancestor = root / "ancestor"
            ancestor.mkdir()
            src = ancestor / "source"
            src.mkdir()
            marker = src / "source-sentinel"
            marker.write_bytes(b"source is unchanged")

            # A descendant does not exist yet, so this must be rejected by the
            # resolved-path overlap check rather than by target contents.
            self._assert_refused(self, src, src / "new-pack")

            # The ancestor is necessarily non-empty, but the error must still
            # be a destination refusal before config/shard reads.
            self._assert_refused(self, src, ancestor)

            link = root / "source-link"
            os.symlink(src, link, target_is_directory=True)
            self._assert_refused(self, src, link)
            self.assertEqual(marker.read_bytes(), b"source is unchanged")


class DependencyPreflightTests(unittest.TestCase):
    def test_full_conversion_preflight_exercises_mlx_quantize(self):
        from unittest import mock
        fake = (("U32", (1, 8), b"\0" * 32),
                ("BF16", (1, 1), b"\0" * 2),
                ("BF16", (1, 1), b"\0" * 2))
        with mock.patch.object(sys.modules[__name__], "mlx_affine_quant",
                               return_value=fake) as quant:
            preflight_full_conversion()
        quant.assert_called_once()

    def test_failed_quantizer_preflight_is_named(self):
        from unittest import mock
        with mock.patch.object(sys.modules[__name__], "mlx_affine_quant",
                               side_effect=RuntimeError("mlx unavailable")):
            with self.assertRaises(SystemExit) as cm:
                preflight_full_conversion()
        self.assertIn("quantization preflight", str(cm.exception))


class PlanTests(unittest.TestCase):
    def test_name_table_covers_every_category(self):
        d = mk_dims()
        cases = {
            "model.layers.1.mlp.experts.2.gate_proj.weight": "bank_slice",
            "model.layers.1.mlp.experts.2.gate_proj.weight_scale": "bank_slice",
            "model.layers.1.mlp.experts.2.down_proj.weight_scale": "bank_slice",
            "model.layers.0.self_attn.qkv_proj.weight": "fp8_affine",
            "model.layers.0.self_attn.qkv_proj.weight_scale_inv": "fp8_affine",
            "model.layers.0.mlp.gate_proj.weight": "fp8_affine",
            "model.layers.0.mlp.down_proj.weight_scale_inv": "fp8_affine",
            "model.layers.3.self_attn.o_proj.weight": "verbatim",
            "model.layers.3.self_attn.attention_sink_bias": "verbatim",
            "model.layers.3.input_layernorm.weight": "verbatim",
            "model.layers.3.post_attention_layernorm.weight": "verbatim",
            "model.layers.3.mlp.gate.weight": "verbatim",
            "model.layers.3.mlp.gate.e_score_correction_bias": "verbatim",
            "model.norm.weight": "verbatim",
            "model.embed_tokens.weight": "verbatim",
            "lm_head.weight": "verbatim",
            "model.mtp.layers.0.self_attn.qkv_proj.weight": "mtp_fp8_affine",
            "model.mtp.layers.0.mlp.gate_proj.weight_scale_inv": "mtp_fp8_affine",
            "model.mtp.layers.0.eh_proj.weight": "mtp_verbatim",
            "model.mtp.layers.1.final_layernorm.weight": "mtp_verbatim",
            "visual.blocks.0.attn.qkv.weight": "drop",
            "audio_encoder.projection.mlp.0.weight": "drop",
            "speech_embeddings.0.weight": "drop",
        }
        for key, want in cases.items():
            action, _ = plan_tensor(key, {"dtype": "U8", "shape": [1, 1]}, d)
            self.assertEqual(action, want, key)
        # payloads pin the output names / stack coordinates
        _, payload = plan_tensor(
            "model.layers.1.mlp.experts.2.down_proj.weight_scale",
            {"dtype": "U8", "shape": [1, 1]}, d)
        self.assertEqual(payload, (1, "down", 2, "scales"))
        _, payload = plan_tensor("model.mtp.layers.2.eh_proj.weight",
                                 {"dtype": "BF16", "shape": [1, 1]}, d)
        self.assertEqual(payload, "mtp.layers.2.eh_proj.weight")
        self.assertEqual(
            mtp_out_name("model.mtp.layers.0.self_attn.qkv_proj.weight"),
            "mtp.layers.0.self_attn.qkv_proj.weight")

    def test_required_source_set_catches_missing_root_and_qkv_pair(self):
        d = Dims(layers=3, hidden=96, intermediate=128, moe_intermediate=64,
                 experts=3, nq=4, kv_ga=1, kv_swa=2, qk_dim=6, v_dim=4,
                 ga_layers=(0,), mtp_layers=1)
        present = set(required_source_tensors(d))
        for missing in (
                "lm_head.weight",
                "model.layers.0.self_attn.qkv_proj.weight",
                "model.layers.0.self_attn.qkv_proj.weight_scale_inv",
                "model.mtp.layers.0.self_attn.qkv_proj.weight",
                "model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv"):
            self.assertIn(missing, present)
            self.assertIn(missing, missing_required_tensors(present - {missing}, d))

    def test_collect_plan_reports_required_names_missing_from_index(self):
        from tempfile import TemporaryDirectory
        from unittest import mock
        d = mk_dims()
        with TemporaryDirectory() as td:
            src = Path(td)
            (src / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {
                    "model.layers.0.self_attn.qkv_proj.weight": "shard"
                }
            }))
            (src / "shard").write_bytes(b"")
            with mock.patch.object(sys.modules[__name__], "read_header",
                                   return_value=({}, 0)):
                result = collect_plan(src, d)
            errors = result[4]
            self.assertIn("missing required tensor: lm_head.weight", errors)
            self.assertIn(
                "missing required tensor: "
                "model.layers.0.self_attn.qkv_proj.weight_scale_inv", errors)

    def test_unclassified_names_are_named_not_skipped(self):
        d = mk_dims()
        with self.assertRaises(ValueError) as cm:
            plan_tensor("model.layers.1.mlp.experts.0.gate_proj.mystery",
                        {"dtype": "U8", "shape": [1, 1]}, d)
        self.assertIn("gate_proj.mystery", str(cm.exception))


def self_test():
    import io
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    buf = io.StringIO()
    result = unittest.TextTestRunner(stream=buf, verbosity=0).run(suite)
    if not result.wasSuccessful():
        sys.stderr.write(buf.getvalue())
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="MiMo-V2.6 HF -> mlx-serve pack converter",
        epilog="Routed experts stay at their NATIVE mxfp4 e8m0 block-32 bytes "
               "(byte-identical, U8->U32 view relabel). fp8 linears are "
               "dequantized and re-quantized to the 8-bit affine pack "
               "convention; o_proj/embed/lm_head/norms/gates stay bf16. "
               "Vision + audio are DROPPED in v1 (manifest note in the pack "
               "README).")
    ap.add_argument("--src", default=None, help="local HF checkpoint dir (read-only)")
    ap.add_argument("--dst", default=None, help="output pack dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="name/shape audit against the source index (headers only)")
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic unit tests and exit")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.src or (not args.dst and not args.dry_run):
        ap.error("--src is required, and --dst unless --dry-run")
    return convert(os.path.expanduser(args.src),
                   os.path.expanduser(args.dst or "."),
                   dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())