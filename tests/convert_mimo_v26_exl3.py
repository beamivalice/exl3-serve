#!/usr/bin/env python3
"""Replace MiMo-V2.6-Flash's MXFP4 routed experts with stacked EXL3 trellis shards.

Input: the original MiMo-V2.6-Flash-RL checkpoint (`model_type: mimo_v2`, routed
experts stored MXFP4, one tensor per expert). Output: a pack whose trunk is the
SOURCE's own files, hard-linked, and whose routed experts are EXL3 at a
fractional rate (K 2.5 by default, TINY codebook), stacked `[E, ...]` per
projection so a gather kernel indexes expert e on axis 0.

Per MoE layer L and projection P one shard `model-exl3-L{LL}-{P}.safetensors`:
  model.layers.{L}.mlp.switch_mlp.{P}.trellis  U16 [E, in/16, out/16, 16*K]
  model.layers.{L}.mlp.switch_mlp.{P}.suh      F16 [E, in]
  model.layers.{L}.mlp.switch_mlp.{P}.svh      F16 [E, out]
(in, out) = (hidden, moe_intermediate) for gate/up, the transpose for down.

Default quantization is LDLQ under the imatrix Hessian ROTATED into the inner
basis the search works in, preceded by a per-expert global codebook-scale search;
`--quantizer direct` is the calibration-free path. `--window` (default 16) is the
codeword width the search hashes; the pack records it in `expert_quant.window`
and every decoder masks to it (the engine admits 8..16), so a narrower window
trades weight error for search time at the same bits per weight.

  python3 tests/convert_mimo_v26_exl3.py --self-test
  python3 tests/convert_mimo_v26_exl3.py \\
      --src /Users/beam/llm/models/MiMo-V2.6-Flash-RL --dst /path/to/out \\
      --imatrix /Users/beam/llm/models/calib/mimo-v2.6-flash-rl-imatrix.safetensors
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_dsv4_weights import write_safetensors_raw  # noqa: E402

K_DEFAULT = 2.5
CODEBOOK_DEFAULT = "tiny"
WINDOW_DEFAULT = 16
BATCH_EXPERTS_DEFAULT = 32
SCRATCH_GB_DEFAULT = 4.0
# `quantize_tiles_mlx` evaluates before it returns, so the accumulated seconds are
# GPU-busy wall time: what fraction of the run is the trellis search itself.
SEARCH_STATS = {"launches": 0, "tiles": 0, "seconds": 0.0,
                "min_launch_tiles": 1 << 62, "max_launch_tiles": 0}


def reset_search_stats() -> None:
    SEARCH_STATS.update(launches=0, tiles=0, seconds=0.0,
                        min_launch_tiles=1 << 62, max_launch_tiles=0)


def search_stats_snapshot() -> dict:
    out = dict(SEARCH_STATS)
    if out["launches"] == 0:
        out["min_launch_tiles"] = 0
    return out

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(weight|weight_scale)$"
)
SWITCH_FMT = "model.layers.{layer}.mlp.switch_mlp.{proj}"
NP_DTYPE = {
    "BF16": np.uint16, "F16": np.float16, "F32": np.float32, "F64": np.float64,
    "U16": np.uint16, "U32": np.uint32, "I32": np.int32, "I64": np.int64, "U8": np.uint8,
    "I8": np.int8, "F8_E4M3": np.uint8, "BOOL": np.bool_,
}


# ---------------------------------------------------------------- safetensors

def _read_header_raw(path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    return header, 8 + hlen


def read_header(path) -> tuple[dict, int]:
    header, data_off = _read_header_raw(path)
    header.pop("__metadata__", None)
    return header, data_off


def read_stamp(path) -> dict:
    """The shard's `__metadata__`: what settings wrote it."""
    meta = _read_header_raw(path)[0].get("__metadata__")
    return dict(meta) if isinstance(meta, dict) else {}


def read_raw(path, data_off: int, meta: dict) -> np.ndarray:
    b, e = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_off + b)
        raw = f.read(e - b)
    if len(raw) != e - b:
        raise RuntimeError(f"{path}: short read for {e - b} bytes")
    dt = NP_DTYPE.get(meta["dtype"])
    if dt is None:
        raise RuntimeError(f"{path}: unsupported dtype {meta['dtype']}")
    return np.frombuffer(raw, dtype=dt).reshape(meta["shape"])


# ------------------------------------------------------------------- MXFP4
#
# The authoritative layout is the one the engine feeds to
# `mlx_quantized_matmul(..., "mxfp4")` (src/mimo_quant_test.zig, origin/main):
#
#   weight      U8 [out, in/2]   — row-major over OUTPUT rows; byte b of row r
#                                  holds input columns 2b (LOW nibble) and
#                                  2b+1 (HIGH nibble). MLX reads the same bytes
#                                  as little-endian U32, which is this order.
#   weight_scale U8 [out, in/32] — one e8m0 exponent per 32 consecutive INPUT
#                                  columns of the same output row.
#
#   nibble -> E2M1: magnitude = [0, .5, 1, 1.5, 2, 3, 4, 6][code & 7],
#                   negated when code & 8 (code 8 is negative zero).
#   scale  -> E8M0: 0 => 0.0, 0xff => +inf, else 2^(code - 127).
#   W_hf[r, c] = e8m0(scale[r, c/32]) * e2m1(nibble(r, c))
#
# EXL3's public layout is (in_features, out_features), so the public weight is
# W_hf transposed.

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def e2m1_value(codes: np.ndarray) -> np.ndarray:
    """E2M1 magnitudes with the sign bit applied; `codes` are 4-bit nibbles."""
    c = np.asarray(codes, dtype=np.uint8)
    mag = E2M1[c & 0x7]
    return np.where((c & 0x8) != 0, -mag, mag).astype(np.float32)


def e8m0_value(codes: np.ndarray) -> np.ndarray:
    """E8M0 block exponents: 0 is zero, 0xff is infinity, else 2^(code-127)."""
    c = np.asarray(codes, dtype=np.uint8).astype(np.int32)
    with np.errstate(over="ignore"):
        out = np.exp2((c - 127).astype(np.float32))
    out = np.where(c == 0, np.float32(0.0), out)
    return np.where(c == 255, np.float32(np.inf), out).astype(np.float32)


def dequant_mxfp4(weight: np.ndarray, scale: np.ndarray, *, block: int = 32) -> np.ndarray:
    """MXFP4 [out, in/2] + [out, in/block] -> float32 [out, in]."""
    w = np.asarray(weight, dtype=np.uint8)
    s = np.asarray(scale, dtype=np.uint8)
    if w.ndim != 2 or s.ndim != 2:
        raise RuntimeError(f"expected 2-D mxfp4 pair, got {w.shape} and {s.shape}")
    out_rows, half = w.shape
    in_dim = half * 2
    if s.shape != (out_rows, in_dim // block):
        raise RuntimeError(f"scale {s.shape} does not cover {w.shape} at block {block}")
    nibbles = np.empty((out_rows, in_dim), dtype=np.uint8)
    nibbles[:, 0::2] = w & 0x0F
    nibbles[:, 1::2] = w >> 4
    out = e2m1_value(nibbles) * np.repeat(e8m0_value(s), block, axis=1)
    # e8m0 code 0xff is +inf and the 128-point Hadamard would spread it over a whole
    # block; real MiMo codes sit in the 110s, so refuse rather than pack garbage.
    if not np.isfinite(out).all():
        raise RuntimeError(f"mxfp4 block decoded to a non-finite value "
                           f"(scale codes {np.unique(s[~np.isfinite(e8m0_value(s))])})")
    return out


def public_from_mxfp4(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """EXL3 public weight (in_features, out_features) from the MXFP4 pair."""
    return np.ascontiguousarray(dequant_mxfp4(weight, scale).T)


# ------------------------------------------------------------------ K / rate

def parse_k(text: str | float | int) -> float | int:
    """A rate in 1/16-bit steps; integral rates stay ints so config `k` is 4, not 4.0."""
    value = float(text)
    scaled = value * 16.0
    n = int(round(scaled))
    if abs(scaled - n) > 1e-9:
        raise ValueError(f"K must be a multiple of 1/16, got {text}")
    if not 32 <= n <= 128:
        raise ValueError(f"K must be in [2, 8], got {text}")
    return n // 16 if n % 16 == 0 else n / 16.0


def validate_window(window: int, k) -> int:
    """The search hashes a `window`-bit codeword and every decoder must mask the sliding
    window to the SAME width — PonyExl3 437d49c made that a decode parameter, and the
    pack carries it in `expert_quant.window` (absent = 16). A narrower window is a real
    lever now, but a pack whose window the decoder does not know still reads as noise,
    so it is part of the format and of the resume stamp."""
    _ensure_lib()
    from ponyexl3.ref.trellis import fresh_bit_pattern
    w = int(window)
    k_max = max(fresh_bit_pattern(k))
    if not (k_max < w <= 24 and w - k_max <= 16):
        raise RuntimeError(
            f"window {w} is outside the search's range for K={k}: it must exceed the "
            f"widest fresh-bit step ({k_max}) by 1..16 bits and stay at most 24")
    return w


def packed_hw(k: float | int) -> int:
    _ensure_lib()
    from ponyexl3.ref.trellis import packed_halfwords
    return packed_halfwords(k)


def k_from_packed(last: int) -> float | int:
    _ensure_lib()
    from ponyexl3.ref.trellis import rate_from_packed
    return rate_from_packed(last)


# ------------------------------------------------------------------ ponyexl3

def _ensure_lib() -> None:
    lib = os.environ.get("EXL3_CONVERT_LIB", "/Users/beam/llm/ponyexl3")
    if lib not in sys.path:
        sys.path.insert(0, lib)


def codebook_mode(codebook: str):
    _ensure_lib()
    from ponyexl3.ref.codebook import CodebookMode
    try:
        return {"mcg": CodebookMode.MCG, "mul1": CodebookMode.MUL1, "tiny": CodebookMode.TINY}[codebook]
    except KeyError:
        raise RuntimeError(f"unknown codebook {codebook!r}; expected mcg, mul1 or tiny") from None


def _tile_helpers():
    """PonyExl3's kernel-order tile permutation, the one `quantize_tiles_mlx` expects."""
    _ensure_lib()
    try:
        from ponyexl3.convert.direct import _inner_to_kernel_tiles, _kernel_tiles_to_inner
    except ImportError as exc:
        raise RuntimeError(f"ponyexl3 tile helpers unavailable: {exc}") from None
    return _inner_to_kernel_tiles, _kernel_tiles_to_inner


# Measured: throughput saturates at ~1024 tiles per launch and a wider one buys nothing,
# while the search's own 256 MiB default is ~63 tiles at K2.5 and leaves the GPU idle.
# The cap matters because PonyExl3 f981047 packs its backpointers (4.26 MB -> 1.11 MB of
# scratch per tile at K2.5), so the same budget would now ask for ~3.8x more tiles.
LAUNCH_TILES_CAP = 1024


def launch_tiles_for(k, window: int, scratch_gb: float) -> int:
    """Tiles per Metal launch: the smaller of the scratch budget and the saturation cap."""
    _ensure_lib()
    from ponyexl3.convert.metal_search import _scratch_bytes_per_tile
    per = _scratch_bytes_per_tile(k, window)
    return max(1, min(LAUNCH_TILES_CAP, int(scratch_gb * (1 << 30)) // per))


def search_tiles(tiles: np.ndarray, *, k, cb, window: int, chunk: int,
                 want_decoded: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """Trellis-search kernel-order tiles, `chunk` at a time. Returns (states, decoded).

    The chunking is ours rather than `quantize_tiles_mlx`'s so the results leave
    the GPU per launch instead of being concatenated there first."""
    _ensure_lib()
    import mlx.core as mx
    from ponyexl3.convert.metal_search import quantize_tiles_mlx
    n = int(tiles.shape[0])
    states = np.empty((n, 256), dtype=np.uint16)
    decoded = np.empty((n, 256), dtype=np.float32) if want_decoded else None
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        t0 = time.perf_counter()
        q, idx = quantize_tiles_mlx(
            mx.array(tiles[start:stop], dtype=mx.float32), k, cb,
            window=window, max_scratch_bytes=1 << 62,
        )
        SEARCH_STATS["seconds"] += time.perf_counter() - t0
        SEARCH_STATS["launches"] += 1
        SEARCH_STATS["tiles"] += stop - start
        SEARCH_STATS["min_launch_tiles"] = min(SEARCH_STATS["min_launch_tiles"], stop - start)
        SEARCH_STATS["max_launch_tiles"] = max(SEARCH_STATS["max_launch_tiles"], stop - start)
        if decoded is not None:
            decoded[start:stop] = np.array(q)
        states[start:stop] = np.array(idx).astype(np.uint16, copy=False)
        del q, idx
    return states, decoded


def pack_states(states: np.ndarray, k, in_tiles: int, out_tiles: int) -> np.ndarray:
    """(in_tiles*out_tiles, 256) codewords -> packed U16 [in_tiles, out_tiles, 16*K]."""
    _ensure_lib()
    from ponyexl3.ref.trellis import fresh_from_states, pack_trellis
    fresh = fresh_from_states(states.reshape(in_tiles, out_tiles, 256), k)
    return pack_trellis(fresh, k).astype(np.uint16, copy=False)


def quantize_inner_direct(inner: np.ndarray, *, k, cb, window: int, chunk: int):
    """One inner-domain matrix -> (packed, reconstructed inner). Fractional K."""
    to_tiles, from_tiles = _tile_helpers()
    rows, cols = inner.shape
    tiles = to_tiles(np.ascontiguousarray(inner, dtype=np.float32))
    states, decoded = search_tiles(tiles, k=k, cb=cb, window=window, chunk=chunk,
                                   want_decoded=True)
    packed = pack_states(states, k, rows // 16, cols // 16)
    return packed, from_tiles(decoded, rows, cols)


# ------------------------------------------------------------------ calibration

def diag_hessian(v: np.ndarray) -> np.ndarray:
    return np.diag(np.asarray(v, dtype=np.float32).reshape(-1))


def imatrix_expert_vector(flat: np.ndarray, expert: int, dim: int) -> np.ndarray:
    arr = np.asarray(flat, dtype=np.float32).reshape(-1)
    return arr[expert * dim : (expert + 1) * dim].copy()


def calib_for_expert(routed_tokens: int, moments: np.ndarray) -> tuple[str, np.ndarray | None]:
    """An expert no calibration token reached has no moments to weight; it falls back
    to the Gaussian prior, exactly as the qwen4_exp converter does."""
    if routed_tokens <= 0:
        return "ldlq-gaussian-256", None
    return "imatrix-diagonal", np.asarray(moments, dtype=np.float32).reshape(-1)


def imatrix_layer_keys(layer: int) -> tuple[str, str, str]:
    """src/imatrix.zig keys a MiMo capture by the SOURCE layer's expert-block prefix."""
    p = f"model.layers.{layer}.mlp.experts."
    return p + "gate_up_proj", p + "down_proj", p + "gate_up_proj.rows"


def imatrix_layer_complete(store: dict, layer: int) -> bool:
    return all(key in store for key in imatrix_layer_keys(layer))


def load_imatrix(path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file
    return load_file(str(path))


def ldl_factor(hessian: np.ndarray):
    _ensure_lib()
    from ponyexl3.convert.hessian import block_ldl, prepare_hessian_for_ldl
    prepared = prepare_hessian_for_ldl(hessian)
    l = block_ldl(prepared.hessian).l.astype(np.float32, copy=True)
    l[np.diag_indices_from(l)] = np.float32(0.0)
    return l


def ldl_is_feedbackless(l: np.ndarray) -> bool:
    return not np.any(l)


# --------------------------------------------- the imatrix in the INNER basis
#
# The imatrix diagonal is per PUBLIC input channel, but the trellis search runs on
# `regularize_public_weight`'s INNER matrix, whose rows are the public rows scaled by
# `suh` and then mixed by a 128-point Hadamard. PonyExl3 rotates its calibration
# activations the same way before forming a Hessian (`public_activations_to_inner` =
# `had_r_128(x, pre_scale=suh)`, used by `ldlq_quantize_layer`), and the quantization
# objective is basis-invariant, so the Hessian LDLQ needs is
#
#     H_inner = R^T diag(h) R,   R = blockdiag_b( diag(suh_b) @ H128 )
#
# which per 128-block is `H128 @ diag(suh_b**2 * h_b) @ H128` — DENSE. Feeding LDLQ the
# raw public diagonal instead would hand it an identity factor and silently discard the
# calibration. R is block diagonal, so H_inner is too: each 128-block factors alone, and
# with `buf_size_rows = 128` the LDLQ loop's cross-buffer term is identically zero.

HAD_BLOCK = 128


def hadamard_matrix(n: int = HAD_BLOCK) -> np.ndarray:
    _ensure_lib()
    from ponyexl3.ref.hadamard import hadamard_128, sylvester_hadamard, HAD_SCALE
    return hadamard_128(np.float32) if n == HAD_BLOCK else (
        sylvester_hadamard(n, np.float32) * np.float32(1.0 / np.sqrt(n)))


def inner_hessian_blocks(diag_public: np.ndarray, suh: np.ndarray,
                         block: int = HAD_BLOCK) -> np.ndarray:
    """The rotated Hessian, as its [nb, block, block] diagonal blocks."""
    h = np.asarray(diag_public, dtype=np.float32).reshape(-1)
    s = np.asarray(suh, dtype=np.float32).reshape(-1)
    if h.shape != s.shape:
        raise ValueError(f"imatrix diagonal {h.shape} does not match suh {s.shape}")
    if h.shape[0] % block:
        raise ValueError(f"{h.shape[0]} channels is not a multiple of {block}")
    hm = hadamard_matrix(block)
    d = np.maximum(h, 0.0) * (s * s)
    return np.einsum("ik,bk,kj->bij", hm, d.reshape(-1, block), hm, optimize=True).astype(np.float32)


def inner_ldl_blocks(blocks: np.ndarray, *, sigma_reg: float = 0.025,
                     ldl_block: int = 16, max_retries: int = 10) -> np.ndarray:
    """`prepare_hessian_for_ldl` + `block_ldl` restricted to each 128-block, batched in
    float32 through LAPACK instead of one 4096-row float64 Cholesky per expert. The
    damping is global (it reads the whole diagonal), the factorization is per block."""
    h = np.array(blocks, dtype=np.float32, copy=True)
    nb, n, _ = h.shape
    if n % ldl_block:
        raise ValueError(f"block size {n} is not a multiple of {ldl_block}")
    idx = np.arange(n)
    diag = h[:, idx, idx]
    dead = diag == 0
    if np.any(dead):
        h[:, idx, idx] = np.where(dead, np.float32(1.0), diag)
        diag = h[:, idx, idx]
    diag_mean = float(np.mean(diag)) if diag.size else 0.0
    if diag_mean > 0.0:
        h[:, idx, idx] += np.float32(sigma_reg * diag_mean)
    retries = 0
    while True:
        try:
            chol = np.linalg.cholesky(h.astype(np.float64)).astype(np.float32)
            break
        except np.linalg.LinAlgError:
            retries += 1
            if retries > max_retries or diag_mean <= 0.0:
                raise
            h[:, idx, idx] += np.float32(2.0 * sigma_reg * diag_mean)
    l = chol.copy()
    for i in range(n // ldl_block):
        c0, c1 = i * ldl_block, (i + 1) * ldl_block
        inv = np.linalg.inv(chol[:, c0:c1, c0:c1].astype(np.float64)).astype(np.float32)
        l[:, :, c0:c1] = np.matmul(l[:, :, c0:c1], inv)
        l[:, c0:c1, c0:c1] = np.eye(ldl_block, dtype=np.float32)
    l[:, idx, idx] = np.float32(0.0)      # the loop reads only the strict lower part
    return l


def public_from_inner_mlx(inner_mx, suh_mx, svh_mx):
    """`reconstruct_public_weights`' outer half, on a reconstruction we already hold —
    the search's own decoded inner is what a decoder produces, so no decode is needed."""
    _ensure_lib()
    from ponyexl3.mlx.hadamard import preapply_had_left_mlx, preapply_had_right_mlx
    w = preapply_had_left_mlx(inner_mx.astype(_mx().float32)) * suh_mx.reshape(-1, 1)
    return preapply_had_right_mlx(w) * svh_mx.reshape(1, -1)


def _mx():
    import mlx.core as mx
    return mx


def weighted_rel_err(public: np.ndarray, public_hat: np.ndarray,
                     channel_weights: np.ndarray) -> float:
    """`tests/dsv4_imatrix.weighted_rel_err`, on the EXL3 public layout: sum_i om_i
    (W_hat - W)^2 over sum_i om_i W^2, the channel weights indexing INPUT channels —
    which is the FIRST axis here and the last one in the HF layout that file uses."""
    om = np.asarray(channel_weights, dtype=np.float32).reshape(-1, 1)
    d = (public_hat - public).astype(np.float32)
    num = float((om * d * d).sum())
    den = float((om * public.astype(np.float32) ** 2).sum())
    return num / max(den, 1e-30)


def imatrix_ldl_blocks(diag_public: np.ndarray, suh: np.ndarray) -> np.ndarray:
    return inner_ldl_blocks(inner_hessian_blocks(diag_public, suh))


def prior_ldl_blocks(suh: np.ndarray) -> np.ndarray:
    """An expert no calibration token reached: a FLAT public Hessian, through the same
    rotation. It is not isotropic in the inner basis — |suh| varies inside a 128-block,
    so `H128 diag(suh**2) H128` is dense — which is exactly the structure the search
    should still see when the calibration has nothing to say about the channels."""
    return inner_ldl_blocks(inner_hessian_blocks(np.ones(np.size(suh), dtype=np.float32), suh))


def _search_mlx(tiles_mx, *, k, cb, window: int, scratch_bytes: int):
    """`quantize_tiles_mlx` with the stats bookkeeping; it evaluates before returning,
    so the accumulated seconds are GPU-busy time."""
    _ensure_lib()
    from ponyexl3.convert.metal_search import quantize_tiles_mlx, _scratch_bytes_per_tile
    n = int(tiles_mx.shape[0])
    per_launch = max(1, min(LAUNCH_TILES_CAP,
                            scratch_bytes // _scratch_bytes_per_tile(k, window)))
    t0 = time.perf_counter()
    decoded, states = quantize_tiles_mlx(
        tiles_mx, k, cb, window=window,
        max_scratch_bytes=per_launch * _scratch_bytes_per_tile(k, window))
    SEARCH_STATS["seconds"] += time.perf_counter() - t0
    launches = (n + per_launch - 1) // per_launch
    SEARCH_STATS["launches"] += launches
    SEARCH_STATS["tiles"] += n
    size = min(n, per_launch)
    SEARCH_STATS["min_launch_tiles"] = min(SEARCH_STATS["min_launch_tiles"], size)
    SEARCH_STATS["max_launch_tiles"] = max(SEARCH_STATS["max_launch_tiles"], size)
    return decoded, states


# --------------------------------------------------- the global codebook scale
#
# `regularize_public_weight` divides by `regularize.CODEBOOK_SCALE`, which is MCG's
# measured RMS (1.2437). TINY's is 1.6281, so regularizing for TINY and searching at
# g = 1 leaves the inner matrix off the codebook's own scale — measured +5.1% MSE on a
# real MiMo expert. PonyExl3's driver corrects for it with a golden-section search over
# a sampled tile diagonal (`sample_tile_matrix` / `g_scale_gss` / `apply_global_scale`,
# convert/direct.py); this is that search, GPU-resident and run for every expert in the
# batch at once. `apply_global_scale`'s bookkeeping is `inner *= g`, `suh /= g`, which
# leaves the reconstructed public weight unchanged — nothing new is stored on disk.

# The bracket is per codebook, around the ratio of its RMS to the regularizer's MCG
# constant (TINY 1.63/1.24 = 1.31, MUL1 1.00/1.24 = 0.80, MCG 1.0); the measured optimum
# sits below the pure ratio because the search is not a plain scale match. Narrow
# brackets plus tol 0.03 keep the whole search at 10 evaluations of 128 tiles — about
# 4% of the expert's own 32768, where PonyExl3's default (13 x 768) would be 30%.
G_SCALE_BRACKET = {"tiny": (0.7, 1.9), "mul1": (0.4, 1.2), "mcg": (0.5, 1.5)}
G_SCALE_TOL = 0.03
G_SCALE_WIDTH = 3
G_SCALE_TILES = 128
G_SCALE_MAX_EVALS = 10


def g_scale_iterations(low: float, high: float, tol: float = G_SCALE_TOL) -> int:
    """Golden section shrinks the bracket by a fixed factor, so every expert needs the
    same number of steps — which is what lets the whole batch run in lock-step."""
    resphi = 2.0 - (1.0 + math.sqrt(5.0)) / 2.0
    width, steps = high - low, 0
    while width > tol and steps < G_SCALE_MAX_EVALS - 2:
        width *= 1.0 - resphi
        steps += 1
    return steps


def sample_tile_index(rows: int, cols: int, *, count: int = G_SCALE_TILES,
                      width: int = G_SCALE_WIDTH) -> np.ndarray:
    """Flat offsets of PonyExl3's wrapped tile diagonal (`sample_tile_matrix`), thinned
    to `count` tiles: the score is one scalar per expert, so a couple of hundred tiles
    pin it, while the full diagonal would cost about half the expert's own search."""
    tk, tn = rows // 16, cols // 16
    pairs = [((i % tk) * 16, ((i + w) % tn) * 16)
             for i in range(max(tk, tn)) for w in range(width)]
    pairs = pairs[:: max(1, len(pairs) // count)][:count]
    r = np.arange(16, dtype=np.int64)
    idx = np.empty((len(pairs), 16, 16), dtype=np.int64)
    for t, (r0, c0) in enumerate(pairs):
        idx[t] = (r0 + r)[:, None] * cols + (c0 + r)[None, :]
    return idx.reshape(len(pairs), 256).astype(np.int32)


def sample_tiles_mlx(inner_mx, index: np.ndarray):
    """Gather [N, T, 256] row-major sample tiles out of the resident inner matrices."""
    import mlx.core as mx
    n_exp, rows, cols = inner_mx.shape
    flat = mx.take(inner_mx.reshape(n_exp, rows * cols),
                   mx.array(index.reshape(-1)), axis=1)
    return flat.reshape(n_exp, index.shape[0], 256)


def g_scale_search_mlx(tiles_mx, *, k, cb, codebook: str, window: int, scratch_bytes: int):
    """One global scale per expert, golden section in lock-step across the batch.

    `tiles_mx` is [N, T, 256] row-major 16x16 tiles. Each iteration scales every
    expert's sample by ITS own candidate and runs ONE batched search over all of them;
    the bracket shrinks by the same factor for every expert, so the step count is
    shared and only which endpoint moves differs."""
    _ensure_lib()
    import mlx.core as mx
    from ponyexl3.convert.direct import _TENSOR_CORE_PERM
    n_exp, n_tiles, _ = tiles_mx.shape
    kernel = mx.take(tiles_mx, mx.array(_TENSOR_CORE_PERM.astype(np.int32)), axis=2)

    def score(g):
        decoded, _ = _search_mlx((kernel * g.reshape(n_exp, 1, 1)).reshape(n_exp * n_tiles, 256),
                                 k=k, cb=cb, window=window, scratch_bytes=scratch_bytes)
        delta = decoded.reshape(n_exp, n_tiles, 256) / g.reshape(n_exp, 1, 1) - kernel
        return mx.mean(delta * delta, axis=(1, 2))

    low, high = G_SCALE_BRACKET[codebook]
    resphi = 2.0 - (1.0 + math.sqrt(5.0)) / 2.0
    a = mx.full((n_exp,), low, dtype=mx.float32)
    b = mx.full((n_exp,), high, dtype=mx.float32)
    x1, x2 = a + resphi * (b - a), b - resphi * (b - a)
    f1, f2 = score(x1), score(x2)
    for _ in range(g_scale_iterations(low, high)):
        left = f1 < f2                      # keep [a, x2]; else keep [x1, b]
        a_n, b_n = mx.where(left, a, x1), mx.where(left, x2, b)
        probe = mx.where(left, a_n + resphi * (b_n - a_n), b_n - resphi * (b_n - a_n))
        f_probe = score(probe)
        a, b = a_n, b_n
        x1, x2, f1, f2 = (mx.where(left, probe, x2), mx.where(left, x1, probe),
                          mx.where(left, f_probe, f2), mx.where(left, f1, f_probe))
        mx.eval(a, b, x1, x2, f1, f2)
    out = (a + b) / 2.0
    mx.eval(out)
    return out


def ldlq_group_mlx(inner_mx, lblocks_mx, *, k, cb, window: int, scratch_bytes: int,
                   block: int = HAD_BLOCK, feedback_rows: int = 16,
                   want_recon: bool = False):
    """Reverse LDLQ over a whole batch of experts, resident on the GPU.

    `inner_mx` is [N, rows, cols]; `lblocks_mx` is [N, nb, block, block], the strictly
    lower part of each 128-channel block's LDL factor. Row buffers, reconstruction, the
    compensation GEMMs and the packed trellis all stay MLX arrays — only the finished
    trellis leaves the device. Every feedback step's search is ONE call over all N
    experts' rows (PonyExl3's `ldlq_quantize_group` batches sibling linears the same way);
    the rotation is block diagonal, so buffers are the Hadamard's own 128 rows and the
    cross-buffer compensation term of `_ldlq_inner_matrix_mlx` is identically zero."""
    _ensure_lib()
    import mlx.core as mx
    from ponyexl3.convert.direct import _inner_to_kernel_tiles_mlx, _kernel_tiles_to_inner_mlx
    from ponyexl3.convert.mlx_trellis import pack_trellis_mlx

    n_exp, rows, cols = inner_mx.shape
    nb = rows // block
    if lblocks_mx.shape != (n_exp, nb, block, block):
        raise ValueError(f"LDL blocks {lblocks_mx.shape} do not match {(n_exp, nb, block, block)}")
    out_tiles = cols // 16
    rows_per_step = feedback_rows
    packed_rows: list = [None] * (rows // 16)
    recon_rows: list = [None] * (rows // 16) if want_recon else []
    for b in range(nb - 1, -1, -1):
        lo = b * block
        w = inner_mx[:, lo : lo + block]
        l = lblocks_mx[:, b]
        recon = mx.zeros_like(w)
        for bj in range(block, 0, -rows_per_step):
            bi = bj - rows_per_step
            step = w[:, bi:bj]
            if bj < block:
                err = w[:, bj:] - recon[:, bj:]
                step = step + mx.matmul(mx.swapaxes(l[:, bj:, bi:bj], 1, 2), err)
            flat = step.reshape(n_exp * rows_per_step, cols)
            decoded, states = _search_mlx(_inner_to_kernel_tiles_mlx(flat), k=k, cb=cb,
                                          window=window, scratch_bytes=scratch_bytes)
            packed = pack_trellis_mlx(
                states.reshape(n_exp * (rows_per_step // 16), out_tiles, 256), k)
            back = _kernel_tiles_to_inner_mlx(decoded, n_exp * rows_per_step, cols)
            recon = mx.slice_update(recon, back.reshape(n_exp, rows_per_step, cols),
                                    start_indices=mx.array([0, bi, 0], dtype=mx.int32),
                                    axes=(0, 1, 2))
            mx.eval(recon, packed)
            for r in range(rows_per_step // 16):
                packed_rows[(lo + bi) // 16 + r] = packed[r :: rows_per_step // 16]
            if want_recon:
                shaped = back.reshape(n_exp, rows_per_step, cols)
                for r in range(rows_per_step // 16):
                    recon_rows[(lo + bi) // 16 + r] = shaped[:, r * 16 : (r + 1) * 16]
                mx.eval(*[x for x in recon_rows if x is not None][-rows_per_step // 16:])
            del decoded, states, back
        del recon, w, l
    stacked = mx.stack(packed_rows, axis=1)      # [N, rows/16, out_tiles, 16*K]
    mx.eval(stacked)
    packed_np = np.array(stacked).astype(np.uint16, copy=False)
    if not want_recon:
        return packed_np
    recon = mx.concatenate(recon_rows, axis=1)   # [N, rows, cols]
    mx.eval(recon)
    return packed_np, recon


def ldlq_group(inners, factors, *, k, cb, window: int, chunk: int,
               buf_size_rows: int = 128, feedback_rows: int = 16) -> np.ndarray:
    """Reverse 16-row LDLQ over several experts in LOCK-STEP.

    Each expert keeps its own L, compensation and reconstruction; at every feedback
    step their rows are concatenated so the Metal search runs ONCE over the whole
    group instead of once per expert per 16-row block (PonyExl3's
    `ldlq_quantize_group` does the same for sibling linears). With one expert this
    is exactly `ldlq_inner_matrix`'s loop."""
    to_tiles, from_tiles = _tile_helpers()
    n = len(inners)
    rows, cols = inners[0].shape
    weight = np.stack([np.ascontiguousarray(x, dtype=np.float32) for x in inners])
    recon = np.zeros_like(weight)
    prod = np.zeros_like(weight)
    packed = np.empty((n, rows // 16, cols // 16, packed_hw(k)), dtype=np.uint16)
    j = rows
    while j > 0:
        i = max(0, j - buf_size_rows)
        for bj in range(j - i, 0, -feedback_rows):
            bi = max(0, bj - feedback_rows)
            lo, hi = i + bi, i + bj
            blocks = []
            for e in range(n):
                comp = prod[e, lo:hi].copy()
                err = weight[e, hi:j] - recon[e, hi:j]
                if err.size:
                    comp += factors[e][hi:j, lo:hi].T @ err
                blocks.append(weight[e, lo:hi] + comp)
            tiles = np.concatenate([to_tiles(b) for b in blocks], axis=0)
            states, decoded = search_tiles(tiles, k=k, cb=cb, window=window,
                                           chunk=chunk, want_decoded=True)
            per = ((hi - lo) // 16) * (cols // 16)
            for e in range(n):
                sl = slice(e * per, (e + 1) * per)
                packed[e, lo // 16 : hi // 16] = pack_states(
                    states[sl], k, (hi - lo) // 16, cols // 16)
                recon[e, lo:hi] = from_tiles(decoded[sl], hi - lo, cols)
        if i > 0:
            for e in range(n):
                prod[e, :i] += factors[e][i:j, :i].T @ (weight[e, i:j] - recon[e, i:j])
        j = i
    return packed


# ------------------------------------------------------------- expert batches

def expert_seed(layer: int, expert: int, proj: str) -> int:
    """Stable per (layer, expert, projection), so `--layers` and `--resume` runs
    reproduce the same pack as a whole-model run."""
    return (layer * 1_000_003 + expert * 17 + PROJECTIONS.index(proj)) & 0x7FFF_FFFF


class PreparedBank(NamedTuple):
    """The host-only half of one expert batch."""
    inner: np.ndarray            # [N, rows, cols] regularized, PRE global scale
    suh: np.ndarray              # [N, rows] float32, PRE global scale
    svh: np.ndarray              # [N, cols] float32
    factor: np.ndarray | None    # [N, nb, 128, 128] LDL blocks, None for --quantizer direct
    fallbacks: int


def prepare_expert_bank(publics, seeds, calibrations, *, quantizer: str) -> PreparedBank:
    """Regularize each expert and build its LDL factor. Pure host work, so it runs on
    the prefetch thread while the GPU searches the previous batch.

    The factor is built from the PRE-scale `suh` deliberately: the global scale takes
    suh -> suh/g, which takes the rotated Hessian to H/g**2, and `block_ldl`'s
    unit-diagonal factor is invariant under a positive scalar (the Cholesky's sqrt(c)
    cancels against the inverse of its own diagonal block, and the sigma_reg damping
    scales with the diagonal). So the LDL need not wait for the scale search."""
    _ensure_lib()
    from ponyexl3.convert.regularize import regularize_public_weight
    regs = [regularize_public_weight(np.asarray(p, dtype=np.float32), seed=s)
            for p, s in zip(publics, seeds)]
    inner = np.stack([r.inner for r in regs])
    suh = np.stack([r.suh for r in regs], axis=0)
    svh = np.stack([r.svh for r in regs], axis=0)
    del regs
    if quantizer == "direct":
        return PreparedBank(inner, suh, svh, None, 0)
    blocks, fallbacks = [], 0
    for ei, cal in enumerate(calibrations):
        if cal is None:
            fallbacks += 1
            blocks.append(prior_ldl_blocks(suh[ei]))
        else:
            blocks.append(imatrix_ldl_blocks(cal, suh[ei]))
    return PreparedBank(inner, suh, svh, np.stack(blocks), fallbacks)


def quantize_prepared_bank(
    prep: PreparedBank, publics, calibrations, *,
    k, codebook: str, window: int, scratch_bytes: int,
    g_scale: bool = True, scale_out: list | None = None, err_out: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """The GPU half: global scale, LDLQ, and the quality read-out."""
    _ensure_lib()
    import mlx.core as mx
    cb = codebook_mode(codebook)
    n, rows, cols = prep.inner.shape
    inner = mx.array(prep.inner, dtype=mx.float32)
    suh = prep.suh

    if g_scale:
        g = g_scale_search_mlx(sample_tiles_mlx(inner, sample_tile_index(rows, cols)),
                               k=k, cb=cb, codebook=codebook, window=window,
                               scratch_bytes=scratch_bytes)
        host = np.array(g).astype(np.float32)
        # `apply_global_scale`: (inner * g, suh / g) reconstructs the same public weight,
        # so the scale needs no field of its own on disk.
        inner = inner * g.reshape(n, 1, 1)
        suh = suh / host[:, None]
        if scale_out is not None:
            scale_out.extend(float(v) for v in host)

    factor = (mx.zeros((n, rows // HAD_BLOCK, HAD_BLOCK, HAD_BLOCK), dtype=mx.float32)
              if prep.factor is None else mx.array(prep.factor, dtype=mx.float32))
    want_err = err_out is not None
    result = ldlq_group_mlx(inner, factor, k=k, cb=cb, window=window,
                            scratch_bytes=scratch_bytes, want_recon=want_err)
    del inner, factor
    suh16 = suh.astype(np.float16)
    svh16 = prep.svh.astype(np.float16)
    if want_err:
        trellis, recon = result
        suh_mx = mx.array(suh16.astype(np.float32))
        svh_mx = mx.array(svh16.astype(np.float32))
        for ei in range(n):
            hat = np.array(public_from_inner_mlx(recon[ei], suh_mx[ei], svh_mx[ei]))
            w = calibrations[ei] if calibrations[ei] is not None else np.ones(rows, np.float32)
            err_out.append(weighted_rel_err(publics[ei], hat, w))
        del recon, suh_mx, svh_mx
    else:
        trellis = result
    # ONE clear per expert bank. Clearing per launch measured 1.27x slower on the real
    # shape: the split-scratch launch allocates two buffer shapes per call, and dropping
    # the pool between launches makes every one of them a fresh allocation.
    mx.clear_cache()
    return trellis, suh16, svh16, prep.fallbacks


def quantize_expert_bank(
    publics: list[np.ndarray],
    seeds: list[int],
    calibrations: list[np.ndarray | None],
    *,
    k, codebook: str, window: int, quantizer: str, scratch_bytes: int,
    g_scale: bool = True, scale_out: list | None = None, err_out: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Both halves back to back — what a caller outside the converter's pipeline wants."""
    prep = prepare_expert_bank(publics, seeds, calibrations, quantizer=quantizer)
    return quantize_prepared_bank(prep, publics, calibrations, k=k, codebook=codebook,
                                  window=window, scratch_bytes=scratch_bytes,
                                  g_scale=g_scale, scale_out=scale_out, err_out=err_out)


# ------------------------------------------------------------------- pack plan

def layer_proj_shard(layer: int, proj: str) -> str:
    return f"model-exl3-L{layer:02d}-{proj}.safetensors"


def switch_base(layer: int, proj: str) -> str:
    return SWITCH_FMT.format(layer=layer, proj=proj)


def exl3_keys(base: str) -> tuple[str, str, str]:
    return base + ".trellis", base + ".suh", base + ".svh"


def parse_expert_key(key: str):
    m = EXPERT_RE.match(key)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3) + "_proj", m.group(4)


def plan_source(index: dict) -> dict:
    """Classify the source shards. A file holding ONLY routed experts is dropped; any
    file with a non-expert tensor is hard-linked whole and its expert keys simply leave
    the index — a hard link costs no bytes, so the trunk shards stay untouched."""
    weight_map = index["weight_map"]
    files: dict[str, dict[str, list[str]]] = {}
    for key, fname in weight_map.items():
        slot = files.setdefault(fname, {"expert": [], "other": []})
        slot["expert" if parse_expert_key(key) else "other"].append(key)
    hardlink = sorted(f for f, s in files.items() if s["other"])
    drop = sorted(f for f, s in files.items() if not s["other"])
    mixed = sorted(f for f, s in files.items() if s["other"] and s["expert"])
    keep_keys = [k for k, f in weight_map.items() if parse_expert_key(k) is None]
    return {"hardlink": hardlink, "drop": drop, "mixed": mixed, "keep_keys": keep_keys,
            "files": files}


def moe_layers(config: dict) -> list[int]:
    freq = config.get("moe_layer_freq")
    layers = int(config["num_hidden_layers"])
    if not isinstance(freq, list) or len(freq) != layers:
        raise RuntimeError(f"moe_layer_freq must list {layers} entries")
    return [i for i, f in enumerate(freq) if int(f) != 0]


def parse_layer_range(text: str | None, available: list[int]) -> list[int]:
    if not text:
        return list(available)
    allowed = set(available)
    picked: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            lo, hi = part.split("-", 1)
            span = range(int(lo), int(hi) + 1)
        else:
            span = [int(part)]
        for layer in span:
            if layer not in allowed:
                raise RuntimeError(f"layer {layer} is not a MoE layer of this checkpoint")
            if layer not in picked:
                picked.append(layer)
    return sorted(picked)


def expert_quant_block(*, k, codebook: str, window: int, quantizer: str, calibration: str,
                       source: str = "convert", **extra) -> dict:
    block = {"format": "exl3", "k": k, "codebook": codebook, "window": int(window),
             "out_scales": "svh", "source": source, "quantizer": quantizer,
             "calibration": calibration}
    block.update(extra)
    return block


def rewrite_config(config: dict, block: dict) -> dict:
    """Carry the EXL3 block and strip the MXFP4 claim: nothing in the output pack is
    MXFP4 any more, so `store_dtype` and the block size must not survive to be read
    as a routed-expert layout. The removed value is recorded inside `expert_quant`."""
    cfg = json.loads(json.dumps(config))
    quant = cfg.get("quantization_config")
    replaced = None
    if isinstance(quant, dict):
        replaced = quant.pop("store_dtype", None)
        quant.pop("mxfp4_block_size", None)
    out_block = dict(block)
    if replaced is not None:
        out_block["replaced_store_dtype"] = replaced
    cfg["expert_quant"] = out_block
    return cfg


def format_k(k) -> str:
    """The rate as the stamp and the config spell it: "2.5", "4"."""
    return str(int(k)) if float(k).is_integer() else repr(float(k))


def file_sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# Bump on any change to the pack format or to what the quantizer produces — never for a
# comment or a test. A shard's stamp carries this, and `--resume` refuses a shard whose
# stamp differs, so hashing the file itself would make an editorial change cost a rerun.
CONVERTER_VERSION = "mimo-exl3-2-ldlq-rotated-gss"


def converter_version() -> str:
    return CONVERTER_VERSION


QUANTIZER_STAMP = {"direct": "direct", "ldlq": "ldlq-rotated"}


def shard_stamp(*, k, codebook: str, window: int, quantizer: str,
                imatrix_sha: str | None, g_scale: str = "gss") -> dict[str, str]:
    """What a shard was made with. Every value is a string: safetensors `__metadata__`
    is a string map, and `--resume` compares the whole dict."""
    return {
        "format": "exl3",
        "k": format_k(k),
        "codebook": codebook,
        "window": str(int(window)),
        "quantizer": QUANTIZER_STAMP[quantizer],
        "g_scale": g_scale,
        "imatrix_sha256": imatrix_sha or "none",
        "converter": converter_version(),
    }


def shard_reuse_refusal(path, n_experts: int, in_dim: int, out_dim: int, k,
                        stamp: dict[str, str] | None = None) -> str | None:
    """None when `--resume` may keep this shard, else why it must be rewritten."""
    p = Path(path)
    if not p.is_file():
        return "absent"
    try:
        header, data_off = read_header(p)
    except Exception as exc:
        return f"unreadable header ({exc})"
    trellis = [key for key in header if key.endswith(".trellis")]
    if len(trellis) != 1:
        return f"{len(trellis)} trellis tensors, expected 1"
    if header[trellis[0]]["dtype"] != "U16":
        return f"trellis dtype {header[trellis[0]]['dtype']}, expected U16"
    shape = list(header[trellis[0]]["shape"])
    want = [n_experts, in_dim // 16, out_dim // 16, packed_hw(k)]
    if shape != want:
        return f"trellis shape {shape}, expected {want}"
    base = trellis[0][: -len(".trellis")]
    for suffix, want_shape in ((".suh", [n_experts, in_dim]), (".svh", [n_experts, out_dim])):
        meta = header.get(base + suffix)
        if meta is None:
            return f"missing {suffix.lstrip('.')}"
        if meta["dtype"] != "F16" or list(meta["shape"]) != want_shape:
            return f"{suffix.lstrip('.')} is {meta['dtype']}{list(meta['shape'])}, expected F16{want_shape}"
    end = max(meta["data_offsets"][1] for meta in header.values())
    if p.stat().st_size < data_off + end:
        return "truncated payload"
    if stamp is None:
        return None
    have = read_stamp(p)
    if not have:
        return "no stamp"
    differing = [f"{key}={have.get(key, '<absent>')} != {value}"
                 for key, value in sorted(stamp.items()) if have.get(key) != value]
    return "stamp " + ", ".join(differing) if differing else None


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError as exc:
        raise RuntimeError(
            f"cannot hard-link {src} -> {dst} ({exc.strerror}); the destination must "
            f"sit on the source's filesystem"
        ) from None


def stage_non_expert(src: Path, dst: Path, plan: dict) -> int:
    """Hard-link everything that is not a dropped expert shard: the trunk shards, the
    tokenizer, the chat template, the code files and the asset trees."""
    linked = 0
    dropped = set(plan["drop"])
    for root, dirs, names in os.walk(src):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        rel_root = Path(root).relative_to(src)
        for name in names:
            if name.startswith("."):
                continue
            rel = rel_root / name if str(rel_root) != "." else Path(name)
            if str(rel) in dropped or str(rel) in ("model.safetensors.index.json", "config.json"):
                continue
            source = Path(root) / name
            if not source.is_file() or source.is_symlink():
                continue
            _link(source, dst / rel)
            linked += 1
    return linked


# ---------------------------------------------------------------- source reads

class SourceReader:
    """Per-expert MXFP4 tensor reads against a cached set of shard headers."""

    def __init__(self, src: Path, weight_map: dict):
        self.src = Path(src)
        self.weight_map = weight_map
        self._headers: dict[str, tuple[dict, int]] = {}

    def header(self, fname: str):
        if fname not in self._headers:
            self._headers[fname] = read_header(self.src / fname)
        return self._headers[fname]

    def tensor(self, key: str) -> np.ndarray:
        fname = self.weight_map.get(key)
        if fname is None:
            raise RuntimeError(f"source index has no {key}")
        header, off = self.header(fname)
        return read_raw(self.src / fname, off, header[key])

    def public(self, layer: int, expert: int, proj: str) -> np.ndarray:
        base = f"model.layers.{layer}.mlp.experts.{expert}.{proj}."
        return public_from_mxfp4(self.tensor(base + "weight"), self.tensor(base + "weight_scale"))


# --------------------------------------------------------------------- convert

def convert(
    src, dst, *,
    k=K_DEFAULT, codebook: str = CODEBOOK_DEFAULT, window: int = WINDOW_DEFAULT,
    quantizer: str = "ldlq", imatrix: dict | None = None, imatrix_sha: str | None = None,
    layers: str | None = None,
    batch_experts: int = BATCH_EXPERTS_DEFAULT, resume: bool = False,
    scratch_gb: float = SCRATCH_GB_DEFAULT, g_scale: bool = True, quality: bool = True,
    verbose: bool = True,
) -> dict:
    src, dst = Path(src), Path(dst)
    window = validate_window(window, k)
    dst.mkdir(parents=True, exist_ok=True)
    reset_search_stats()
    config = json.loads((src / "config.json").read_text())
    index = json.loads((src / "model.safetensors.index.json").read_text())
    if quantizer == "ldlq" and imatrix is None:
        raise RuntimeError("--quantizer ldlq needs --imatrix")
    plan = plan_source(index)
    hidden = int(config["hidden_size"])
    inter = int(config["moe_intermediate_size"])
    experts = int(config["n_routed_experts"])
    all_moe = moe_layers(config)
    picked = parse_layer_range(layers, all_moe)
    partial = picked != all_moe

    t_stage = time.perf_counter()
    linked = stage_non_expert(src, dst, plan)
    t_stage = time.perf_counter() - t_stage

    reader = SourceReader(src, index["weight_map"])
    weight_map = {key: index["weight_map"][key] for key in plan["keep_keys"]}
    scratch_bytes = max(1, int(scratch_gb * (1 << 30)))
    import mlx.core as mx
    # The pool is no longer dropped between launches (1.27x), so cap it instead.
    mx.set_cache_limit(4 * scratch_bytes)
    chunk = launch_tiles_for(k, window, scratch_gb)
    cal_tag = "imatrix-diagonal" if imatrix is not None else (
        "ldlq-gaussian-256" if quantizer == "ldlq" else "none-direct")
    stamp = shard_stamp(k=k, codebook=codebook, window=window, quantizer=quantizer,
                        imatrix_sha=imatrix_sha, g_scale="gss" if g_scale else "one")
    stats = {"shards": 0, "skipped": 0, "prior_fallbacks": 0, "layers": picked,
             "partial": partial, "linked": linked, "stage_seconds": t_stage,
             "launch_tiles": chunk, "calibration": cal_tag, "seconds": 0.0,
             "dropped_shards": len(plan["drop"]), "mixed_shards": plan["mixed"],
             "rewritten": [], "weighted_rel_err": {}}
    if verbose:
        print(f"staged {linked} files in {t_stage:.1f}s, dropped {len(plan['drop'])} expert "
              f"shards, {len(plan['mixed'])} mixed shard(s) linked with their expert keys "
              f"left out of the index", flush=True)

    for layer in picked:
        rows_vec = None
        gu_flat = dn_flat = None
        if imatrix is not None:
            gu_key, dn_key, rows_key = imatrix_layer_keys(layer)
            if not imatrix_layer_complete(imatrix, layer):
                raise RuntimeError(f"imatrix has no complete entry for layer {layer}")
            gu_flat, dn_flat = imatrix[gu_key], imatrix[dn_key]
            rows_vec = np.asarray(imatrix[rows_key], dtype=np.float32).reshape(-1)
        for proj in PROJECTIONS:
            in_dim, out_dim = (inter, hidden) if proj == "down_proj" else (hidden, inter)
            shard = layer_proj_shard(layer, proj)
            base = switch_base(layer, proj)
            if resume:
                refusal = shard_reuse_refusal(dst / shard, experts, in_dim, out_dim, k, stamp)
                if refusal is None:
                    stats["skipped"] += 1
                    for key in exl3_keys(base):
                        weight_map[key] = shard
                    if verbose:
                        print(f"skip {shard}", flush=True)
                    continue
                if refusal != "absent":
                    stats["rewritten"].append(f"{shard}: {refusal}")
                    if verbose:
                        print(f"rewrite {shard}: {refusal}", flush=True)
            t0 = time.perf_counter()
            trellis = np.empty((experts, in_dim // 16, out_dim // 16, packed_hw(k)), dtype=np.uint16)
            suh = np.empty((experts, in_dim), dtype=np.float16)
            svh = np.empty((experts, out_dim), dtype=np.float16)
            flat = dn_flat if proj == "down_proj" else gu_flat
            fallbacks = 0
            scales: list[float] = []
            errs: list[float] = []
            before = search_stats_snapshot()

            def load_batch(start: int, stop: int):
                """Read, dequantize, regularize and LDL-factor one batch — every host
                stage there is. Runs on the prefetch thread while the GPU searches the
                previous batch, so only MLX work sits on the critical path."""
                publics, seeds, cals = [], [], []
                for ei in range(start, stop):
                    publics.append(reader.public(layer, ei, proj))
                    seeds.append(expert_seed(layer, ei, proj))
                    if imatrix is None:
                        cals.append(None)
                    else:
                        _, vec = calib_for_expert(
                            int(rows_vec[ei]) if rows_vec is not None else 1,
                            imatrix_expert_vector(flat, ei, in_dim),
                        )
                        cals.append(vec)
                return (prepare_expert_bank(publics, seeds, cals, quantizer=quantizer),
                        publics, cals)

            step = max(1, batch_experts)
            spans = [(s, min(experts, s + step)) for s in range(0, experts, step)]
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mimo-exl3-read") as pool:
                pending = pool.submit(load_batch, *spans[0])
                for idx, (start, stop) in enumerate(spans):
                    prep, publics, cals = pending.result()
                    pending = (pool.submit(load_batch, *spans[idx + 1])
                               if idx + 1 < len(spans) else None)
                    bt, bsuh, bsvh, bf = quantize_prepared_bank(
                        prep, publics, cals, k=k, codebook=codebook, window=window,
                        scratch_bytes=scratch_bytes, g_scale=g_scale, scale_out=scales,
                        err_out=errs if quality else None,
                    )
                    del prep
                    trellis[start:stop] = bt
                    suh[start:stop] = bsuh
                    svh[start:stop] = bsvh
                    fallbacks += bf
                    del publics, bt, bsuh, bsvh
            named = {
                base + ".trellis": ("U16", trellis.shape, trellis.tobytes()),
                base + ".suh": ("F16", suh.shape, suh.tobytes()),
                base + ".svh": ("F16", svh.shape, svh.tobytes()),
            }
            write_safetensors_raw(str(dst / shard), named, metadata=stamp)
            for key in named:
                weight_map[key] = shard
            dt = time.perf_counter() - t0
            after = search_stats_snapshot()
            gpu = after["seconds"] - before["seconds"]
            launches = after["launches"] - before["launches"]
            searched = after["tiles"] - before["tiles"]
            stats["shards"] += 1
            stats["prior_fallbacks"] += fallbacks
            if errs:
                stats["weighted_rel_err"][f"L{layer}.{proj}"] = list(errs)
            stats["seconds"] += dt
            if verbose:
                print(f"wrote {shard}  {dt:.1f}s  e={experts} in={in_dim} out={out_dim} "
                      f"prior_fallback={fallbacks}  g[{min(scales, default=1.0):.2f},"
                      f"{max(scales, default=1.0):.2f}] mean {sum(scales) / max(len(scales), 1):.3f}"
                      f"  werr {sum(errs) / max(len(errs), 1):.5f}"
                      f"  search {gpu:.1f}s "
                      f"({100.0 * gpu / max(dt, 1e-9):.1f}% of wall) "
                      f"launches={launches} tiles={searched} "
                      f"({searched / max(launches, 1):.0f}/launch, "
                      f"{searched / max(gpu, 1e-9):.0f} tiles/s)", flush=True)
            del trellis, suh, svh, named

    total = 0
    for fname in sorted(set(weight_map.values())):
        path = dst / fname
        if path.is_file():
            total += os.path.getsize(path)
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    stats["stamp"] = stamp
    block = expert_quant_block(k=k, codebook=codebook, window=window,
                               quantizer=stamp["quantizer"], calibration=cal_tag,
                               imatrix_sha256=stamp["imatrix_sha256"],
                               converter=stamp["converter"])
    (dst / "config.json").write_text(json.dumps(rewrite_config(config, block), indent=2))
    stats["total_size"] = total
    stats["search"] = search_stats_snapshot()
    if verbose and partial:
        print(f"PARTIAL pack: only layers {picked} carry EXL3 experts", flush=True)
    return stats


# ----------------------------------------------------------------- self-tests

def _mxfp4_bytes(values: np.ndarray, scale_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack nibble codes [out, in] and e8m0 codes [out, in/32] into source layout."""
    v = np.asarray(values, dtype=np.uint8)
    packed = (v[:, 0::2] & 0x0F) | ((v[:, 1::2] & 0x0F) << 4)
    return packed.astype(np.uint8), np.asarray(scale_codes, dtype=np.uint8)


class Mxfp4LayoutTests(unittest.TestCase):
    def test_even_column_is_the_low_nibble_and_odd_the_high(self):
        codes = np.zeros((1, 64), dtype=np.uint8)
        codes[0, 0] = 0x2   # 1.0
        codes[0, 1] = 0xD   # -3.0
        w, s = _mxfp4_bytes(codes, np.full((1, 2), 127, dtype=np.uint8))
        self.assertEqual(int(w[0, 0]), 0x2 | (0xD << 4))
        deq = dequant_mxfp4(w, s)
        self.assertEqual(deq.shape, (1, 64))
        self.assertEqual(float(deq[0, 0]), 1.0)
        self.assertEqual(float(deq[0, 1]), -3.0)

    def test_every_e2m1_code_decodes_to_the_table(self):
        want = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        got = e2m1_value(np.arange(16, dtype=np.uint8))
        for code, expected in enumerate(want):
            self.assertEqual(float(got[code]), expected, msg=f"code {code}")
        self.assertTrue(np.signbit(got[8]))

    def test_e8m0_zero_infinity_and_powers_of_two(self):
        got = e8m0_value(np.array([0, 125, 126, 127, 128, 129, 254, 255], dtype=np.uint8))
        self.assertEqual(float(got[0]), 0.0)
        self.assertEqual(float(got[1]), 0.25)
        self.assertEqual(float(got[3]), 1.0)
        self.assertEqual(float(got[4]), 2.0)
        self.assertTrue(np.isfinite(got[6]))
        self.assertTrue(np.isinf(got[7]))

    def test_one_scale_covers_thirty_two_input_columns_of_its_own_row(self):
        codes = np.full((2, 64), 0x2, dtype=np.uint8)  # every value 1.0
        scales = np.array([[128, 127], [127, 126]], dtype=np.uint8)
        deq = dequant_mxfp4(*_mxfp4_bytes(codes, scales))
        self.assertTrue(np.all(deq[0, :32] == 2.0))
        self.assertTrue(np.all(deq[0, 32:] == 1.0))
        self.assertTrue(np.all(deq[1, :32] == 1.0))
        self.assertTrue(np.all(deq[1, 32:] == 0.5))

    def test_public_weight_is_the_transpose_in_by_out(self):
        rng = np.random.default_rng(0)
        codes = rng.integers(0, 16, size=(8, 64), dtype=np.uint8)
        w, s = _mxfp4_bytes(codes, np.full((8, 2), 127, dtype=np.uint8))
        deq = dequant_mxfp4(w, s)
        pub = public_from_mxfp4(w, s)
        self.assertEqual(pub.shape, (64, 8))
        self.assertTrue(np.array_equal(pub, deq.T))

    def test_an_infinite_block_scale_is_refused_not_packed(self):
        """e8m0 0xff is +inf and the 128-point Hadamard would spread it over a whole
        block. Real MiMo codes sit in the 110s, so this is a guard, not a path."""
        codes = np.full((1, 64), 0x2, dtype=np.uint8)
        w, s = _mxfp4_bytes(codes, np.array([[255, 127]], dtype=np.uint8))
        with self.assertRaises(RuntimeError):
            dequant_mxfp4(w, s)

    def test_a_scale_that_does_not_cover_the_weight_is_refused(self):
        w = np.zeros((2, 32), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            dequant_mxfp4(w, np.zeros((2, 1), dtype=np.uint8))


class RateTests(unittest.TestCase):
    def test_a_k_that_is_not_a_sixteenth_is_refused(self):
        for bad in (2.3, 2.51, 3.999):
            with self.assertRaises(ValueError):
                parse_k(bad)

    def test_a_k_outside_two_to_eight_is_refused(self):
        for bad in (1.5, 8.5):
            with self.assertRaises(ValueError):
                parse_k(bad)

    def test_a_sixteenth_multiple_is_admitted_and_integers_stay_int(self):
        self.assertEqual(parse_k(2.5), 2.5)
        self.assertEqual(parse_k("2.5625"), 2.5625)
        self.assertIsInstance(parse_k(4), int)
        self.assertEqual(parse_k(4), 4)

    def test_packed_halfwords_are_sixteen_k(self):
        self.assertEqual(packed_hw(2.5), 40)
        self.assertEqual(packed_hw(3), 48)
        self.assertEqual(k_from_packed(40), 2.5)


class RoundTripTests(unittest.TestCase):
    """The packed bytes must decode back to what the search chose."""

    def _case(self, k, codebook="tiny", window=16, rows=128, cols=128):
        _ensure_lib()
        from ponyexl3.convert.regularize import regularize_public_weight
        from ponyexl3.ref.reconstruct import reconstruct_inner
        from ponyexl3.ref.trellis import fresh_from_states, unpack_trellis
        rng = np.random.default_rng(3)
        public = rng.standard_normal((rows, cols), dtype=np.float32)
        reg = regularize_public_weight(public, seed=1)
        cb = codebook_mode(codebook)
        to_tiles, from_tiles = _tile_helpers()
        tiles = to_tiles(reg.inner)
        states, decoded = search_tiles(tiles, k=k, cb=cb, window=window, chunk=4096,
                                       want_decoded=True)
        packed = pack_states(states, k, rows // 16, cols // 16)
        return reg, states, decoded, packed, unpack_trellis, fresh_from_states, \
            reconstruct_inner, from_tiles

    def test_pack_unpack_preserves_every_fresh_bit(self):
        k = 2.5
        reg, states, _dec, packed, unpack, fresh, _rec, _fi = self._case(k)
        self.assertEqual(packed.shape[-1], packed_hw(k))
        back = unpack(packed, k)
        self.assertTrue(np.array_equal(
            fresh(back, k), fresh(states.reshape(*back.shape[:2], 256), k)))

    def test_decoded_pack_matches_the_searchs_own_reconstruction(self):
        k = 2.5
        reg, states, decoded, packed, _u, _f, reconstruct_inner, from_tiles = self._case(k)
        want = from_tiles(decoded, 128, 128)
        got = reconstruct_inner(packed, k, tiny=True).astype(np.float32)
        self.assertTrue(np.allclose(got, want, rtol=0, atol=1e-3),
                        msg=f"max |diff| {float(np.abs(got - want).max())}")

    def test_the_window_must_sit_in_the_searchs_own_range(self):
        for good in (12, 16, 18):
            self.assertEqual(validate_window(good, 2.5), good)
        for bad in (3, 25, 2):
            with self.assertRaises(RuntimeError):
                validate_window(bad, 2.5)

    def test_the_searched_window_is_exactly_what_the_decoders_mask(self):
        """The window is a DECODE parameter (PonyExl3 437d49c): at w12 the searched
        states, the packed bits read back as 16-bit sliding windows and masked, and the
        masked reference decode must all agree with the search's own values. Checking
        the packed BITS alone passes even when the decoders read something else, which
        is exactly what hid this before the window became a decode parameter."""
        _ensure_lib()
        from ponyexl3.ref.codebook import window_mask
        for window in (12, 16):
            reg, states, decoded, packed, unpack, _fresh, reconstruct_inner, from_tiles = \
                self._case(2.5, window=window)
            mask = np.uint16(window_mask(window))
            back = unpack(packed, 2.5)
            self.assertTrue(np.array_equal(back & mask,
                                           states.reshape(*back.shape[:2], 256) & mask),
                            msg=f"window {window}: packed bits are not the searched states")
            got = np.asarray(reconstruct_inner(packed, 2.5, tiny=True, window=window),
                             dtype=np.float32)
            want = from_tiles(decoded, 128, 128)
            self.assertTrue(np.allclose(got, want, rtol=0, atol=1e-3),
                            msg=f"window {window}: max |diff| "
                                f"{float(np.abs(got - want).max())}")

    def test_a_pack_read_at_the_wrong_window_is_noise(self):
        """Red-on-revert for the stamp carrying `window`: decoding a w12 pack as w16."""
        _ensure_lib()
        from ponyexl3.ref.reconstruct import reconstruct_inner
        reg, _states, decoded, packed, _u, _f, _r, from_tiles = self._case(2.5, window=12)
        want = from_tiles(decoded, 128, 128)
        right = np.asarray(reconstruct_inner(packed, 2.5, tiny=True, window=12), np.float32)
        wrong = np.asarray(reconstruct_inner(packed, 2.5, tiny=True, window=16), np.float32)
        self.assertLess(float(np.mean((right - want) ** 2)), 1e-5)
        self.assertGreater(float(np.mean((wrong - want) ** 2)), 0.1)


class QualityMetricTests(unittest.TestCase):
    def test_the_metric_is_dsv4s_definition_with_the_axis_this_layout_uses(self):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dsv4_imatrix import weighted_rel_err as reference
        rng = np.random.default_rng(61)
        hf = rng.standard_normal((8, 32), dtype=np.float32)       # [out, in]
        hat = hf + 0.1 * rng.standard_normal((8, 32), dtype=np.float32)
        om = np.abs(rng.standard_normal(32, dtype=np.float32)) + 0.1
        self.assertAlmostEqual(weighted_rel_err(hf.T, hat.T, om), reference(hf, hat, om),
                               places=6)

    def test_the_reconstruction_matches_ponyexl3s_public_decode(self):
        """The inner the search already produced, through the outer transforms, must be
        what `reconstruct_public_weights` gets by decoding the packed trellis."""
        _ensure_lib()
        import mlx.core as mx
        from ponyexl3.convert.regularize import regularize_public_weight
        from ponyexl3.ref.reconstruct import reconstruct_public_weights
        rng = np.random.default_rng(62)
        reg = regularize_public_weight(rng.standard_normal((HAD_BLOCK, HAD_BLOCK),
                                                           dtype=np.float32), seed=2)
        cb = codebook_mode("tiny")
        packed, recon = ldlq_group_mlx(
            mx.array(reg.inner[None], dtype=mx.float32),
            mx.zeros((1, 1, HAD_BLOCK, HAD_BLOCK), dtype=mx.float32),
            k=2.5, cb=cb, window=16, scratch_bytes=1 << 29, want_recon=True)
        suh16, svh16 = reg.suh.astype(np.float16), reg.svh.astype(np.float16)
        got = np.array(public_from_inner_mlx(recon[0], mx.array(suh16.astype(np.float32)),
                                             mx.array(svh16.astype(np.float32))))
        want = np.asarray(reconstruct_public_weights(packed[0], suh16, svh16, 2.5, tiny=True),
                          dtype=np.float32)
        rel = float(np.sqrt(np.sum((got - want) ** 2) / np.sum(want ** 2)))
        self.assertLess(rel, 2e-3, msg=f"relative disagreement {rel}")

    def test_the_converter_reports_one_error_per_expert(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "src"
            write_synthetic_source(root)
            stats = convert(root, Path(td) / "out", k=2.5, codebook="tiny", window=16,
                            quantizer="ldlq", imatrix=ConvertTests._imatrix([8.0] * SYN_EXPERTS),
                            batch_experts=2, layers="1", scratch_gb=0.5, verbose=False)
            werr = stats["weighted_rel_err"]
            self.assertEqual(sorted(werr), [f"L1.{p}" for p in sorted(PROJECTIONS)])
            for key, values in werr.items():
                self.assertEqual(len(values), SYN_EXPERTS, msg=key)
                self.assertTrue(all(0.0 < v < 2.0 for v in values), msg=f"{key}: {values}")


class GlobalScaleTests(unittest.TestCase):
    """The converter regularizes against MCG's codebook RMS; TINY's is 1.63, so without
    this search the inner matrix sits off the codebook's own scale."""

    def _regularized(self, rows=HAD_BLOCK, cols=4 * HAD_BLOCK, seed=41):
        _ensure_lib()
        from ponyexl3.convert.regularize import regularize_public_weight
        rng = np.random.default_rng(seed)
        return regularize_public_weight(rng.standard_normal((rows, cols), dtype=np.float32),
                                        seed=1)

    def test_the_search_is_bounded_to_ten_evaluations(self):
        """PonyExl3's default (13 evaluations x 768 tiles) is 30% of a MiMo expert."""
        for low, high in G_SCALE_BRACKET.values():
            self.assertLessEqual(g_scale_iterations(low, high) + 2, G_SCALE_MAX_EVALS)
        self.assertLessEqual(G_SCALE_TILES * G_SCALE_MAX_EVALS, 2048)
        self.assertLess(g_scale_iterations(0.7, 1.9, 0.9), g_scale_iterations(0.7, 1.9))

    def test_every_codebook_brackets_its_own_expected_scale(self):
        self.assertEqual(set(G_SCALE_BRACKET), {"tiny", "mul1", "mcg"})
        lo, hi = G_SCALE_BRACKET["tiny"]
        self.assertLess(lo, 1.31)
        self.assertGreater(hi, 1.31)

    def test_the_sample_is_ponyexl3s_wrapped_tile_diagonal(self):
        _ensure_lib()
        import mlx.core as mx
        from ponyexl3.convert.regularize import sample_tile_matrix
        reg = self._regularized()
        rows, cols = reg.inner.shape
        full = sample_tile_matrix(reg.inner, width=G_SCALE_WIDTH)
        index = sample_tile_index(rows, cols, count=1 << 30)
        got = np.array(sample_tiles_mlx(mx.array(reg.inner[None], dtype=mx.float32), index))
        self.assertEqual(got.shape[1], full.shape[0] // 16)
        for t in range(got.shape[1]):
            self.assertTrue(np.array_equal(got[0, t].reshape(16, 16), full[t * 16:(t + 1) * 16]))

    def test_the_thinned_sample_keeps_the_diagonals_own_tiles(self):
        reg = self._regularized()
        rows, cols = reg.inner.shape
        thin = sample_tile_index(rows, cols, count=8)
        full = sample_tile_index(rows, cols, count=1 << 30)
        self.assertEqual(thin.shape[0], 8)
        self.assertTrue(set(map(tuple, thin)).issubset(set(map(tuple, full))))

    def test_the_search_agrees_with_the_scalar_golden_section(self):
        _ensure_lib()
        import mlx.core as mx
        from ponyexl3.convert.regularize import g_scale_gss
        reg = self._regularized()
        rows, cols = reg.inner.shape
        index = sample_tile_index(rows, cols, count=32)
        tiles = sample_tiles_mlx(mx.array(reg.inner[None], dtype=mx.float32), index)
        cb = codebook_mode("tiny")
        got = float(np.array(g_scale_search_mlx(tiles, k=2.5, cb=cb, codebook="tiny",
                                                window=16, scratch_bytes=1 << 29))[0])
        sample = np.array(tiles)[0]

        def score(scale: float) -> float:
            kernel = np.ascontiguousarray(sample[:, _tile_perm()])
            _states, decoded = search_tiles(
                np.ascontiguousarray(kernel * np.float32(scale)),
                k=2.5, cb=cb, window=16, chunk=4096, want_decoded=True)
            return float(np.mean((decoded / np.float32(scale) - kernel) ** 2,
                                 dtype=np.float64))
        low, high = G_SCALE_BRACKET["tiny"]
        want = g_scale_gss(score, low=low, high=high, tol=G_SCALE_TOL).scale
        self.assertAlmostEqual(got, want, delta=2 * G_SCALE_TOL,
                               msg=f"mlx {got} vs scalar {want}")

    def test_the_found_scale_beats_one_on_the_sampled_tiles(self):
        _ensure_lib()
        import mlx.core as mx
        reg = self._regularized()
        rows, cols = reg.inner.shape
        tiles = sample_tiles_mlx(mx.array(reg.inner[None], dtype=mx.float32),
                                 sample_tile_index(rows, cols, count=64))
        cb = codebook_mode("tiny")
        g = float(np.array(g_scale_search_mlx(tiles, k=2.5, cb=cb, codebook="tiny",
                                              window=16, scratch_bytes=1 << 29))[0])
        sample = np.array(tiles)[0][:, _tile_perm()]

        def mse(scale):
            _states, decoded = search_tiles(np.ascontiguousarray(sample * np.float32(scale)),
                                            k=2.5, cb=cb, window=16, chunk=4096,
                                            want_decoded=True)
            return float(np.mean((decoded / np.float32(scale) - sample) ** 2, dtype=np.float64))
        self.assertGreater(g, 1.0, "TINY's RMS is above MCG's, so the scale must rise")
        self.assertLess(mse(g), mse(1.0))

    def test_the_scaled_pair_reconstructs_the_same_public_weight(self):
        """`apply_global_scale`'s contract: nothing new is stored, the pair cancels."""
        _ensure_lib()
        from ponyexl3.ref.reconstruct import reconstruct_public_weights
        reg = self._regularized(rows=HAD_BLOCK, cols=HAD_BLOCK)
        cb = codebook_mode("tiny")
        g = np.float32(1.27)
        plain = quantize_inner_direct(reg.inner, k=2.5, cb=cb, window=16, chunk=4096)[0]
        scaled = quantize_inner_direct(reg.inner * g, k=2.5, cb=cb, window=16, chunk=4096)[0]
        a = np.asarray(reconstruct_public_weights(scaled, (reg.suh / g).astype(np.float16),
                                                  reg.svh.astype(np.float16), 2.5, tiny=True),
                       dtype=np.float32)
        b = np.asarray(reconstruct_public_weights(plain, reg.suh.astype(np.float16),
                                                  reg.svh.astype(np.float16), 2.5, tiny=True),
                       dtype=np.float32)
        src = np.asarray(reconstruct_public_weights(scaled, reg.suh.astype(np.float16),
                                                    reg.svh.astype(np.float16), 2.5, tiny=True),
                         dtype=np.float32)
        self.assertFalse(np.allclose(a, src), "dividing suh by g must matter")
        rel = float(np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2)))
        self.assertLess(rel, 0.6, "both must reconstruct the same public weight, up to rounding")


MIMO_SOURCE = os.environ.get("MIMO_V2_SOURCE", "/Users/beam/llm/models/MiMo-V2.6-Flash-RL")


class RealExpertWindowTests(unittest.TestCase):
    """The w12 decode bar on a REAL MiMo expert, not a synthetic matrix. Skips when the
    checkpoint is absent; `MIMO_V2_SOURCE` points it elsewhere."""

    def test_a_real_experts_w12_pack_decodes_to_what_the_search_chose(self):
        src = Path(MIMO_SOURCE)
        if not (src / "model.safetensors.index.json").is_file():
            self.skipTest(f"no MiMo checkpoint at {src}")
        _ensure_lib()
        import mlx.core as mx
        from ponyexl3.convert.regularize import regularize_public_weight
        from ponyexl3.ref.codebook import window_mask
        from ponyexl3.ref.reconstruct import reconstruct_inner
        from ponyexl3.ref.trellis import unpack_trellis
        index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
        public = SourceReader(src, index).public(1, 0, "gate_proj")[:256, :256]
        reg = regularize_public_weight(public, seed=expert_seed(1, 0, "gate_proj"))
        cb, window = codebook_mode("tiny"), 12
        to_tiles, from_tiles = _tile_helpers()
        tiles = to_tiles(reg.inner)
        states, decoded = search_tiles(tiles, k=2.5, cb=cb, window=window, chunk=1024,
                                       want_decoded=True)
        packed = pack_states(states, 2.5, 16, 16)
        mask = np.uint16(window_mask(window))
        back = unpack_trellis(packed, 2.5)
        self.assertTrue(np.array_equal(back & mask, states.reshape(16, 16, 256) & mask))
        got = np.asarray(reconstruct_inner(packed, 2.5, tiny=True, window=window), np.float32)
        self.assertTrue(np.allclose(got, from_tiles(decoded, 256, 256), rtol=0, atol=1e-3))
        del mx


def _tile_perm() -> np.ndarray:
    _ensure_lib()
    from ponyexl3.convert.direct import _TENSOR_CORE_PERM
    return _TENSOR_CORE_PERM


class LdlTests(unittest.TestCase):
    def test_a_diagonal_hessian_gives_a_feedbackless_factor(self):
        rng = np.random.default_rng(5)
        v = np.abs(rng.standard_normal(64).astype(np.float32)) + 0.05
        self.assertTrue(ldl_is_feedbackless(ldl_factor(diag_hessian(v))))

    def test_a_captured_hessian_keeps_its_feedback(self):
        _ensure_lib()
        from ponyexl3.convert.hessian import capture_hessian
        rng = np.random.default_rng(6)
        h = capture_hessian(rng.standard_normal((256, 64), dtype=np.float32))
        self.assertFalse(ldl_is_feedbackless(ldl_factor(h)))

    def test_the_rotated_hessian_is_what_ponyexl3_builds_from_the_same_statistic(self):
        """Blockwise `H128 diag(suh^2 h) H128` must equal ponyexl3's own activation
        rotation applied to any activation matrix with that public covariance."""
        _ensure_lib()
        from ponyexl3.convert.hessian import capture_hessian, public_activations_to_inner
        rng = np.random.default_rng(31)
        rows = 2 * HAD_BLOCK
        h = np.abs(rng.standard_normal(rows, dtype=np.float32)) + 0.05
        suh = (rng.standard_normal(rows, dtype=np.float32)).astype(np.float32)
        acts = np.diag(np.sqrt(h)).astype(np.float32)
        want = capture_hessian(public_activations_to_inner(acts, suh), normalize=False)
        got = inner_hessian_blocks(h, suh)
        self.assertEqual(got.shape, (2, HAD_BLOCK, HAD_BLOCK))
        for b in range(2):
            lo = b * HAD_BLOCK
            self.assertTrue(np.allclose(got[b], want[lo:lo + HAD_BLOCK, lo:lo + HAD_BLOCK],
                                        rtol=1e-4, atol=1e-4))
        off = want.copy()
        for b in range(2):
            lo = b * HAD_BLOCK
            off[lo:lo + HAD_BLOCK, lo:lo + HAD_BLOCK] = 0.0
        self.assertLess(float(np.abs(off).max()), 1e-3, "the rotation is block diagonal")

    def test_the_rotated_hessian_is_dense_so_the_factor_carries_feedback(self):
        """The whole point: a public diagonal is NOT diagonal in the inner basis, so
        LDLQ gets real feedback instead of an identity factor."""
        rng = np.random.default_rng(32)
        rows = 2 * HAD_BLOCK
        h = np.abs(rng.standard_normal(rows, dtype=np.float32)) + 0.05
        suh = rng.standard_normal(rows, dtype=np.float32)
        blocks = inner_hessian_blocks(h, suh)
        offdiag = blocks - np.stack([np.diag(np.diag(b)) for b in blocks])
        self.assertGreater(float(np.abs(offdiag).max()), 1e-3)
        l = inner_ldl_blocks(blocks)
        self.assertFalse(ldl_is_feedbackless(l))
        self.assertTrue(np.all(l[:, np.arange(HAD_BLOCK), np.arange(HAD_BLOCK)] == 0))
        self.assertTrue(np.all(np.triu(l, 1) == 0), "L must be lower triangular")

    def test_the_uncalibrated_diagonal_would_have_had_no_feedback_at_all(self):
        """Red-on-revert for the basis bug: the UNROTATED public diagonal factors to
        exactly the identity, which is what silently disabled the calibration."""
        rng = np.random.default_rng(33)
        h = np.abs(rng.standard_normal(HAD_BLOCK, dtype=np.float32)) + 0.05
        self.assertTrue(ldl_is_feedbackless(ldl_factor(diag_hessian(h))))

    def test_the_block_factorization_agrees_with_ponyexl3s_dense_one(self):
        _ensure_lib()
        from ponyexl3.convert.hessian import block_ldl, prepare_hessian_for_ldl
        rng = np.random.default_rng(34)
        rows = 2 * HAD_BLOCK
        h = np.abs(rng.standard_normal(rows, dtype=np.float32)) + 0.05
        suh = rng.standard_normal(rows, dtype=np.float32)
        blocks = inner_hessian_blocks(h, suh)
        dense = np.zeros((rows, rows), dtype=np.float32)
        for b in range(2):
            lo = b * HAD_BLOCK
            dense[lo:lo + HAD_BLOCK, lo:lo + HAD_BLOCK] = blocks[b]
        want = block_ldl(prepare_hessian_for_ldl(dense).hessian, block_size=16).l.copy()
        want[np.diag_indices_from(want)] = 0.0
        got = inner_ldl_blocks(blocks)
        for b in range(2):
            lo = b * HAD_BLOCK
            self.assertTrue(np.allclose(got[b], want[lo:lo + HAD_BLOCK, lo:lo + HAD_BLOCK],
                                        rtol=2e-3, atol=2e-3),
                            msg=f"block {b} max |diff| "
                                f"{float(np.abs(got[b] - want[lo:lo+HAD_BLOCK, lo:lo+HAD_BLOCK]).max())}")

    def _ldlq_case(self, n=3, rows=HAD_BLOCK, cols=HAD_BLOCK, seed=35):
        _ensure_lib()
        from ponyexl3.convert.regularize import regularize_public_weight
        rng = np.random.default_rng(seed)
        regs, blocks, dense, diags = [], [], [], []
        for i in range(n):
            reg = regularize_public_weight(rng.standard_normal((rows, cols), dtype=np.float32),
                                           seed=i)
            h = np.abs(rng.standard_normal(rows, dtype=np.float32)) + 0.05
            l = inner_ldl_blocks(inner_hessian_blocks(h, reg.suh))
            regs.append(reg)
            blocks.append(l)
            diags.append(h)
            full = np.zeros((rows, rows), dtype=np.float32)
            for b in range(rows // HAD_BLOCK):
                lo = b * HAD_BLOCK
                full[lo:lo + HAD_BLOCK, lo:lo + HAD_BLOCK] = l[b]
            dense.append(full)
        return regs, blocks, dense, diags

    def test_grouped_mlx_ldlq_with_no_feedback_is_exactly_the_direct_search(self):
        """With a zero factor the compensation is exactly 0.0, so the MLX loop must
        reproduce the direct search halfword for halfword — the structural check that
        its tiling, packing and expert order are right."""
        import mlx.core as mx
        regs, blocks, _dense, _d = self._ldlq_case()
        zero = np.zeros_like(np.stack(blocks))
        got = ldlq_group_mlx(mx.array(np.stack([r.inner for r in regs]), dtype=mx.float32),
                             mx.array(zero, dtype=mx.float32),
                             k=2.5, cb=codebook_mode("tiny"), window=16, scratch_bytes=1 << 29)
        for i, reg in enumerate(regs):
            want = quantize_inner_direct(reg.inner, k=2.5, cb=codebook_mode("tiny"),
                                         window=16, chunk=4096)[0]
            self.assertTrue(np.array_equal(got[i], want), msg=f"expert {i} diverged")

    def test_grouped_mlx_ldlq_reconstructs_as_well_as_the_reference_loop(self):
        """An MLX-resident LDLQ is NOT byte-identical to the numpy loop — the
        compensation GEMM reduces in a different order and the greedy search amplifies
        it — so the bar is the reconstruction error, not the bytes."""
        import mlx.core as mx
        from ponyexl3.ref.reconstruct import reconstruct_inner
        regs, blocks, dense, _d = self._ldlq_case()
        cb = codebook_mode("tiny")
        got = ldlq_group_mlx(mx.array(np.stack([r.inner for r in regs]), dtype=mx.float32),
                             mx.array(np.stack(blocks), dtype=mx.float32),
                             k=2.5, cb=cb, window=16, scratch_bytes=1 << 29)
        want = ldlq_group([r.inner for r in regs], dense, k=2.5, cb=cb, window=16, chunk=4096)
        for i, reg in enumerate(regs):
            def rel(packed):
                w = np.asarray(reconstruct_inner(packed, 2.5, tiny=True), dtype=np.float32)
                return float(np.sqrt(np.sum((w - reg.inner) ** 2) / np.sum(reg.inner ** 2)))
            self.assertLess(rel(got[i]), rel(want[i]) * 1.02, msg=f"expert {i} regressed")

    def test_rotated_feedback_beats_the_direct_search_on_the_calibrated_objective(self):
        """The reason the rotation matters: LDLQ under the rotated Hessian must lower
        the imatrix-weighted output error against the same experts quantized directly."""
        import mlx.core as mx
        from ponyexl3.ref.reconstruct import reconstruct_inner
        regs, blocks, _dense, diags = self._ldlq_case(n=4, rows=HAD_BLOCK, cols=4 * HAD_BLOCK)
        cb = codebook_mode("tiny")
        fed = ldlq_group_mlx(mx.array(np.stack([r.inner for r in regs]), dtype=mx.float32),
                             mx.array(np.stack(blocks), dtype=mx.float32),
                             k=2.5, cb=cb, window=16, scratch_bytes=1 << 29)
        better = 0
        for i, reg in enumerate(regs):
            plain = quantize_inner_direct(reg.inner, k=2.5, cb=cb, window=16, chunk=4096)[0]
            hb = inner_hessian_blocks(diags[i], reg.suh)[0]

            def weighted(packed):
                d = np.asarray(reconstruct_inner(packed, 2.5, tiny=True),
                               dtype=np.float32) - reg.inner
                return float(np.sum(d * (hb @ d)))
            if weighted(fed[i]) < weighted(plain):
                better += 1
        self.assertGreaterEqual(better, 3, "feedback did not help on 3 of 4 experts")

    def test_the_ldl_factor_does_not_depend_on_the_global_scale(self):
        """What lets the prefetch thread build the factor before the scale is known:
        suh -> suh/g takes the Hessian to H/g**2, and a unit-diagonal LDL factor is
        invariant under a positive scalar."""
        rng = np.random.default_rng(52)
        suh = rng.standard_normal(HAD_BLOCK, dtype=np.float32)
        h = np.abs(rng.standard_normal(HAD_BLOCK, dtype=np.float32)) + 0.05
        plain = imatrix_ldl_blocks(h, suh)
        for g in (0.8, 1.21, 1.9):
            scaled = imatrix_ldl_blocks(h, suh / np.float32(g))
            rel = float(np.abs(plain - scaled).max() / max(float(np.abs(plain).max()), 1e-9))
            self.assertLess(rel, 1e-4, msg=f"g={g} moved the factor by {rel}")

    def test_the_prepared_half_carries_every_host_stage(self):
        """Nothing the prefetch thread can do may stay on the critical path."""
        rng = np.random.default_rng(53)
        pubs = [rng.standard_normal((HAD_BLOCK, HAD_BLOCK), dtype=np.float32) for _ in range(2)]
        cals = [np.abs(rng.standard_normal(HAD_BLOCK, dtype=np.float32)) + 0.05, None]
        prep = prepare_expert_bank(pubs, [1, 2], cals, quantizer="ldlq")
        self.assertEqual(prep.inner.shape, (2, HAD_BLOCK, HAD_BLOCK))
        self.assertEqual(prep.factor.shape, (2, 1, HAD_BLOCK, HAD_BLOCK))
        self.assertEqual(prep.fallbacks, 1)
        self.assertIsNone(prepare_expert_bank(pubs, [1, 2], cals, quantizer="direct").factor)

    def test_the_prior_is_a_flat_public_hessian_through_the_same_rotation(self):
        """`|suh|` varies inside a 128-block, so a flat PUBLIC Hessian is NOT isotropic
        in the inner basis: the prior must be rotated like the calibrated one."""
        rng = np.random.default_rng(51)
        suh = rng.standard_normal(HAD_BLOCK, dtype=np.float32)
        got = prior_ldl_blocks(suh)
        want = inner_ldl_blocks(inner_hessian_blocks(np.ones(HAD_BLOCK, np.float32), suh))
        self.assertTrue(np.array_equal(got, want))
        self.assertGreater(float(np.abs(got).max()), 0.05,
                           "a varying suh makes the prior dense")
        flat = np.full(HAD_BLOCK, 3.0, dtype=np.float32)
        constant = inner_ldl_blocks(inner_hessian_blocks(np.ones(HAD_BLOCK, np.float32), flat))
        self.assertLess(float(np.abs(constant).max()), 1e-4,
                        "a CONSTANT suh really would rotate to a diagonal, which is the "
                        "only case the old comment described")

    def test_zero_routed_tokens_fall_back_to_the_gaussian_prior(self):
        moments = np.arange(4, dtype=np.float32)
        self.assertEqual(calib_for_expert(0, moments), ("ldlq-gaussian-256", None))
        mode, vec = calib_for_expert(7, moments)
        self.assertEqual(mode, "imatrix-diagonal")
        self.assertTrue(np.array_equal(vec, moments))

    def test_the_imatrix_keys_are_mimos_own_expert_names(self):
        self.assertEqual(
            imatrix_layer_keys(3),
            ("model.layers.3.mlp.experts.gate_up_proj",
             "model.layers.3.mlp.experts.down_proj",
             "model.layers.3.mlp.experts.gate_up_proj.rows"))
        store = {k: np.zeros(1, dtype=np.float32) for k in imatrix_layer_keys(3)}
        self.assertTrue(imatrix_layer_complete(store, 3))
        self.assertFalse(imatrix_layer_complete(store, 4))


# --- a synthetic MiMo-shaped source -----------------------------------------

SYN_HIDDEN = 128
SYN_INTER = 128
SYN_EXPERTS = 2
SYN_LAYERS = 3


def _syn_config() -> dict:
    return {
        "model_type": "mimo_v2", "hidden_size": SYN_HIDDEN, "moe_intermediate_size": SYN_INTER,
        "n_routed_experts": SYN_EXPERTS, "num_experts_per_tok": 2,
        "num_hidden_layers": SYN_LAYERS, "moe_layer_freq": [0] + [1] * (SYN_LAYERS - 1),
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3",
                                "store_dtype": "mxfp4", "mxfp4_block_size": 32},
    }


def write_synthetic_source(root: Path) -> dict:
    """A two-layer MiMo shape: one trunk shard with the dense layer, one pure-expert
    shard per expert, and a tokenizer file to hard-link."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    weight_map: dict[str, str] = {}
    trunk = {
        "model.layers.0.mlp.gate_proj.weight": ("BF16", (SYN_INTER, SYN_HIDDEN),
                                                rng.integers(0, 1 << 16, SYN_INTER * SYN_HIDDEN,
                                                             dtype=np.uint16).tobytes()),
        "model.layers.1.mlp.gate.weight": ("F32", (SYN_EXPERTS, SYN_HIDDEN),
                                           rng.standard_normal((SYN_EXPERTS, SYN_HIDDEN),
                                                               dtype=np.float32).tobytes()),
    }
    write_safetensors_raw(str(root / "model-trunk.safetensors"), trunk)
    for key in trunk:
        weight_map[key] = "model-trunk.safetensors"
    for ei in range(SYN_EXPERTS):
        named: dict[str, tuple] = {}
        for layer in range(1, SYN_LAYERS):
            for proj in PROJECTIONS:
                in_dim, out_dim = (SYN_INTER, SYN_HIDDEN) if proj == "down_proj" else (SYN_HIDDEN, SYN_INTER)
                w = rng.integers(0, 256, size=(out_dim, in_dim // 2), dtype=np.uint8)
                s = rng.integers(120, 132, size=(out_dim, in_dim // 32), dtype=np.uint8)
                base = f"model.layers.{layer}.mlp.experts.{ei}.{proj}."
                named[base + "weight"] = ("U8", w.shape, w.tobytes())
                named[base + "weight_scale"] = ("U8", s.shape, s.tobytes())
        fname = f"model-ep{ei}.safetensors"
        write_safetensors_raw(str(root / fname), named)
        for key in named:
            weight_map[key] = fname
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 0}, "weight_map": weight_map}, indent=2))
    (root / "config.json").write_text(json.dumps(_syn_config(), indent=2))
    (root / "tokenizer.json").write_text("{}")
    (root / "chat_template.jinja").write_text("{{ messages }}")
    return weight_map


class PlanTests(unittest.TestCase):
    def test_pure_expert_shards_drop_and_everything_else_links(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "src"
            write_synthetic_source(root)
            index = json.loads((root / "model.safetensors.index.json").read_text())
            plan = plan_source(index)
            self.assertEqual(plan["drop"], ["model-ep0.safetensors", "model-ep1.safetensors"])
            self.assertEqual(plan["hardlink"], ["model-trunk.safetensors"])
            self.assertEqual(plan["mixed"], [])
            self.assertTrue(all(parse_expert_key(k) is None for k in plan["keep_keys"]))

    def test_a_mixed_shard_is_linked_not_dropped(self):
        index = {"weight_map": {
            "model.layers.1.mlp.experts.0.gate_proj.weight": "mixed.safetensors",
            "model.layers.1.input_layernorm.weight": "mixed.safetensors",
        }}
        plan = plan_source(index)
        self.assertEqual(plan["mixed"], ["mixed.safetensors"])
        self.assertEqual(plan["hardlink"], ["mixed.safetensors"])
        self.assertEqual(plan["drop"], [])
        self.assertEqual(plan["keep_keys"], ["model.layers.1.input_layernorm.weight"])

    def test_the_expert_key_grammar_is_mimos(self):
        self.assertEqual(
            parse_expert_key("model.layers.7.mlp.experts.13.down_proj.weight_scale"),
            (7, 13, "down_proj", "weight_scale"))
        self.assertIsNone(parse_expert_key("model.layers.7.mlp.gate.weight"))
        self.assertIsNone(parse_expert_key("model.mtp.layers.0.mlp.gate_proj.weight"))

    def test_moe_layers_come_from_moe_layer_freq(self):
        self.assertEqual(moe_layers(_syn_config()), [1, 2])
        with self.assertRaises(RuntimeError):
            moe_layers({"num_hidden_layers": 3, "moe_layer_freq": [0, 1]})

    def test_a_layer_range_names_only_moe_layers(self):
        self.assertEqual(parse_layer_range("1-3,5", [1, 2, 3, 4, 5]), [1, 2, 3, 5])
        self.assertEqual(parse_layer_range(None, [1, 2]), [1, 2])
        with self.assertRaises(RuntimeError):
            parse_layer_range("0", [1, 2])


class ConfigTests(unittest.TestCase):
    def test_k_is_a_json_number_not_a_string(self):
        block = expert_quant_block(k=2.5, codebook="tiny", window=16,
                                   quantizer="direct", calibration="none-direct")
        text = json.dumps(block)
        self.assertIn('"k": 2.5', text)
        self.assertNotIn('"k": "2.5"', text)
        self.assertEqual(block["out_scales"], "svh")
        self.assertEqual(block["format"], "exl3")

    def test_an_integer_rate_stays_an_integer(self):
        self.assertIn('"k": 4', json.dumps(expert_quant_block(
            k=parse_k(4), codebook="tiny", window=16, quantizer="direct", calibration="x")))

    def test_the_mxfp4_store_dtype_cannot_survive_the_rewrite(self):
        cfg = rewrite_config(_syn_config(), expert_quant_block(
            k=2.5, codebook="tiny", window=16, quantizer="direct", calibration="none-direct"))
        self.assertNotIn("store_dtype", cfg["quantization_config"])
        self.assertNotIn("mxfp4_block_size", cfg["quantization_config"])
        self.assertEqual(cfg["quantization_config"]["quant_method"], "fp8")
        self.assertEqual(cfg["expert_quant"]["replaced_store_dtype"], "mxfp4")

    def test_the_rewrite_does_not_mutate_the_source_config(self):
        cfg = _syn_config()
        rewrite_config(cfg, {"format": "exl3"})
        self.assertEqual(cfg["quantization_config"]["store_dtype"], "mxfp4")


class ConvertTests(unittest.TestCase):
    def _convert(self, td, **kw):
        root, out = Path(td) / "src", Path(td) / "out"
        if not root.exists():
            write_synthetic_source(root)
        args = dict(k=2.5, codebook="tiny", window=16, quantizer="direct",
                    batch_experts=2, scratch_gb=0.5, verbose=False)
        args.update(kw)
        return root, out, convert(root, out, **args)

    def test_a_synthetic_pack_has_one_shard_per_layer_and_projection(self):
        with tempfile.TemporaryDirectory() as td:
            root, out, stats = self._convert(td)
            self.assertEqual(stats["shards"], 6)
            index = json.loads((out / "model.safetensors.index.json").read_text())
            wm = index["weight_map"]
            for layer in (1, 2):
                for proj in PROJECTIONS:
                    shard = layer_proj_shard(layer, proj)
                    self.assertTrue((out / shard).is_file())
                    base = switch_base(layer, proj)
                    for key in exl3_keys(base):
                        self.assertEqual(wm[key], shard)
            self.assertTrue(all(parse_expert_key(k) is None for k in wm))
            self.assertNotIn("model-ep0.safetensors", set(wm.values()))
            self.assertFalse((out / "model-ep0.safetensors").exists())
            self.assertTrue((out / "model-trunk.safetensors").is_file())
            self.assertEqual(os.stat(out / "model-trunk.safetensors").st_ino,
                             os.stat(root / "model-trunk.safetensors").st_ino)
            self.assertTrue((out / "tokenizer.json").is_file())
            self.assertTrue((out / "chat_template.jinja").is_file())
            self.assertEqual(index["metadata"]["total_size"],
                             sum(os.path.getsize(out / f) for f in sorted(set(wm.values()))))

    def test_the_shard_geometry_is_the_contract(self):
        with tempfile.TemporaryDirectory() as td:
            _root, out, _stats = self._convert(td)
            header, _ = read_header(out / layer_proj_shard(1, "gate_proj"))
            base = switch_base(1, "gate_proj")
            self.assertEqual(header[base + ".trellis"]["dtype"], "U16")
            self.assertEqual(header[base + ".trellis"]["shape"],
                             [SYN_EXPERTS, SYN_HIDDEN // 16, SYN_INTER // 16, 40])
            self.assertEqual(header[base + ".suh"]["shape"], [SYN_EXPERTS, SYN_HIDDEN])
            self.assertEqual(header[base + ".svh"]["shape"], [SYN_EXPERTS, SYN_INTER])
            self.assertEqual(header[base + ".suh"]["dtype"], "F16")
            down, _ = read_header(out / layer_proj_shard(1, "down_proj"))
            dbase = switch_base(1, "down_proj")
            self.assertEqual(down[dbase + ".trellis"]["shape"],
                             [SYN_EXPERTS, SYN_INTER // 16, SYN_HIDDEN // 16, 40])

    def test_the_pack_reconstructs_the_dequantized_source(self):
        with tempfile.TemporaryDirectory() as td:
            root, out, _stats = self._convert(td)
            _ensure_lib()
            from ponyexl3.ref.reconstruct import reconstruct_public_weights
            reader = SourceReader(root, json.loads(
                (root / "model.safetensors.index.json").read_text())["weight_map"])
            want = reader.public(1, 0, "gate_proj")
            header, off = read_header(out / layer_proj_shard(1, "gate_proj"))
            base = switch_base(1, "gate_proj")
            trellis = read_raw(out / layer_proj_shard(1, "gate_proj"), off, header[base + ".trellis"])
            suh = read_raw(out / layer_proj_shard(1, "gate_proj"), off, header[base + ".suh"])
            svh = read_raw(out / layer_proj_shard(1, "gate_proj"), off, header[base + ".svh"])
            got = reconstruct_public_weights(trellis[0], suh[0], svh[0], 2.5, tiny=True)
            got = np.asarray(got, dtype=np.float32)
            num = float(np.sum((got - want) ** 2))
            den = float(np.sum(want ** 2)) + 1e-20
            self.assertLess(math.sqrt(num / den), 0.75)
            self.assertTrue(np.all(np.isfinite(got)))

    def test_resume_skips_a_valid_shard_and_rewrites_another_k(self):
        with tempfile.TemporaryDirectory() as td:
            root, out, first = self._convert(td)
            self.assertEqual(first["skipped"], 0)
            stamp = os.stat(out / layer_proj_shard(1, "gate_proj")).st_mtime_ns
            again = convert(root, out, k=2.5, codebook="tiny", window=16, quantizer="direct",
                            batch_experts=2, scratch_gb=0.5, resume=True, verbose=False)
            self.assertEqual(again["skipped"], 6)
            self.assertEqual(again["shards"], 0)
            self.assertEqual(os.stat(out / layer_proj_shard(1, "gate_proj")).st_mtime_ns, stamp)
            other = convert(root, out, k=3, codebook="tiny", window=16, quantizer="direct",
                            batch_experts=2, scratch_gb=0.5, resume=True, verbose=False)
            self.assertEqual(other["skipped"], 0)
            self.assertEqual(other["shards"], 6)
            header, _ = read_header(out / layer_proj_shard(1, "gate_proj"))
            self.assertEqual(header[switch_base(1, "gate_proj") + ".trellis"]["shape"][-1], 48)

    def test_every_shard_carries_the_settings_that_wrote_it(self):
        with tempfile.TemporaryDirectory() as td:
            _root, out, _stats = self._convert(td)
            got = read_stamp(out / layer_proj_shard(1, "gate_proj"))
            self.assertEqual(got["k"], "2.5")
            self.assertEqual(got["codebook"], "tiny")
            self.assertEqual(got["window"], "16")
            self.assertEqual(got["quantizer"], "direct")
            self.assertEqual(got["imatrix_sha256"], "none")
            self.assertEqual(got["g_scale"], "gss")
            self.assertEqual(got["converter"], CONVERTER_VERSION)
            self.assertEqual(got, shard_stamp(k=2.5, codebook="tiny", window=16,
                                              quantizer="direct", imatrix_sha=None))

    def test_the_stamp_version_is_a_constant_not_the_scripts_bytes(self):
        """Hashing the file would make a comment edit invalidate every written shard."""
        self.assertEqual(converter_version(), CONVERTER_VERSION)
        self.assertNotIn("/", CONVERTER_VERSION)
        self.assertLess(len(CONVERTER_VERSION), 64)

    def test_a_g_scale_one_shard_is_not_adopted_by_a_searched_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root, out, first = self._convert(td, layers="1", g_scale=False)
            self.assertEqual(read_stamp(out / layer_proj_shard(1, "gate_proj"))["g_scale"],
                             "one")
            again = convert(root, out, k=2.5, codebook="tiny", window=16,
                            quantizer="direct", batch_experts=2, layers="1",
                            scratch_gb=0.5, resume=True, verbose=False)
            self.assertEqual(again["skipped"], 0)
            self.assertEqual(len(again["rewritten"]), 3)
            self.assertIn("g_scale=one != gss", again["rewritten"][0])

    def test_a_direct_shard_is_not_adopted_by_an_ldlq_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root, out, _ = self._convert(td, layers="1")
            before = self._payload(out / layer_proj_shard(1, "gate_proj"))
            stats = convert(root, out, k=2.5, codebook="tiny", window=16, quantizer="ldlq",
                            imatrix=self._imatrix([8.0] * SYN_EXPERTS), imatrix_sha="deadbeef",
                            batch_experts=2, layers="1", scratch_gb=0.5, resume=True,
                            verbose=False)
            self.assertEqual(stats["skipped"], 0)
            self.assertEqual(stats["shards"], 3)
            self.assertEqual(len(stats["rewritten"]), 3)
            for line in stats["rewritten"]:
                self.assertIn("quantizer=direct != ldlq-rotated", line)
                self.assertIn("imatrix_sha256=none != deadbeef", line)
            after = read_stamp(out / layer_proj_shard(1, "gate_proj"))
            self.assertEqual(after["quantizer"], "ldlq-rotated")
            self.assertEqual(after["imatrix_sha256"], "deadbeef")
            self.assertNotEqual(self._payload(out / layer_proj_shard(1, "gate_proj")), before)

    def test_the_same_settings_resume_and_a_changed_rate_does_not(self):
        with tempfile.TemporaryDirectory() as td:
            root, out, _ = self._convert(td, layers="1")
            same = convert(root, out, k=2.5, codebook="tiny", window=16, quantizer="direct",
                           batch_experts=2, layers="1", scratch_gb=0.5, resume=True,
                           verbose=False)
            self.assertEqual(same["skipped"], 3)
            self.assertEqual(same["rewritten"], [])
            other = convert(root, out, k=3, codebook="tiny", window=16, quantizer="direct",
                            batch_experts=2, layers="1", scratch_gb=0.5, resume=True,
                            verbose=False)
            self.assertEqual(other["skipped"], 0)
            self.assertEqual(len(other["rewritten"]), 3)
            self.assertIn("trellis shape", other["rewritten"][0])

    def test_an_unstamped_shard_is_named_not_adopted(self):
        with tempfile.TemporaryDirectory() as td:
            root, out, _ = self._convert(td, layers="1")
            shard = out / layer_proj_shard(1, "gate_proj")
            header, off = read_header(shard)
            named = {key: (meta["dtype"], tuple(meta["shape"]),
                           read_raw(shard, off, meta).tobytes()) for key, meta in header.items()}
            write_safetensors_raw(str(shard), named)
            again = convert(root, out, k=2.5, codebook="tiny", window=16, quantizer="direct",
                            batch_experts=2, layers="1", scratch_gb=0.5, resume=True,
                            verbose=False)
            self.assertEqual(again["skipped"], 2)
            self.assertEqual(again["rewritten"], [f"{layer_proj_shard(1, 'gate_proj')}: no stamp"])

    def test_the_buffer_pool_stays_under_the_cache_ceiling(self):
        """The per-launch `clear_cache` is gone (it measured 1.27x slower on the real
        shape), so the ceiling `convert` sets is what bounds the pool instead."""
        import mlx.core as mx
        with tempfile.TemporaryDirectory() as td:
            mx.clear_cache()
            self._convert(td, layers="1", scratch_gb=0.25)
            self.assertLessEqual(mx.get_cache_memory(), 4 * int(0.25 * (1 << 30)))

    def test_without_resume_an_existing_shard_is_rewritten(self):
        with tempfile.TemporaryDirectory() as td:
            root, out, _first = self._convert(td)
            again = convert(root, out, k=2.5, codebook="tiny", window=16, quantizer="direct",
                            batch_experts=2, scratch_gb=0.5, resume=False, verbose=False)
            self.assertEqual(again["skipped"], 0)
            self.assertEqual(again["shards"], 6)

    def test_ldlq_without_an_imatrix_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "src"
            write_synthetic_source(root)
            with self.assertRaises(RuntimeError):
                convert(root, Path(td) / "out", quantizer="ldlq", imatrix=None, verbose=False)

    @staticmethod
    def _payload(path) -> bytes:
        """The tensor bytes alone: two packs made the same way differ only in the stamp."""
        header, off = read_header(path)
        return b"".join(read_raw(path, off, header[key]).tobytes() for key in sorted(header))

    @staticmethod
    def _imatrix(rows_vec) -> dict:
        rng = np.random.default_rng(21)
        store: dict[str, np.ndarray] = {}
        for layer in (1, 2):
            gu, dn, rows = imatrix_layer_keys(layer)
            store[gu] = np.abs(rng.standard_normal(SYN_EXPERTS * SYN_HIDDEN, dtype=np.float32)) + 0.1
            store[dn] = np.abs(rng.standard_normal(SYN_EXPERTS * SYN_INTER, dtype=np.float32)) + 0.1
            store[rows] = np.asarray(rows_vec, dtype=np.float32)
        return store

    def test_ldlq_does_not_collapse_into_the_direct_pack(self):
        """Feeding LDLQ the raw public diagonal produced an identity factor and a pack
        byte-identical to `--quantizer direct`. Under the rotated Hessian the two must
        differ: that difference IS the calibration."""
        with tempfile.TemporaryDirectory() as td:
            root, out, _ = self._convert(td)
            out2 = Path(td) / "out2"
            stats = convert(root, out2, k=2.5, codebook="tiny", window=16, quantizer="ldlq",
                            imatrix=self._imatrix([8.0] * SYN_EXPERTS), batch_experts=2,
                            scratch_gb=0.5, verbose=False)
            self.assertEqual(stats["prior_fallbacks"], 0)
            self.assertEqual(stats["stamp"]["quantizer"], "ldlq-rotated")
            differing = [proj for proj in PROJECTIONS
                         if self._payload(out / layer_proj_shard(1, proj))
                         != self._payload(out2 / layer_proj_shard(1, proj))]
            self.assertEqual(differing, list(PROJECTIONS))

    def test_an_expert_with_no_routed_token_takes_the_prior(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "src"
            write_synthetic_source(root)
            stats = convert(root, Path(td) / "out", k=2.5, codebook="tiny", window=16,
                            quantizer="ldlq", imatrix=self._imatrix([0.0, 5.0]),
                            batch_experts=2, layers="1", scratch_gb=0.5, verbose=False)
            self.assertEqual(stats["prior_fallbacks"], 3)

    def test_a_layer_range_writes_a_partial_pack(self):
        with tempfile.TemporaryDirectory() as td:
            root, out, stats = self._convert(td, layers="1")
            self.assertTrue(stats["partial"])
            self.assertEqual(stats["layers"], [1])
            self.assertEqual(stats["shards"], 3)
            wm = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
            self.assertIn(switch_base(1, "gate_proj") + ".trellis", wm)
            self.assertNotIn(switch_base(2, "gate_proj") + ".trellis", wm)


def self_test() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


# -------------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None, help="source MiMo-V2.6-Flash checkpoint")
    ap.add_argument("--dst", default=None, help="output pack directory")
    ap.add_argument("--k", default=str(K_DEFAULT), help="trellis rate, any multiple of 1/16")
    ap.add_argument("--codebook", default=CODEBOOK_DEFAULT, choices=("tiny", "mul1", "mcg"))
    ap.add_argument("--window", type=int, default=WINDOW_DEFAULT,
                    help="codeword window the search hashes, 8..16 served; stamped into the pack")
    ap.add_argument("--quantizer", default="ldlq", choices=("ldlq", "direct"))
    ap.add_argument("--imatrix", default=None, help="safetensors imatrix; required for ldlq")
    ap.add_argument("--layers", default=None, help="MoE layer range for pilots, e.g. 1 or 1-4,9")
    ap.add_argument("--batch-experts", type=int, default=BATCH_EXPERTS_DEFAULT)
    ap.add_argument("--resume", action="store_true", help="keep shards that already match")
    ap.add_argument("--scratch-gb", type=float, default=SCRATCH_GB_DEFAULT,
                    help="Metal search scratch budget; sets the tiles per launch")
    ap.add_argument("--no-quality", action="store_true",
                    help="skip the per-expert imatrix-weighted reconstruction error")
    ap.add_argument("--no-g-scale", action="store_true",
                    help="skip the global codebook-scale search (regularize at g=1)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not (args.src and args.dst):
        ap.error("--src and --dst are required unless --self-test")
    k = parse_k(args.k)
    imatrix = imatrix_sha = None
    if args.imatrix:
        imatrix_path = os.path.expanduser(args.imatrix)
        imatrix = load_imatrix(imatrix_path)
        imatrix_sha = file_sha256(imatrix_path)
    t0 = time.perf_counter()
    stats = convert(
        os.path.expanduser(args.src), os.path.expanduser(args.dst),
        k=k, codebook=args.codebook, window=args.window, quantizer=args.quantizer,
        imatrix=imatrix, imatrix_sha=imatrix_sha, layers=args.layers,
        batch_experts=args.batch_experts,
        resume=args.resume, scratch_gb=args.scratch_gb, g_scale=not args.no_g_scale,
        quality=not args.no_quality,
    )
    wall = time.perf_counter() - t0
    print(f"pack {args.dst}: {stats['shards']} shards, {stats['skipped']} skipped, "
          f"{stats['prior_fallbacks']} prior fallbacks, {stats['linked']} linked files, "
          f"{stats['total_size'] / 1e9:.2f} GB indexed, {wall:.1f}s wall "
          f"({stats['seconds']:.1f}s quantizing, {stats['stage_seconds']:.1f}s staging)",
          flush=True)
    se = stats["search"]
    print(f"search: {se['seconds']:.1f}s GPU ({100.0 * se['seconds'] / max(wall, 1e-9):.1f}% of "
          f"wall), {se['launches']} launches, {se['tiles']} tiles "
          f"({se['tiles'] / max(se['launches'], 1):.0f}/launch, budget {stats['launch_tiles']}, "
          f"min {se['min_launch_tiles']}, max {se['max_launch_tiles']}), "
          f"{se['tiles'] / max(se['seconds'], 1e-9):.0f} tiles/s", flush=True)
    werr = stats.get("weighted_rel_err") or {}
    if werr:
        print("imatrix-weighted relative reconstruction error "
              "(tests/dsv4_imatrix.weighted_rel_err, public basis):", flush=True)
        for key in sorted(werr):
            v = werr[key]
            print(f"  {key}: mean {sum(v) / len(v):.5f}  min {min(v):.5f}  max {max(v):.5f}"
                  f"  n={len(v)}", flush=True)
        allv = [x for v in werr.values() for x in v]
        by_layer: dict[str, list[float]] = {}
        for key, v in werr.items():
            by_layer.setdefault(key.split(".", 1)[0], []).extend(v)
        for layer_key in sorted(by_layer):
            lv = by_layer[layer_key]
            print(f"  {layer_key} layer mean {sum(lv) / len(lv):.5f}", flush=True)
        print(f"  all shards mean {sum(allv) / len(allv):.5f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
