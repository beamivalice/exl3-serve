#!/usr/bin/env python3
"""The EXL3 conversion machinery both expert converters share.

`tests/convert_mimo_v26_exl3.py` (MXFP4 source, MiMo naming) and
`tests/convert_qwen38_flash_next_exl3.py` (bf16 source, qwen4_exp naming) differ only
in how they READ a checkpoint and how they NAME a pack. Everything between — the
trellis search, the imatrix rotated into the search's own basis, the bounded global
codebook-scale search, grouped LDLQ on the GPU, the shard stamp and the resume rule —
is the same math and lives here, so a fix lands for both packs at once.

PonyExl3 (`EXL3_CONVERT_LIB`, default /Users/beam/llm/ponyexl3) supplies the reference
trellis, codebooks and regularizer; MLX runs the search.
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

PREFETCH_WORKERS_DEFAULT = 3
PREFETCH_DEPTH_DEFAULT = 3

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


# ---------------------------------------------------------------- safetensors

NP_DTYPE = {
    "BF16": np.uint16, "F16": np.float16, "F32": np.float32, "F64": np.float64,
    "U16": np.uint16, "U32": np.uint32, "I32": np.int32, "I64": np.int64, "U8": np.uint8,
    "I8": np.int8, "F8_E4M3": np.uint8, "BOOL": np.bool_,
}


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


def weighted_rel_err_mlx(public, public_hat, channel_weights):
    import mlx.core as mx
    om = channel_weights.reshape(-1, 1)
    delta = (public_hat - public).astype(mx.float32)
    num = mx.sum(om * delta * delta)
    den = mx.sum(om * public.astype(mx.float32) ** 2)
    return num / mx.maximum(den, 1e-30)


def quality_errors_mlx(recon, publics, calibrations, suh16, svh16):
    import mlx.core as mx
    suh_mx = mx.array(suh16, dtype=mx.float32)
    svh_mx = mx.array(svh16, dtype=mx.float32)
    result = []
    for start in range(0, len(publics), 4):
        errors = []
        for ei in range(start, min(start + 4, len(publics))):
            public = publics[ei]
            hat = public_from_inner_mlx(recon[ei], suh_mx[ei], svh_mx[ei])
            weights = (mx.ones(public.shape[0], dtype=mx.float32) if calibrations[ei] is None
                       else mx.array(calibrations[ei], dtype=mx.float32))
            errors.append(weighted_rel_err_mlx(mx.array(public, dtype=mx.float32), hat, weights))
        result.extend(np.array(mx.stack(errors)).tolist())
    return result


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
    regs = [regularize_public_weight(np.asarray(p, dtype=np.float32), seed=s, hessian_diag=cal)
            for p, s, cal in zip(publics, seeds, calibrations)]
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
        err_out.extend(quality_errors_mlx(recon, publics, calibrations, suh16, svh16))
        del recon
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


# ------------------------------------------------------------------ pack stamp

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


QUANTIZER_STAMP = {"direct": "direct", "ldlq": "ldlq-rotated"}


def shard_stamp(*, k, codebook: str, window: int, quantizer: str,
                imatrix_sha: str | None, converter: str,
                g_scale: str = "gss") -> dict[str, str]:
    """What a shard was made with. Every value is a string: safetensors `__metadata__`
    is a string map, and `--resume` compares the whole dict. `converter` is the CALLING
    converter's own version constant — the two packs evolve independently."""
    return {
        "format": "exl3",
        "k": format_k(k),
        "codebook": codebook,
        "window": str(int(window)),
        "quantizer": QUANTIZER_STAMP[quantizer],
        "g_scale": g_scale,
        "imatrix_sha256": imatrix_sha or "none",
        "converter": converter,
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


# ---------------------------------------------------------------- prefetching

def prefetch_counts(window: int, workers: int | None, depth: int | None) -> tuple[int, int]:
    if workers is None:
        workers = PREFETCH_WORKERS_DEFAULT if window == 8 else 2
    if depth is None:
        depth = PREFETCH_DEPTH_DEFAULT if window == 8 else 2
    if workers < 1 or depth < 1:
        raise ValueError("prefetch workers and depth must be positive")
    return workers, depth


def prefetch_batches(pool, load, spans, capacity: int = 2):
    if capacity < 1:
        raise ValueError("prefetch capacity must be positive")
    spans = iter(spans)
    pending = []
    try:
        for _ in range(capacity):
            span = next(spans, None)
            if span is None:
                break
            pending.append(pool.submit(load, *span))
        while pending:
            batch = pending.pop(0).result()
            span = next(spans, None)
            if span is not None:
                pending.append(pool.submit(load, *span))
            yield batch
            del batch
    finally:
        for future in pending:
            future.cancel()
