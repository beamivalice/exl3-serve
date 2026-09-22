#!/usr/bin/env python3
"""Replace routed-expert affine banks in a qwen4_exp 4/8 pack with stacked EXL3 K4.

Input: bf16 HF checkpoint + the existing mixed 4/8 pack. Output: a new pack dir
where every routed gate/up/down bank is EXL3 K4 (MUL1), stacked [E, ...] per
projection so gather kernels index expert e on axis 0. Every non-expert file
from the 4/8 pack is hard-linked; mixed shards are rewritten with the expert
tensors dropped and remaining tensors copied as raw bytes. config.json carries
expert_quant = {format: exl3, k, codebook, window}.

`--quantizer ldlq` runs LDLQ under the imatrix Hessian ROTATED into the inner
basis the search works in, preceded by a per-expert global codebook-scale search;
`--quantizer direct` is the calibration-free path. Calibration rows, when captured,
are the MLP input (hidden) for gate/up and the SwiGLU activation (intermediate) for
down. `--window` (default 16) is the codeword width the search hashes; the pack
records it in `expert_quant.window` and every decoder masks to it (the engine admits
8..16), so a narrower window trades weight error for search time at the same bits.
Every shard carries a `__metadata__` stamp of what wrote it and a resume adopts only
a shard whose whole stamp matches. The shared machinery is `tests/exl3_convert_common.py`.

  python3 tests/convert_qwen38_flash_next_exl3.py --self-test
  python3 tests/convert_qwen38_flash_next_exl3.py \\
      --hf /Users/beam/llm/models/Qwen/Qwen3.8-Flash-Next \\
      --pack /Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit \\
      --dst /path/to/out --quantizer ldlq --k 3 --window 8 --imatrix <imatrix.safetensors>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_dsv4_weights import bf16_to_f32, write_safetensors_raw  # noqa: E402
from convert_qwen38_flash_next import read_header, read_raw, rename  # noqa: E402
import exl3_convert_common as _common  # noqa: E402
from exl3_convert_common import (  # noqa: E402,F401
    HAD_BLOCK, calib_for_expert, codebook_mode, diag_hessian, file_sha256, format_k,
    imatrix_expert_vector, k_from_packed, launch_tiles_for, load_imatrix, packed_hw,
    parse_k, prefetch_batches, prepare_expert_bank, quantize_prepared_bank, read_stamp,
    reset_search_stats, search_stats_snapshot, shard_reuse_refusal, validate_window)

K = 4
CODEBOOK = "mul1"
WINDOW_DEFAULT = 16
BATCH_EXPERTS_DEFAULT = 32
SCRATCH_GB_DEFAULT = 4.0
PROJECTIONS = ("gate", "up", "down")
PACKED_K4 = 256 * K // 16

# Bump on any change to the pack format or to what the quantizer produces — never for a
# comment or a test. A shard's stamp carries this and resume refuses a shard whose stamp
# differs, so hashing the file itself would make an editorial change cost a rerun.
CONVERTER_VERSION = "qwen4-exl3-1-rotated-gss"


def converter_version() -> str:
    return CONVERTER_VERSION


def shard_stamp(**kw) -> dict[str, str]:
    """This converter's own version constant, stamped into every shard it writes."""
    return _common.shard_stamp(converter=converter_version(), **kw)


def expert_seed(layer: int, expert: int, proj: str, *, mtp: bool = False) -> int:
    """Stable per (bank, layer, expert, projection), so a resumed or interrupted run
    reproduces the same pack as one that wrote every shard in a single pass."""
    base = (layer * 1_000_003 + expert * 17 + PROJECTIONS.index(proj)) & 0x7FFF_FFFF
    return (base + 0x4000_0000) & 0x7FFF_FFFF if mtp else base


SWITCH = ".mlp.switch_mlp."
HF_GATE_UP = ".mlp.experts.gate_up_proj"
HF_DOWN = ".mlp.experts.down_proj"


def _load_component_repacker():
    path = Path(__file__).resolve().parents[1] / "scripts" / "repack_exl3.py"
    if not path.is_file():
        raise RuntimeError(f"component repacker not found: {path}")
    spec = importlib.util.spec_from_file_location("_mlx_serve_repack_exl3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load component repacker: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(spec.name, None)
        raise RuntimeError(f"cannot load component repacker {path}: {exc}") from exc
    if not callable(getattr(module, "repack", None)):
        raise RuntimeError(f"component repacker has no repack() API: {path}")
    return module


def repack_component_output(
    source: str | Path,
    destination: str | Path,
    share_with: str | Path | None = None,
):
    source, destination = Path(source), Path(destination)
    share = Path(share_with) if share_with is not None else None
    if not source.is_dir():
        raise RuntimeError(f"component repack source is not a directory: {source}")
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"component output destination must not already exist: {destination}")
    source_real = source.resolve()
    destination_real = destination.resolve()
    if destination_real == source_real or source_real in destination_real.parents:
        raise RuntimeError(
            f"component output destination must be outside staged source: {destination}"
        )
    if share is not None and not share.is_dir():
        raise RuntimeError(f"--share-with is not an existing directory: {share}")
    repacker = _load_component_repacker()
    return repacker.repack(source, destination, share)


def is_pack_expert_key(key: str) -> bool:
    return SWITCH in key


def exl3_keys(base: str) -> tuple[str, str, str]:
    return base + ".trellis", base + ".suh", base + ".svh"


def mlx_switch_base(hf_key: str, proj: str) -> str:
    nk = rename(hf_key)
    if nk.endswith(HF_GATE_UP):
        return nk[: -len("experts.gate_up_proj")] + f"switch_mlp.{proj}_proj"
    if nk.endswith(HF_DOWN):
        return nk[: -len("experts.down_proj")] + "switch_mlp.down_proj"
    raise ValueError(hf_key)


def plan_pack(pack_index: dict) -> dict:
    wm = pack_index["weight_map"]
    files: dict[str, dict[str, list[str]]] = {}
    keep_keys: list[str] = []
    expert_bases: list[str] = []
    for key, fname in wm.items():
        slot = files.setdefault(fname, {"expert": [], "other": []})
        if is_pack_expert_key(key):
            slot["expert"].append(key)
            if key.endswith(".weight"):
                expert_bases.append(key[: -len(".weight")])
        else:
            slot["other"].append(key)
            keep_keys.append(key)
    hardlink, rewrite, drop = [], [], []
    for fname, slot in files.items():
        if slot["expert"] and slot["other"]:
            rewrite.append(fname)
        elif slot["expert"]:
            drop.append(fname)
        else:
            hardlink.append(fname)
    hardlink.sort()
    rewrite.sort()
    drop.sort()
    exl3 = []
    for base in sorted(set(expert_bases)):
        exl3.extend(exl3_keys(base))
    return {
        "hardlink": hardlink,
        "rewrite": rewrite,
        "drop": drop,
        "keep_keys": keep_keys,
        "exl3_keys": exl3,
        "files": files,
    }


def _ensure_lib() -> None:
    lib = os.environ.get("EXL3_CONVERT_LIB", "/Users/beam/llm/ponyexl3")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    try:
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
    except Exception:
        pass


def _quantize_direct_batch(inners: list[np.ndarray], k: int, cb) -> list[np.ndarray]:
    from ponyexl3.convert.direct import quantize_inner_matrix_direct
    fat = np.concatenate(inners, axis=1)
    packed, _, _ = quantize_inner_matrix_direct(
        fat, k=k, cb=cb, search_backend="metal", return_states=False
    )
    out = []
    col = 0
    for inner in inners:
        ot = inner.shape[1] // 16
        out.append(packed[:, col : col + ot].copy())
        col += ot
    return out


def expert_public(bank: np.ndarray, ei: int, lo: int, hi: int) -> np.ndarray:
    """One expert's EXL3 public weight [in, out] out of a resident [E, out, in] bank.

    HF shards stay bf16 until here — a whole f32 copy of a routed bank is 6.7 GB on the
    real checkpoint, and only the batch the GPU is about to search needs to be float."""
    rows = bank[ei, lo:hi]
    if rows.dtype == np.uint16:
        rows = bf16_to_f32(rows)
    return np.ascontiguousarray(np.asarray(rows, dtype=np.float32).T)


def bank_calibrations(
    e: int, in_dim: int, *, calibration, imatrix_flat, routed_rows,
    zero_routed: list | None, layer_key: str,
) -> list:
    """The per-expert calibration diagonal `regularize_public_weight` and the rotated
    Hessian both read. An expert no calibration token reached takes the flat prior."""
    cals = []
    for ei in range(e):
        cal = calibration
        if imatrix_flat is not None:
            ntok = int(routed_rows[ei]) if routed_rows is not None else 1
            mode, vec = calib_for_expert(ntok, imatrix_expert_vector(imatrix_flat, ei, in_dim))
            if mode != "imatrix-diagonal":
                if zero_routed is not None:
                    zero_routed.append(f"{layer_key}#{ei}")
                cal = None
            else:
                cal = vec
        cals.append(cal)
    return cals


def _copy_raw_tensors(src_file: Path, keys: list[str]) -> dict:
    header, data_off = read_header(src_file)
    out = {}
    for key in keys:
        meta = header[key]
        raw = read_raw(src_file, data_off, meta)
        out[key] = (meta["dtype"], tuple(meta["shape"]), np.ascontiguousarray(raw).tobytes())
    return out


def _weighted_row_err(w: np.ndarray, w_hat: np.ndarray, v: np.ndarray) -> float:
    d = (w - w_hat).astype(np.float64)
    num = float(np.sum(v.astype(np.float64)[:, None] * (d * d)))
    den = float(np.sum(v.astype(np.float64)[:, None] * (w.astype(np.float64) ** 2))) + 1e-20
    return float(np.sqrt(num / den))


def _output_err(w: np.ndarray, w_hat: np.ndarray, v: np.ndarray, n_rows: int = 256, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_rows, w.shape[0])).astype(np.float32)
    x *= np.sqrt(np.maximum(v, 1e-8))[None, :]
    y = x @ w.astype(np.float32)
    yh = x @ w_hat.astype(np.float32)
    d = y - yh
    return float(np.sqrt(np.mean(d * d) / (np.mean(y * y) + 1e-20)))


def bench_batch_quality() -> int:
    import time
    _ensure_lib()
    from ponyexl3.convert.direct import quantize_inner_matrix_direct
    from ponyexl3.convert.hessian import block_ldl, ldlq_inner_matrix, prepare_hessian_for_ldl
    from ponyexl3.convert.regularize import regularize_public_weight
    from ponyexl3.ref.codebook import CodebookMode
    from ponyexl3.ref.reconstruct import reconstruct_public_weights
    from convert_dsv4_weights import mlx_affine_dequant_f32, mlx_affine_quant

    rng = np.random.default_rng(0)
    inn, outn = 2560, 640
    w = rng.standard_normal((inn, outn), dtype=np.float32)
    v = (np.abs(rng.standard_normal(inn)) + 0.05).astype(np.float32)
    cb = CodebookMode.MCG

    def quality(what: np.ndarray) -> tuple[float, float]:
        return _weighted_row_err(w, what, v), _output_err(w, what, v)

    wq, sc, bi = mlx_affine_quant(w.T, 4, 64)
    w_aff = mlx_affine_dequant_f32(
        np.frombuffer(wq[2], dtype=np.uint32).reshape(wq[1]),
        np.frombuffer(sc[2], dtype=np.uint16).reshape(sc[1]),
        np.frombuffer(bi[2], dtype=np.uint16).reshape(bi[1]),
        4, 64,
    ).T
    q_aff = quality(w_aff)
    import mlx.core as mx
    mx.set_default_device(mx.gpu)

    reg = regularize_public_weight(w, seed=1)
    t0 = time.perf_counter()
    packed, _, _ = quantize_inner_matrix_direct(reg.inner, k=4, cb=cb, search_backend="metal", return_states=False)
    t_direct = time.perf_counter() - t0
    w_direct = reconstruct_public_weights(packed, reg.suh.astype(np.float16), reg.svh.astype(np.float16), 4, mcg=True).astype(np.float32)
    q_direct = quality(w_direct)

    w_scaled = w * np.sqrt(v)[:, None]
    reg_s = regularize_public_weight(w_scaled, seed=1)
    packed_s, _, _ = quantize_inner_matrix_direct(reg_s.inner, k=4, cb=cb, search_backend="metal", return_states=False)
    w_row = reconstruct_public_weights(packed_s, reg_s.suh.astype(np.float16), reg_s.svh.astype(np.float16), 4, mcg=True).astype(np.float32)
    w_row = w_row / np.sqrt(v)[:, None]
    q_row = quality(w_row)

    t1 = time.perf_counter()
    prep = prepare_hessian_for_ldl(diag_hessian(v))
    ldl = block_ldl(prep.hessian)
    res = ldlq_inner_matrix(reg.inner, ldl.l, k=4, cb=cb, hessian=prep.hessian, search_backend="metal", collect_states=False, compute_proxy=False)
    t_ldlq = time.perf_counter() - t1
    w_ldlq = reconstruct_public_weights(res.packed, reg.suh.astype(np.float16), reg.svh.astype(np.float16), 4, mcg=True).astype(np.float32)
    q_ldlq = quality(w_ldlq)

    print("quality (imatrix-weighted relRMS, output relRMS on 256 rows):")
    print(f"  affine-4-g64          {q_aff[0]:.5f}  {q_aff[1]:.5f}")
    print(f"  direct                {q_direct[0]:.5f}  {q_direct[1]:.5f}  {t_direct:.3f}s")
    print(f"  direct-row-sqrt(v)    {q_row[0]:.5f}  {q_row[1]:.5f}")
    print(f"  ldlq-diag(v)          {q_ldlq[0]:.5f}  {q_ldlq[1]:.5f}  {t_ldlq:.3f}s")

    for n in (16, 32, 64):
        regs = [regularize_public_weight(rng.standard_normal((inn, outn), dtype=np.float32), seed=10 + i) for i in range(n)]
        inners = [r.inner for r in regs]
        t2 = time.perf_counter()
        _quantize_direct_batch(inners, 4, cb)
        dt = time.perf_counter() - t2
        print(f"direct batch N={n}: {dt:.3f}s  {n / dt:.2f} proj/s  pack={n * 73728 / 48 / 3 / (n / dt) / 3600:.2f} h at this rate")
    return 0


def _raw_bytes(path: Path, header: dict, key: str, data_off: int) -> bytes:
    meta = header[key]
    start, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_off + start)
        return f.read(end - start)


def restack_from_exl3(src_dir: str | Path, pack_dir: str | Path, dst: str | Path) -> dict:
    src_dir, pack_dir, dst = Path(src_dir), Path(pack_dir), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    src_idx = json.loads((src_dir / "model.safetensors.index.json").read_text())
    src_cfg = json.loads((src_dir / "config.json").read_text())
    qcfg = src_cfg.get("quantization_config") or {}
    if str(qcfg.get("codebook", "")).lower() != "mul1":
        raise RuntimeError("source codebook is not mul1")
    tcfg = src_cfg.get("text_config") or src_cfg
    n_exp = int(tcfg["num_experts"])
    n_layers = int(tcfg["num_hidden_layers"])
    groups: dict[tuple, dict] = {}
    for key, fname in src_idx["weight_map"].items():
        parsed = parse_src_expert_key(key)
        if parsed is None:
            continue
        prefix, layer, ei, proj, suffix = parsed
        g = groups.setdefault((prefix, layer, proj), {})
        g.setdefault(ei, {})[suffix] = (fname, key)
    for (prefix, layer, proj), experts in groups.items():
        missing = [i for i in range(n_exp) if i not in experts]
        if missing:
            raise RuntimeError(f"missing experts {missing[:5]} for {prefix} L{layer} {proj}")
        for ei in range(n_exp):
            if "mul1" not in experts[ei] or "trellis" not in experts[ei]:
                raise RuntimeError(f"no mul1/trellis for expert {ei} L{layer} {proj}")
    pack_index = json.loads((pack_dir / "model.safetensors.index.json").read_text())
    plan = plan_pack(pack_index)
    for fname in plan["hardlink"]:
        src, out = pack_dir / fname, dst / fname
        if out.exists() or out.is_symlink():
            out.unlink()
        os.link(src, out)
    for name in sorted(os.listdir(pack_dir)):
        if name.startswith(".") or (name.startswith("model-") and name.endswith(".safetensors")):
            continue
        if name in ("model.safetensors.index.json", "config.json"):
            continue
        srcp = pack_dir / name
        if not srcp.is_file():
            continue
        out = dst / name
        if out.exists() or out.is_symlink():
            out.unlink()
        os.link(srcp, out)
    for i, fname in enumerate(plan["rewrite"]):
        other_keys = plan["files"][fname]["other"]
        write_safetensors_raw(str(dst / fname), _copy_raw_tensors(pack_dir / fname, other_keys))
    weight_map = {key: pack_index["weight_map"][key] for key in plan["keep_keys"]}
    header_cache: dict[str, tuple] = {}
    k_hist = {2: 0, 3: 0, 4: 0}

    def header_of(fname: str):
        if fname not in header_cache:
            header_cache[fname] = read_header(src_dir / fname)
        return header_cache[fname]

    for (prefix, layer, proj), experts in sorted(groups.items()):
        shard = layer_proj_shard(layer, proj) if not prefix.startswith("mtp") else f"model-exl3-mtp-L{layer:02d}-{proj}.safetensors"
        e0 = experts[0]
        h0, off0 = header_of(e0["trellis"][0])
        tshape = list(h0[e0["trellis"][1]]["shape"])
        k_from_packed(tshape[-1])
        suh_shape = list(header_of(e0["suh"][0])[0][e0["suh"][1]]["shape"])
        svh_shape = list(header_of(e0["svh"][0])[0][e0["svh"][1]]["shape"])
        tdtype = h0[e0["trellis"][1]]["dtype"]

        def check(suffix, want_shape, want_dtype, ei):
            fname, key = experts[ei][suffix]
            meta = header_of(fname)[0][key]
            if list(meta["shape"]) != list(want_shape) or meta["dtype"] != want_dtype:
                raise RuntimeError(
                    f"{key} is {meta['dtype']}{list(meta['shape'])}, expected {want_dtype}{list(want_shape)}"
                )

        stacked_t = np.empty((n_exp, *tshape), dtype=np.uint16)
        stacked_suh = np.empty((n_exp, *suh_shape), dtype=np.float16)
        stacked_svh = np.empty((n_exp, *svh_shape), dtype=np.float16)
        k_hist[k_from_packed(tshape[-1])] += n_exp
        for ei in range(n_exp):
            check("trellis", tshape, tdtype, ei)
            check("suh", suh_shape, "F16", ei)
            check("svh", svh_shape, "F16", ei)
            tf, tk = experts[ei]["trellis"]
            hf, ho = header_of(tf)
            tb = _raw_bytes(src_dir / tf, hf, tk, ho)
            sh = list(hf[tk]["shape"])
            if sh[-1] != tshape[-1]:
                raise RuntimeError(f"mixed K inside {prefix} L{layer} {proj} expert {ei}: {sh} vs {tshape}")
            stacked_t[ei] = np.frombuffer(tb, dtype=np.uint16).reshape(tshape)
            sf, sk = experts[ei]["suh"]
            hf, ho = header_of(sf)
            stacked_suh[ei] = np.frombuffer(_raw_bytes(src_dir / sf, hf, sk, ho), dtype=np.float16).reshape(suh_shape)
            vf, vk = experts[ei]["svh"]
            hf, ho = header_of(vf)
            stacked_svh[ei] = np.frombuffer(_raw_bytes(src_dir / vf, hf, vk, ho), dtype=np.float16).reshape(svh_shape)
        base = dest_switch_key(prefix, layer, proj, "x")[:-2]
        named = {
            base + ".trellis": ("U16", stacked_t.shape, np.ascontiguousarray(stacked_t).tobytes()),
            base + ".suh": ("F16", stacked_suh.shape, np.ascontiguousarray(stacked_suh).tobytes()),
            base + ".svh": ("F16", stacked_svh.shape, np.ascontiguousarray(stacked_svh).tobytes()),
        }
        write_safetensors_raw(str(dst / shard), named)
        for key in named:
            weight_map[key] = shard
        print(f"restack {shard}", flush=True)
        del stacked_t, stacked_suh, stacked_svh
    total = sum(os.path.getsize(dst / f) for f in sorted(set(weight_map.values())))
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2
    ))
    modal = max(k_hist, key=lambda kk: (k_hist[kk], kk))
    print(f"k histogram tensors={k_hist} modal={modal}", flush=True)
    cfg = json.loads((pack_dir / "config.json").read_text())
    cfg["expert_quant"] = expert_quant_block("restack", k=int(modal))
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    return {"weight_map": weight_map, "k_hist": k_hist, "k": int(modal)}


NGRAM_SHARD_RE = __import__("re").compile(r"\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")
MTP_SRC_PREFIX = "mtp."
MTP_DST_PREFIX = "language_model.mtp."
TENSOR_SUFFIXES = (".weight", ".scales", ".biases", ".trellis", ".suh", ".svh")
AFFINE_BITS = (2, 3, 4, 5, 6, 8)
AFFINE_GROUPS = (32, 64, 128)


def bf16_ngram_shards(src_dir: str | Path) -> list[tuple[str, str]]:
    index = json.loads((Path(src_dir) / "model.safetensors.index.json").read_text())
    found: dict[int, tuple[str, str]] = {}
    for key, fname in index["weight_map"].items():
        m = NGRAM_SHARD_RE.search(key)
        if m:
            found[int(m.group(1))] = (key, fname)
    if not found:
        raise RuntimeError(f"{src_dir} names no ngram_embedding shard")
    missing = [i for i in range(max(found) + 1) if i not in found]
    if missing:
        raise RuntimeError(f"missing ngram shard {missing[:5]} of {max(found) + 1}")
    return [found[i] for i in range(max(found) + 1)]


def write_bf16_ngram_table(src_dir: str | Path, out_path: str | Path, *, chunk: int = 64 << 20) -> dict:
    """Concatenate the checkpoint's bf16 n-gram shards into one `ngram_table.bin`."""
    src_dir = Path(src_dir)
    regions, rows, dim = [], 0, None
    for key, fname in bf16_ngram_shards(src_dir):
        header, data_off = read_header(src_dir / fname)
        meta = header[key]
        if meta["dtype"] != "BF16":
            raise RuntimeError(f"{key} is {meta['dtype']}, expected BF16")
        r, d = meta["shape"]
        if dim is None:
            dim = d
        elif d != dim:
            raise RuntimeError(f"{key} row width {d} != {dim}")
        regions.append((src_dir / fname, data_off + meta["data_offsets"][0], r * d * 2))
        rows += r
    nbytes = rows * dim * 2
    header = {
        "__metadata__": {"format": "mlx-serve-ngram", "bits": "16", "group_size": "0"},
        "weight": {"dtype": "BF16", "shape": [rows, dim], "data_offsets": [0, nbytes]},
    }
    hjson = json.dumps(header).encode()
    hjson += b" " * ((8 - (len(hjson) % 8)) % 8)
    tmp = str(out_path) + ".tmp"
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(hjson)))
        out.write(hjson)
        for path, start, length in regions:
            with open(path, "rb") as f:
                f.seek(start)
                left = length
                while left:
                    block = f.read(min(chunk, left))
                    if not block:
                        raise RuntimeError(f"{path}: short read {left} bytes from the end")
                    out.write(block)
                    left -= len(block)
            print(f"ngram {path.name} {length / 1e9:.1f} GB", flush=True)
    os.replace(tmp, out_path)
    return {"rows": rows, "dim": dim, "shards": len(regions), "bytes": 8 + len(hjson) + nbytes}


def ngram_bin_geometry(path: str | Path) -> dict:
    """The `ngram_table` config block for a `.bin` that already exists on disk."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(hlen))["__metadata__"]
    return {"file": "ngram_table.bin", "bits": int(meta["bits"]),
            "group_size": int(meta["group_size"])}


def read_ngram_bin_row(path: str | Path, r: int) -> np.ndarray:
    """One row as the loader's bits-16 arm reads it: raw bf16 at w_off + r * dim * 2."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        meta = header["weight"]
        dim = meta["shape"][1]
        f.seek(8 + hlen + meta["data_offsets"][0] + r * dim * 2)
        return np.frombuffer(f.read(dim * 2), dtype=np.uint16)


def bf16_to_f32(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.uint32) << 16).view(np.float32)


def to_bf16(a: np.ndarray) -> np.ndarray:
    v = np.asarray(a, dtype=np.float32).view(np.uint32)
    return ((v + 0x7FFF + ((v >> 16) & 1)) >> 16).astype(np.uint16)


def within_one_bf16_ulp(a: np.ndarray, b: np.ndarray) -> bool:
    mag = np.maximum(np.abs(a), np.abs(b))
    ulp = np.where(mag > 0, np.exp2(np.floor(np.log2(np.maximum(mag, 1e-30))) - 7), np.float32(1e-30))
    return bool(np.all(np.abs(a - b) <= ulp))


def _read_tensor(root: str | Path, key: str) -> np.ndarray:
    root = Path(root)
    wm = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    header, data_off = read_header(root / wm[key])
    return read_raw(root / wm[key], data_off, header[key])


def resolve_conventions(dense_dir: str | Path, expert_pack: str | Path, plan: dict) -> dict:
    """A tensor NEITHER side quantizes carries no imatrix content, so it comes from the
    expert pack — the copy already in the loader's convention (our converter bakes the
    +1 of the delta-encoded norms into the stored weight; an HF-convention pack does
    not). The dense source's copy must be the same values or that +1; anything else is
    a refusal, never a silent choice."""
    dense_dir, expert_pack = Path(dense_dir), Path(expert_pack)
    expert_index = expert_pack / "model.safetensors.index.json"
    if not expert_index.exists():
        return {"from_expert": [], "refusals": [], "delta_norms": 0}
    expert_wm = json.loads(expert_index.read_text())["weight_map"]
    quantized = {module_of(k) for k in list(plan["dense"]) + list(expert_wm) if k.endswith(".scales")}
    headers: dict[Path, tuple] = {}

    def load(root: Path, fname: str, key: str):
        path = root / fname
        if path not in headers:
            headers[path] = read_header(path)
        header, data_off = headers[path]
        return header[key], read_raw(path, data_off, header[key])

    moved, refusals, deltas = [], [], 0
    for nk, (orig, fname) in sorted(plan["dense"].items()):
        if nk not in expert_wm or module_of(nk) in quantized:
            continue
        meta_d, raw_d = load(dense_dir, fname, orig)
        meta_e, raw_e = load(expert_pack, expert_wm[nk], nk)
        if meta_d["dtype"] != meta_e["dtype"] or list(meta_d["shape"]) != list(meta_e["shape"]):
            refusals.append({"key": nk, "reason":
                             f"{meta_d['dtype']}{list(meta_d['shape'])} in the dense source vs "
                             f"{meta_e['dtype']}{list(meta_e['shape'])} in the pack"})
            continue
        if np.array_equal(raw_d, raw_e):
            moved.append(nk)
            continue
        if meta_d["dtype"] == "BF16":
            d, e = bf16_to_f32(raw_d).ravel(), bf16_to_f32(raw_e).ravel()
            if within_one_bf16_ulp(e, d + np.float32(1.0)):
                moved.append(nk)
                deltas += 1
                continue
            gap = float(np.max(np.abs(e - d)))
        else:
            gap = float("nan")
        refusals.append({"key": nk, "reason":
                         f"the dense copy differs from the pack by neither 0 nor the +1 delta "
                         f"encoding (max|d|={gap:.6g})"})
    return {"from_expert": moved, "refusals": refusals, "delta_norms": deltas}


def module_of(key: str) -> str:
    for suffix in TENSOR_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def normalize_dense_key(key: str) -> str:
    if key.startswith(MTP_SRC_PREFIX):
        return MTP_DST_PREFIX + key[len(MTP_SRC_PREFIX) :]
    return key


def is_ngram_shard_key(key: str) -> bool:
    return ".ngram_embedding.shard" in key


def plan_compose(dense_wm: dict, expert_wm: dict) -> dict:
    """Dense source wins per MODULE; the EXL3 pack keeps the routed experts and
    whatever module the dense source does not carry (the vision tower)."""
    dense: dict[str, tuple[str, str]] = {}
    for key, fname in dense_wm.items():
        if is_ngram_shard_key(key) or is_pack_expert_key(key):
            continue
        nk = normalize_dense_key(key)
        if nk in dense:
            raise RuntimeError(f"{nk} named twice by the dense source")
        dense[nk] = (key, fname)
    # For a module both packs carry, the DENSE copy wins: it is the more precise one, and
    # our fused hyper-connection read declines a quantized `block_inject_weight`.
    quantized_dense = {module_of(k) for k in dense if k.endswith(".scales")}
    quantized_pack = {module_of(k) for k in expert_wm if k.endswith(".scales")}
    pack_modules = {module_of(k) for k in expert_wm}
    pack_wins = (quantized_dense - quantized_pack) & pack_modules
    for key in [k for k in dense if module_of(k) in pack_wins]:
        del dense[key]
    modules = {module_of(k) for k in dense}
    experts, carried = {}, {}
    for key, fname in expert_wm.items():
        if is_pack_expert_key(key):
            experts[key] = fname
        elif module_of(key) not in modules:
            carried[key] = fname
    return {"dense": dense, "experts": experts, "carried": carried}


def affine_admits(*, w_cols: int, s_cols: int, bits: int, gs: int) -> bool:
    """`expert_quant.affineGeomFromShapes`: the geometry our loader solves."""
    if bits not in AFFINE_BITS or gs not in AFFINE_GROUPS:
        return False
    in_dim = s_cols * gs
    return in_dim > 0 and w_cols * 32 == in_dim * bits


def quant_spec_for(quant_cfg: dict, module: str) -> dict:
    spec = {k: v for k, v in quant_cfg.items() if not isinstance(v, dict)}
    override = quant_cfg.get(module)
    if isinstance(override, dict):
        spec.update(override)
    return spec


def dense_refusals(dense_dir: str | Path, plan: dict) -> list[dict]:
    """Every dense module whose quant geometry our loader cannot take, named."""
    dense_dir = Path(dense_dir)
    cfg = json.loads((dense_dir / "config.json").read_text())
    quant_cfg = cfg.get("quantization") or cfg.get("quantization_config") or {}
    by_module: dict[str, dict[str, tuple[str, str]]] = {}
    for nk, (orig, fname) in plan["dense"].items():
        for suffix in (".weight", ".scales", ".biases"):
            if nk.endswith(suffix):
                by_module.setdefault(module_of(nk), {})[suffix[1:]] = (orig, fname)
    headers: dict[str, tuple[dict, int]] = {}
    out = []
    for module, parts in sorted(by_module.items()):
        if "scales" not in parts:
            continue
        if "weight" not in parts:
            out.append({"key": module, "reason": "scales without a weight"})
            continue
        spec = quant_spec_for(quant_cfg, module_of(parts["scales"][0]))
        mode = str(spec.get("mode", "affine"))
        def shape(part):
            orig, fname = parts[part]
            if fname not in headers:
                headers[fname] = read_header(dense_dir / fname)
            return headers[fname][0][orig]["shape"]
        w_shape, s_shape = shape("weight"), shape("scales")
        reason = None
        if mode != "affine":
            reason = f"quant mode {mode}"
        elif "biases" not in parts:
            reason = "affine without biases"
        elif not affine_admits(w_cols=w_shape[-1], s_cols=s_shape[-1],
                               bits=int(spec.get("bits", 0)), gs=int(spec.get("group_size", 0))):
            reason = f"bits {spec.get('bits')} group {spec.get('group_size')} vs weight {w_shape} scales {s_shape}"
        if reason:
            out.append({"key": module, "reason": reason})
    return out


def _rewrite_subset(src_file: Path, out_file: Path, keys: dict[str, str]) -> None:
    """Copy `keys` (source name -> destination name) out of one shard."""
    header, data_off = read_header(src_file)
    named = {}
    for orig, dest in keys.items():
        meta = header[orig]
        raw = read_raw(src_file, data_off, meta)
        named[dest] = (meta["dtype"], tuple(meta["shape"]), np.ascontiguousarray(raw).tobytes())
    write_safetensors_raw(str(out_file), named)


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.link(src, dst)


def compose_pack(
    dense_dir: str | Path,
    expert_pack: str | Path,
    dst: str | Path,
    *,
    ngram_src: str | Path | None = None,
    ngram_bin: str | Path | None = None,
) -> dict:
    """EXL3 experts from `expert_pack`, every other module from `dense_dir`,
    the n-gram table written bf16 from `ngram_src` (or hard-linked from `ngram_bin`)."""
    dense_dir, expert_pack, dst = Path(dense_dir), Path(expert_pack), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    dense_wm = json.loads((dense_dir / "model.safetensors.index.json").read_text())["weight_map"]
    expert_idx = json.loads((expert_pack / "model.safetensors.index.json").read_text())
    plan = plan_compose(dense_wm, expert_idx["weight_map"])
    refusals = dense_refusals(dense_dir, plan)
    conv = resolve_conventions(dense_dir, expert_pack, plan)
    refusals = refusals + conv["refusals"]
    if refusals:
        raise RuntimeError("dense tensors our loader cannot take: " +
                           "; ".join(f"{r['key']} ({r['reason']})" for r in refusals))
    for key in conv["from_expert"]:
        plan["dense"].pop(key)
        plan["carried"][key] = expert_idx["weight_map"][key]
    print(f"from the pack (loader convention): {len(conv['from_expert'])} dense tensors, "
          f"{conv['delta_norms']} of them delta-encoded norms", flush=True)

    weight_map: dict[str, str] = {}
    by_file: dict[str, dict[str, str]] = {}
    for nk, (orig, fname) in plan["dense"].items():
        by_file.setdefault(fname, {})[orig] = nk
    dropped = {}
    for key, fname in dense_wm.items():
        if normalize_dense_key(key) not in plan["dense"]:
            dropped.setdefault(fname, []).append(key)
    for fname, keys in sorted(by_file.items()):
        renamed = any(orig != dest for orig, dest in keys.items())
        if not renamed and fname not in dropped:
            _link(dense_dir / fname, dst / fname)
        else:
            _rewrite_subset(dense_dir / fname, dst / fname, keys)
        for dest in keys.values():
            weight_map[dest] = fname
        print(f"dense {fname} {len(keys)} tensors", flush=True)

    expert_files: dict[str, dict[str, str]] = {}
    for key, fname in {**plan["experts"], **plan["carried"]}.items():
        expert_files.setdefault(fname, {})[key] = key
    for fname, keys in sorted(expert_files.items()):
        if fname in weight_map.values():
            raise RuntimeError(f"{fname} names a shard on both sides")
        whole = sum(1 for k, f in expert_idx["weight_map"].items() if f == fname) == len(keys)
        if whole:
            _link(expert_pack / fname, dst / fname)
        else:
            _rewrite_subset(expert_pack / fname, dst / fname, keys)
        for key in keys:
            weight_map[key] = fname

    for name in sorted(os.listdir(expert_pack)):
        srcp = expert_pack / name
        if not srcp.is_file() or name.endswith(".safetensors") or name in (
            "model.safetensors.index.json", "config.json", "ngram_table.bin"):
            continue
        _link(srcp, dst / name)

    ngram_info: dict = {}
    if ngram_bin is not None:
        _link(Path(ngram_bin), dst / "ngram_table.bin")
        ngram_info = {"source": str(ngram_bin), "linked": True}
    elif ngram_src is not None:
        ngram_info = write_bf16_ngram_table(ngram_src, dst / "ngram_table.bin")
        ngram_info["source"] = str(ngram_src)

    cfg = json.loads((expert_pack / "config.json").read_text())
    dense_cfg = json.loads((dense_dir / "config.json").read_text())
    # The quantization block describes the tensors it came with: dense modules from
    # the dense source, carried modules (the vision tower) from the expert pack.
    dense_modules = {module_of(k) for k in plan["dense"]}
    carried_modules = {module_of(k) for k in plan["carried"]}
    for block in ("quantization", "quantization_config"):
        src_block = dense_cfg.get(block)
        if src_block is None:
            cfg.pop(block, None)
            continue
        out_block = {}
        for k, v in src_block.items():
            if not isinstance(v, dict):
                out_block[k] = v
            elif normalize_dense_key(k) in dense_modules:
                out_block[normalize_dense_key(k)] = v
        for k, v in (cfg.get(block) or {}).items():
            if isinstance(v, dict) and k in carried_modules:
                out_block[k] = v
        # A quantized routed expert is described by the pack it came from, not by the
        # dense source's width (an affine expert pack's 4-bit experts under an 8-bit
        # dense block); an EXL3 pack quantizes no expert and adds nothing here.
        top = {k: v for k, v in out_block.items() if not isinstance(v, dict)}
        for key in plan["experts"]:
            if not key.endswith(".scales"):
                continue
            spec = quant_spec_for(cfg.get(block) or {}, module_of(key))
            if spec != top:
                out_block[module_of(key)] = spec
        cfg[block] = out_block
    cfg["ngram_table"] = ngram_bin_geometry(dst / "ngram_table.bin")
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    total = sum(os.path.getsize(dst / f) for f in sorted(set(weight_map.values())))
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    return {"weight_map": weight_map, "refusals": refusals, "ngram": ngram_info,
            "carried": sorted(plan["carried"]), "bytes": total,
            "from_expert": conv["from_expert"], "delta_norms": conv["delta_norms"]}


def expert_quant_block(source: str, **extra) -> dict:
    """The `expert_quant` block the server admits: exl3, K, one of mul1|tiny|mcg, and
    the codeword `window` every decoder must mask to (absent reads as 16)."""
    block = {"format": "exl3", "k": int(K), "codebook": CODEBOOK, "window": WINDOW_DEFAULT,
             "out_scales": "svh", "source": source}
    block.update(extra)
    return block


def convert_pack(
    hf_dir: str | Path,
    pack_dir: str | Path,
    dst: str | Path,
    *,
    quantizer: str = "direct",
    k=K,
    codebook: str = CODEBOOK,
    window: int = WINDOW_DEFAULT,
    calibration: np.ndarray | None = None,
    imatrix: dict[str, np.ndarray] | None = None,
    imatrix_sha: str | None = None,
    batch_size: int = BATCH_EXPERTS_DEFAULT,
    scratch_gb: float = SCRATCH_GB_DEFAULT,
    g_scale: bool = True,
    quality: bool = True,
    resume: bool = True,
) -> dict:
    hf_dir = Path(hf_dir)
    pack_dir = Path(pack_dir)
    dst = Path(dst)
    window = validate_window(window, k)
    reset_search_stats()
    dst.mkdir(parents=True, exist_ok=True)
    pack_index = json.loads((pack_dir / "model.safetensors.index.json").read_text())
    plan = plan_pack(pack_index)
    for fname in plan["hardlink"]:
        src = pack_dir / fname
        out = dst / fname
        if out.exists() or out.is_symlink():
            out.unlink()
        os.link(src, out)
    for name in sorted(os.listdir(pack_dir)):
        if name.startswith("."):
            continue
        if name.startswith("model-") and name.endswith(".safetensors"):
            continue
        if name in ("model.safetensors.index.json", "config.json"):
            continue
        src = pack_dir / name
        if not src.is_file():
            continue
        out = dst / name
        if out.exists() or out.is_symlink():
            out.unlink()
        os.link(src, out)
    for i, fname in enumerate(plan["rewrite"]):
        other_keys = plan["files"][fname]["other"]
        tensors = _copy_raw_tensors(pack_dir / fname, other_keys)
        write_safetensors_raw(str(dst / fname), tensors)
    hf_index_path = hf_dir / "model.safetensors.index.json"
    if hf_index_path.exists():
        hf_map = json.loads(hf_index_path.read_text())["weight_map"]
    else:
        keys = list(read_header(hf_dir / "model.safetensors")[0])
        hf_map = {key: "model.safetensors" for key in keys}
    weight_map = {key: pack_index["weight_map"][key] for key in plan["keep_keys"]}
    for key, pack_name in list(weight_map.items()):
        if pack_name in plan["drop"]:
            raise RuntimeError(f"keep key {key} pointed at dropped shard {pack_name}")
    zero_routed: list[str] = []
    skipped = 0
    rewritten: list[str] = []
    werr: dict[str, list[float]] = {}
    import mlx.core as mx
    scratch_bytes = max(1, int(scratch_gb * (1 << 30)))
    # The search's buffer pool is no longer dropped between launches, so cap it instead.
    mx.set_cache_limit(4 * scratch_bytes)
    cal_tag = ("imatrix-diagonal" if imatrix is not None else
               "captured-rows" if calibration is not None else
               "ldlq-gaussian-256" if quantizer == "ldlq" else "none-direct")
    stamp = shard_stamp(k=k, codebook=codebook, window=window, quantizer=quantizer,
                        imatrix_sha=imatrix_sha, g_scale="gss" if g_scale else "one")

    def emit(layer: int, proj: str, base: str, bank, lo: int, hi: int,
             imat_flat, rows_vec, lkey: str, mtp: bool):
        nonlocal skipped
        e, in_dim, out_dim = bank.shape[0], bank.shape[2], hi - lo
        shard = layer_proj_shard(layer, proj, mtp=mtp)
        dest = dst / shard
        if resume:
            refusal = shard_reuse_refusal(dest, e, in_dim, out_dim, k, stamp)
            if refusal is None:
                skipped += 1
                for suffix in (".trellis", ".suh", ".svh"):
                    weight_map[base + suffix] = shard
                print(f"skip {shard}", flush=True)
                return
            if refusal != "absent":
                rewritten.append(f"{shard}: {refusal}")
                print(f"rewrite {shard}: {refusal}", flush=True)
        t0 = time.perf_counter()
        before = search_stats_snapshot()
        trellis = np.empty((e, in_dim // 16, out_dim // 16, packed_hw(k)), dtype=np.uint16)
        suh = np.empty((e, in_dim), dtype=np.float16)
        svh = np.empty((e, out_dim), dtype=np.float16)
        cals = bank_calibrations(e, in_dim, calibration=calibration, imatrix_flat=imat_flat,
                                 routed_rows=rows_vec, zero_routed=zero_routed, layer_key=lkey)
        scales: list[float] = []
        errs: list[float] = []
        fallbacks = 0

        def load_batch(start: int, stop: int):
            """Read, dequantize, regularize and LDL-factor one batch — every host stage
            there is. Runs on the prefetch thread while the GPU searches the previous."""
            publics = [expert_public(bank, ei, lo, hi) for ei in range(start, stop)]
            seeds = [expert_seed(layer, ei, proj, mtp=mtp) for ei in range(start, stop)]
            return (prepare_expert_bank(publics, seeds, cals[start:stop], quantizer=quantizer),
                    publics, cals[start:stop])

        step = max(1, int(batch_size))
        spans = [(s, min(e, s + step)) for s in range(0, e, step)]
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="qwen4-exl3-read") as pool:
            for (start, stop), (prep, publics, bcals) in zip(
                    spans, prefetch_batches(pool, load_batch, spans)):
                bt, bsuh, bsvh, bf = quantize_prepared_bank(
                    prep, publics, bcals, k=k, codebook=codebook, window=window,
                    scratch_bytes=scratch_bytes, g_scale=g_scale, scale_out=scales,
                    err_out=errs if quality else None)
                del prep, publics
                trellis[start:stop] = bt
                suh[start:stop] = bsuh
                svh[start:stop] = bsvh
                fallbacks += bf
                del bt, bsuh, bsvh
        named = {
            base + ".trellis": ("U16", trellis.shape, trellis.tobytes()),
            base + ".suh": ("F16", suh.shape, suh.tobytes()),
            base + ".svh": ("F16", svh.shape, svh.tobytes()),
        }
        write_safetensors_raw(str(dest), named, metadata=stamp)
        for key in named:
            weight_map[key] = shard
        if errs:
            werr[shard] = list(errs)
        dt = time.perf_counter() - t0
        after = search_stats_snapshot()
        gpu = after["seconds"] - before["seconds"]
        print(f"wrote {shard}  {dt:.1f}s  e={e} in={in_dim} out={out_dim} "
              f"prior_fallback={fallbacks}  g[{min(scales, default=1.0):.2f},"
              f"{max(scales, default=1.0):.2f}]  werr {sum(errs) / max(len(errs), 1):.5f}"
              f"  search {gpu:.1f}s ({100.0 * gpu / max(dt, 1e-9):.1f}% of wall, "
              f"{(after['tiles'] - before['tiles']) / max(gpu, 1e-9):.0f} tiles/s)", flush=True)
        del trellis, suh, svh, named

    def read_bank(hf_file: str, hf_key: str):
        header, data_off = read_header(hf_dir / hf_file)
        return read_raw(hf_dir / hf_file, data_off, header[hf_key])

    for hf_key, hf_file in hf_map.items():
        mtp = hf_key.startswith(MTP_SRC_PREFIX)
        if hf_key.endswith("experts.gate_up_proj"):
            arr = read_bank(hf_file, hf_key)
            half = arr.shape[1] // 2
            gu_flat = None if imatrix is None else imatrix.get(hf_key)
            gu_rows = None if imatrix is None else imatrix.get(hf_key + ".rows")
            layer = parse_layer_from_hf_key(hf_key)
            for proj, lo in (("gate", 0), ("up", half)):
                base = mlx_switch_base(hf_key, proj)
                emit(layer, proj, base, arr, lo, lo + half, gu_flat, gu_rows,
                     hf_key + "." + proj, mtp)
            del arr
        elif hf_key.endswith("experts.down_proj"):
            arr = read_bank(hf_file, hf_key)
            base = mlx_switch_base(hf_key, "down")
            parent = hf_key.replace("experts.down_proj", "experts.gate_up_proj")
            dn_flat = None if imatrix is None else imatrix.get(hf_key)
            gu_rows = None if imatrix is None else imatrix.get(parent + ".rows")
            layer = parse_layer_from_hf_key(hf_key)
            emit(layer, "down", base, arr, 0, arr.shape[1], dn_flat, gu_rows, hf_key, mtp)
            del arr
    total = 0
    for fname in sorted(set(weight_map.values())):
        total += os.path.getsize(dst / fname)
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2
    ))
    cfg = json.loads((pack_dir / "config.json").read_text())
    cfg["expert_quant"] = expert_quant_block(
        "convert", k=k, codebook=codebook, window=int(window), quantizer=stamp["quantizer"],
        calibration=cal_tag, imatrix_sha256=stamp["imatrix_sha256"], converter=stamp["converter"])
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    plan["zero_routed"] = zero_routed
    plan["calibration"] = cal_tag
    plan["skipped"] = skipped
    plan["rewritten"] = rewritten
    plan["stamp"] = stamp
    plan["weighted_rel_err"] = werr
    plan["search"] = search_stats_snapshot()
    return plan


def layer_proj_shard(layer: int, proj: str, *, mtp: bool = False) -> str:
    """The MTP head carries its OWN `layers.0` MoE bank at the trunk's exact geometry,
    so the two need different file names or the second silently adopts the first."""
    return f"model-exl3-{'mtp-' if mtp else ''}L{layer:02d}-{proj}.safetensors"


def parse_layer_from_hf_key(hf_key: str) -> int:
    import re
    m = re.search(r"\.layers\.(\d+)\.", hf_key)
    if not m:
        raise ValueError(hf_key)
    return int(m.group(1))


def shard_is_valid(path: str | Path, n_experts: int, in_dim: int, out_dim: int, k,
                   stamp: dict[str, str] | None = None) -> bool:
    """Geometry alone cannot tell a K3/w8/LDLQ shard from a K3/w16/direct one; a resume
    that adopts by shape ships a pack whose layers were quantized different ways."""
    return shard_reuse_refusal(path, n_experts, in_dim, out_dim, k, stamp) is None


def imatrix_layer_keys(layer: int) -> tuple[str, str, str]:
    p = f"model.language_model.layers.{layer}.mlp.experts."
    return p + "gate_up_proj", p + "down_proj", p + "gate_up_proj.rows"


def imatrix_layer_complete(store: dict[str, np.ndarray], layer: int) -> bool:
    a, b, c = imatrix_layer_keys(layer)
    return a in store and b in store and c in store


SRC_EXPERT_RE = __import__("re").compile(
    r"^(model\.language_model\.|mtp\.)layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(trellis|suh|svh|mul1)$"
)


def dest_switch_key(src_prefix: str, layer: int, proj: str, suffix: str) -> str:
    if src_prefix.startswith("mtp"):
        return f"language_model.mtp.layers.{layer}.mlp.switch_mlp.{proj}_proj.{suffix}"
    return f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}_proj.{suffix}"


def parse_src_expert_key(key: str):
    m = SRC_EXPERT_RE.match(key)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)


class RestackTests(unittest.TestCase):
    def test_src_expert_key_parse(self):
        p = parse_src_expert_key("model.language_model.layers.3.mlp.experts.9.gate_proj.trellis")
        self.assertEqual(p[1:], (3, 9, "gate", "trellis"))
        self.assertEqual(
            dest_switch_key(p[0], 3, "gate", "trellis"),
            "language_model.model.layers.3.mlp.switch_mlp.gate_proj.trellis",
        )
        p = parse_src_expert_key("mtp.layers.0.mlp.experts.1.down_proj.mul1")
        self.assertEqual(
            dest_switch_key(p[0], 0, "down", "svh"),
            "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.svh",
        )

    def _write_case(self, td, *, odd_expert=None):
        rng = np.random.default_rng(4)
        src, pack, dst = td / "src", td / "pack", td / "out"
        src.mkdir(); pack.mkdir()
        e, h, i = 2, 128, 256
        if True:
            tensors = {}
            for ei in range(e):
                for proj, inn, outn in (("gate", h, i), ("up", h, i), ("down", i, h)):
                    base = f"model.language_model.layers.0.mlp.experts.{ei}.{proj}_proj"
                    trellis = rng.integers(0, 65535, (inn // 16, outn // 16, PACKED_K4), dtype=np.uint16)
                    tshape = trellis.shape
                    suh_dtype = "F16"
                    if odd_expert is not None and ei == odd_expert[0] and proj == "gate":
                        if odd_expert[1] == "shape":
                            tshape = (outn // 16, inn // 16, PACKED_K4)
                        else:
                            suh_dtype = "BF16"
                    tensors[base + ".trellis"] = ("I16", tshape, trellis.tobytes())
                    tensors[base + ".suh"] = (suh_dtype, (inn,), np.zeros(inn, np.float16).tobytes())
                    tensors[base + ".svh"] = ("F16", (outn,), np.ones(outn, np.float16).tobytes())
                    tensors[base + ".mul1"] = ("I32", (), np.int32(1).tobytes())
            write_safetensors_raw(str(src / "model-00001-of-00001.safetensors"), tensors)
            (src / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}
            }))
            (src / "config.json").write_text(json.dumps({
                "quantization_config": {"codebook": "mul1", "bits": 4.05, "out_scales": "always"},
                "text_config": {"num_experts": e, "hidden_size": h, "moe_intermediate_size": i, "num_hidden_layers": 1},
            }))
            write_safetensors_raw(str(pack / "model-00001.safetensors"), {
                "language_model.model.embed_tokens.weight": ("F32", (4,), np.zeros(4, np.float32).tobytes()),
            })
            dummy = np.zeros((e, 8), np.uint32)
            write_safetensors_raw(str(pack / "model-00002.safetensors"), {
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": ("U32", dummy.shape, dummy.tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales": ("F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases": ("F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": ("U32", dummy.shape, dummy.tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.scales": ("F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.biases": ("F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
            })
            write_safetensors_raw(str(pack / "model-00003.safetensors"), {
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": ("U32", dummy.shape, dummy.tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales": ("F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.biases": ("F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.gate.weight": ("F32", (e, h), np.zeros((e, h), np.float32).tobytes()),
            })
            (pack / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 1},
                "weight_map": {
                    "language_model.model.embed_tokens.weight": "model-00001.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": ("model-00002.safetensors"),
                    "language_model.model.layers.0.mlp.switch_mlp.up_proj.scales": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.up_proj.biases": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": "model-00003.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales": "model-00003.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.down_proj.biases": "model-00003.safetensors",
                    "language_model.model.layers.0.mlp.gate.weight": "model-00003.safetensors",
                },
            }))
            (pack / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
            (pack / "tokenizer.json").write_text("{}")
        return src, pack, dst, e, h, i

    def test_restack_two_experts_into_stacked_banks(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src, pack, dst, e, h, i = self._write_case(td)
            restack_from_exl3(src, pack, dst)
            cfg = json.loads((dst / "config.json").read_text())
            self.assertEqual(cfg["expert_quant"]["codebook"], "mul1")
            self.assertEqual(cfg["expert_quant"]["k"], 4)
            idx = json.loads((dst / "model.safetensors.index.json").read_text())
            wm = idx["weight_map"]
            gkey = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis"
            self.assertIn(gkey, wm)
            header, _ = read_header(dst / wm[gkey])
            self.assertEqual(tuple(header[gkey]["shape"]), (e, h // 16, i // 16, PACKED_K4))
            src_header, src_off = read_header(src / "model-00001-of-00001.safetensors")
            src_t = np.frombuffer(
                _raw_bytes(src / "model-00001-of-00001.safetensors", src_header,
                           "model.language_model.layers.0.mlp.experts.1.gate_proj.trellis", src_off),
                dtype=np.uint16,
            ).reshape((h // 16, i // 16, PACKED_K4))
            stacked_h, stacked_off = read_header(dst / wm[gkey])
            st_meta = stacked_h[gkey]
            stacked = np.frombuffer(
                _raw_bytes(dst / wm[gkey], stacked_h, gkey, stacked_off), dtype=np.uint16
            ).reshape(st_meta["shape"])
            np.testing.assert_array_equal(stacked[1], src_t)
            self.assertEqual(
                os.stat(dst / "model-00001.safetensors").st_ino,
                os.stat(pack / "model-00001.safetensors").st_ino,
            )

    def test_an_expert_whose_own_header_disagrees_is_refused(self):
        # Every expert is read with expert 0's shape and dtype; a source whose
        # element count matches but whose header does not must not restack.
        for odd in (("shape",), ("dtype",)):
            with tempfile.TemporaryDirectory() as td:
                src, pack, dst, _, _, _ = self._write_case(Path(td), odd_expert=(1, odd[0]))
                with self.assertRaises(RuntimeError):
                    restack_from_exl3(src, pack, dst)

    def test_restack_mixed_k_per_tensor_keeps_each_last_dim(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src, pack, dst = td / "src", td / "pack", td / "out"
            src.mkdir(); pack.mkdir()
            e, h, i = 2, 128, 128
            packed = {"gate": packed_hw(3), "up": packed_hw(3), "down": packed_hw(2)}
            tensors = {}
            for layer in range(2):
                for ei in range(e):
                    for proj, inn, outn in (("gate", h, i), ("up", h, i), ("down", i, h)):
                        base = f"model.language_model.layers.{layer}.mlp.experts.{ei}.{proj}_proj"
                        trellis = rng.integers(0, 65535, (inn // 16, outn // 16, packed[proj]), dtype=np.uint16)
                        tensors[base + ".trellis"] = ("I16", trellis.shape, trellis.tobytes())
                        tensors[base + ".suh"] = ("F16", (inn,), np.zeros(inn, np.float16).tobytes())
                        tensors[base + ".svh"] = ("F16", (outn,), np.ones(outn, np.float16).tobytes())
                        tensors[base + ".mul1"] = ("I32", (), np.int32(1).tobytes())
            write_safetensors_raw(str(src / "model-00001-of-00001.safetensors"), tensors)
            (src / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}
            }))
            (src / "config.json").write_text(json.dumps({
                "quantization_config": {"codebook": "mul1", "bits": 3.05, "out_scales": "always"},
                "text_config": {"num_experts": e, "hidden_size": h, "moe_intermediate_size": i, "num_hidden_layers": 2},
            }))
            write_safetensors_raw(str(pack / "model-00001.safetensors"), {
                "language_model.model.embed_tokens.weight": ("F32", (4,), np.zeros(4, np.float32).tobytes()),
            })
            dummy = np.zeros((e, 8), np.uint32)
            pack_tensors = {}
            wm = {"language_model.model.embed_tokens.weight": "model-00001.safetensors"}
            for layer in range(2):
                for proj in ("gate", "up", "down"):
                    for part, dt, sh, arr in (
                        ("weight", "U32", dummy.shape, dummy),
                        ("scales", "F16", (e, 2), np.zeros((e, 2), np.float16)),
                        ("biases", "F16", (e, 2), np.zeros((e, 2), np.float16)),
                    ):
                        key = f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}_proj.{part}"
                        pack_tensors[key] = (dt, sh, arr.tobytes())
                        wm[key] = "model-00002.safetensors"
            write_safetensors_raw(str(pack / "model-00002.safetensors"), pack_tensors)
            (pack / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": wm}))
            (pack / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
            (pack / "tokenizer.json").write_text("{}")
            restack_from_exl3(src, pack, dst)
            cfg = json.loads((dst / "config.json").read_text())
            self.assertEqual(cfg["expert_quant"]["codebook"], "mul1")
            self.assertEqual(cfg["expert_quant"]["k"], 3)
            idx = json.loads((dst / "model.safetensors.index.json").read_text())
            want = {"gate": packed["gate"], "up": packed["up"], "down": packed["down"]}
            for layer in range(2):
                for proj, inn, outn in (("gate", h, i), ("up", h, i), ("down", i, h)):
                    key = f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}_proj.trellis"
                    header, _ = read_header(dst / idx["weight_map"][key])
                    self.assertEqual(tuple(header[key]["shape"]), (e, inn // 16, outn // 16, want[proj]))


def _write_resume_fixture(hf: Path, pack: Path, rng, *, with_mtp: bool = False) -> None:
    hf.mkdir(); pack.mkdir()
    e, hidden, inter = 2, 128, 128
    prefixes = ["model.language_model."] + ([MTP_SRC_PREFIX] if with_mtp else [])
    tensors, hf_map = {}, {}
    for prefix in prefixes:
        gu = f"{prefix}layers.0.mlp.experts.gate_up_proj"
        dn = f"{prefix}layers.0.mlp.experts.down_proj"
        tensors[gu] = ("F32", (e, 2 * inter, hidden),
                       rng.standard_normal((e, 2 * inter, hidden), dtype=np.float32).tobytes())
        tensors[dn] = ("F32", (e, hidden, inter),
                       rng.standard_normal((e, hidden, inter), dtype=np.float32).tobytes())
        hf_map[gu] = hf_map[dn] = "model.safetensors"
    write_safetensors_raw(str(hf / "model.safetensors"), tensors)
    (hf / "config.json").write_text("{}")
    (hf / "model.safetensors.index.json").write_text(json.dumps({"weight_map": hf_map}))
    write_safetensors_raw(str(pack / "model-00001.safetensors"), {
        "language_model.model.embed_tokens.weight": ("F32", (4,), np.zeros(4, np.float32).tobytes()),
    })
    dummy = np.zeros((e, 8), dtype=np.uint32)
    zeros_e2 = np.zeros((e, 2), np.float16)
    write_safetensors_raw(str(pack / "model-00002.safetensors"), {
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": ("U32", dummy.shape, dummy.tobytes()),
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales": ("F16", (e, 2), zeros_e2.tobytes()),
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases": ("F16", (e, 2), zeros_e2.tobytes()),
        "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": ("U32", dummy.shape, dummy.tobytes()),
        "language_model.model.layers.0.mlp.switch_mlp.up_proj.scales": ("F16", (e, 2), zeros_e2.tobytes()),
        "language_model.model.layers.0.mlp.switch_mlp.up_proj.biases": ("F16", (e, 2), zeros_e2.tobytes()),
    })
    write_safetensors_raw(str(pack / "model-00003.safetensors"), {
        "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": ("U32", dummy.shape, dummy.tobytes()),
        "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales": ("F16", (e, 2), zeros_e2.tobytes()),
        "language_model.model.layers.0.mlp.switch_mlp.down_proj.biases": ("F16", (e, 2), zeros_e2.tobytes()),
        "language_model.model.layers.0.mlp.gate.weight": ("F32", (e, hidden), np.zeros((e, hidden), np.float32).tobytes()),
    })
    (pack / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 1},
        "weight_map": {
            "language_model.model.embed_tokens.weight": "model-00001.safetensors",
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": "model-00002.safetensors",
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales": "model-00002.safetensors",
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases": "model-00002.safetensors",
            "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": "model-00002.safetensors",
            "language_model.model.layers.0.mlp.switch_mlp.up_proj.scales": "model-00002.safetensors",
            "language_model.model.layers.0.mlp.switch_mlp.up_proj.biases": "model-00002.safetensors",
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": "model-00003.safetensors",
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales": "model-00003.safetensors",
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.biases": "model-00003.safetensors",
            "language_model.model.layers.0.mlp.gate.weight": "model-00003.safetensors",
        },
    }))
    (pack / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    (pack / "tokenizer.json").write_text("{}")


class ResumeTests(unittest.TestCase):
    def test_layer_projection_shard_names_are_stable(self):
        self.assertEqual(layer_proj_shard(0, "gate"), "model-exl3-L00-gate.safetensors")
        self.assertEqual(layer_proj_shard(47, "down"), "model-exl3-L47-down.safetensors")

    def test_existing_valid_shard_is_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / layer_proj_shard(3, "up")
            e, h, i = 2, 128, 128
            trellis = np.zeros((e, h // 16, i // 16, PACKED_K4), dtype=np.uint16)
            write_safetensors_raw(str(p), {
                "language_model.model.layers.3.mlp.switch_mlp.up_proj.trellis": (
                    "U16", trellis.shape, trellis.tobytes()),
                "language_model.model.layers.3.mlp.switch_mlp.up_proj.suh": (
                    "F16", (e, h), np.zeros((e, h), np.float16).tobytes()),
                "language_model.model.layers.3.mlp.switch_mlp.up_proj.svh": (
                    "F16", (e, i), np.zeros((e, i), np.float16).tobytes()),
            })
            self.assertTrue(shard_is_valid(p, e, h, i, 4))
            self.assertFalse(shard_is_valid(p, e, 256, i, 4))
            self.assertFalse(shard_is_valid(p, e, h, i, 3))
            self.assertFalse(shard_is_valid(Path(td) / "missing.safetensors", e, h, i, 4))

    def test_second_convert_skips_existing_shards(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            hf, pack, dst = td / "hf", td / "pack", td / "out"
            _write_resume_fixture(hf, pack, rng)
            convert_pack(hf, pack, dst, quantizer="direct")
            mtimes = {p.name: p.stat().st_mtime_ns for p in dst.glob("model-exl3-L00-*.safetensors")}
            self.assertEqual(len(mtimes), 3)
            plan = convert_pack(hf, pack, dst, quantizer="direct")
            self.assertEqual(plan["skipped"], 3)
            for p in dst.glob("model-exl3-L00-*.safetensors"):
                self.assertEqual(p.stat().st_mtime_ns, mtimes[p.name])

    def test_tiny_k3_conversion_names_its_codebook(self):
        rng = np.random.default_rng(13)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            hf, pack, dst = td / "hf", td / "pack", td / "out"
            _write_resume_fixture(hf, pack, rng)
            convert_pack(hf, pack, dst, quantizer="direct", k=3, codebook="tiny")
            cfg = json.loads((dst / "config.json").read_text())
            self.assertEqual(cfg["expert_quant"]["codebook"], "tiny")
            self.assertEqual(cfg["expert_quant"]["k"], 3)
            with self.assertRaises(RuntimeError):
                codebook_mode("mul2")

    def test_resume_at_another_k_rewrites_the_shards(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            hf, pack, dst = td / "hf", td / "pack", td / "out"
            _write_resume_fixture(hf, pack, rng)
            convert_pack(hf, pack, dst, quantizer="direct", k=4)
            plan = convert_pack(hf, pack, dst, quantizer="direct", k=3)
            self.assertEqual(plan["skipped"], 0)
            for proj in ("gate", "up", "down"):
                p = dst / layer_proj_shard(0, proj)
                header, _ = read_header(p)
                key = f"language_model.model.layers.0.mlp.switch_mlp.{proj}_proj.trellis"
                self.assertEqual(header[key]["shape"][-1], packed_hw(3))

    def test_imatrix_layer_complete_is_the_three_expert_keys(self):
        store = {
            "model.language_model.layers.2.mlp.experts.gate_up_proj": np.zeros(4),
            "model.language_model.layers.2.mlp.experts.down_proj": np.zeros(4),
        }
        self.assertFalse(imatrix_layer_complete(store, 2))
        store["model.language_model.layers.2.mlp.experts.gate_up_proj.rows"] = np.zeros(2)
        self.assertTrue(imatrix_layer_complete(store, 2))
        self.assertFalse(imatrix_layer_complete(store, 1))


class CalibTests(unittest.TestCase):
    def test_channel_moments_become_a_diagonal_hessian(self):
        v = np.array([0.5, 2.0, 0.0, 1.25], dtype=np.float32)
        h = diag_hessian(v)
        self.assertEqual(h.shape, (4, 4))
        np.testing.assert_array_equal(np.diag(h), v)
        self.assertEqual(float(h[0, 1]), 0.0)

    def test_zero_routed_tokens_fall_back_to_gaussian(self):
        mode, vec = calib_for_expert(0, np.ones(4, dtype=np.float32))
        self.assertEqual(mode, "ldlq-gaussian-256")
        self.assertIsNone(vec)
        mode, vec = calib_for_expert(3, np.array([1.0, 2.0], dtype=np.float32))
        self.assertEqual(mode, "imatrix-diagonal")
        np.testing.assert_array_equal(vec, np.array([1.0, 2.0], dtype=np.float32))

    def test_convert_records_imatrix_diagonal_and_zero_routed(self):
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            hf, pack, dst = td / "hf", td / "pack", td / "out"
            hf.mkdir(); pack.mkdir()
            e, hidden, inter = 2, 128, 128
            write_safetensors_raw(str(hf / "model.safetensors"), {
                "model.language_model.layers.0.mlp.experts.gate_up_proj": (
                    "F32", (e, 2 * inter, hidden), rng.standard_normal((e, 2 * inter, hidden), dtype=np.float32).tobytes()),
                "model.language_model.layers.0.mlp.experts.down_proj": (
                    "F32", (e, hidden, inter), rng.standard_normal((e, hidden, inter), dtype=np.float32).tobytes()),
            })
            (hf / "config.json").write_text("{}")
            (hf / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {
                    "model.language_model.layers.0.mlp.experts.gate_up_proj": "model.safetensors",
                    "model.language_model.layers.0.mlp.experts.down_proj": "model.safetensors",
                }
            }))
            write_safetensors_raw(str(pack / "model-00001.safetensors"), {
                "language_model.model.embed_tokens.weight": ("F32", (4,), np.zeros(4, np.float32).tobytes()),
            })
            dummy = np.zeros((e, 8), dtype=np.uint32)
            write_safetensors_raw(str(pack / "model-00002.safetensors"), {
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": ("U32", dummy.shape, dummy.tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales": ("F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases": ("F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": ("U32", dummy.shape, dummy.tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.scales": ("F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.biases": ("F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
            })
            write_safetensors_raw(str(pack / "model-00003.safetensors"), {
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": ("U32", dummy.shape, dummy.tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales": ("F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.biases": ("F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.gate.weight": ("F32", (e, hidden), np.zeros((e, hidden), np.float32).tobytes()),
            })
            (pack / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 1},
                "weight_map": {
                    "language_model.model.embed_tokens.weight": "model-00001.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.up_proj.scales": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.up_proj.biases": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": "model-00003.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales": "model-00003.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.down_proj.biases": "model-00003.safetensors",
                    "language_model.model.layers.0.mlp.gate.weight": "model-00003.safetensors",
                },
            }))
            (pack / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
            (pack / "tokenizer.json").write_text("{}")
            imat = {
                "model.language_model.layers.0.mlp.experts.gate_up_proj": np.ones(e * hidden, dtype=np.float32),
                "model.language_model.layers.0.mlp.experts.down_proj": np.ones(e * inter, dtype=np.float32),
                "model.language_model.layers.0.mlp.experts.gate_up_proj.rows": np.array([4.0, 0.0], dtype=np.float32),
            }
            plan = convert_pack(hf, pack, dst, quantizer="direct", imatrix=imat)
            cfg = json.loads((dst / "config.json").read_text())
            self.assertEqual(cfg["expert_quant"]["calibration"], "imatrix-diagonal")
            self.assertEqual(plan["calibration"], "imatrix-diagonal")
            self.assertTrue(any("#1" in z for z in plan["zero_routed"]))

    def test_gate_up_and_down_split_the_concatenated_imatrix(self):
        e, h, i = 2, 4, 8
        gu = np.arange(e * h, dtype=np.float32)
        dn = np.arange(e * i, dtype=np.float32) + 10
        self.assertEqual(tuple(imatrix_expert_vector(gu, 1, h)), (4.0, 5.0, 6.0, 7.0))
        self.assertEqual(tuple(imatrix_expert_vector(dn, 0, i)), tuple(np.arange(8, dtype=np.float32) + 10))


class PlanTests(unittest.TestCase):
    def test_pack_expert_keys_are_the_switch_mlp_banks(self):
        self.assertTrue(is_pack_expert_key(
            "language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight"))
        self.assertTrue(is_pack_expert_key(
            "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.scales"))
        self.assertFalse(is_pack_expert_key(
            "language_model.model.layers.3.mlp.shared_expert.down_proj.weight"))
        self.assertFalse(is_pack_expert_key(
            "language_model.model.layers.3.mlp.gate.weight"))

    def test_stacked_dialect_names_three_tensors_per_projection(self):
        trellis, suh, svh = exl3_keys(
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj")
        self.assertEqual(trellis, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis")
        self.assertEqual(suh, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.suh")
        self.assertEqual(svh, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.svh")

    def test_plan_classifies_hardlink_rewrite_and_drop(self):
        idx = {
            "weight_map": {
                "language_model.model.embed_tokens.weight": "model-00001.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": "model-00002.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales": "model-00002.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases": "model-00002.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": "model-00002.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.scales": "model-00002.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.biases": "model-00002.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": "model-00003.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales": "model-00003.safetensors",
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.biases": "model-00003.safetensors",
                "language_model.model.layers.0.mlp.gate.weight": "model-00003.safetensors",
            }
        }
        plan = plan_pack(idx)
        self.assertEqual(plan["hardlink"], ["model-00001.safetensors"])
        self.assertEqual(plan["drop"], ["model-00002.safetensors"])
        self.assertEqual(plan["rewrite"], ["model-00003.safetensors"])
        self.assertIn("language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis", plan["exl3_keys"])
        self.assertNotIn("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight", plan["keep_keys"])
        self.assertIn("language_model.model.layers.0.mlp.gate.weight", plan["keep_keys"])


class ExpertQuantBlockTests(unittest.TestCase):
    def test_both_writers_emit_the_block_the_server_admits(self):
        for source in ("restack", "convert"):
            block = expert_quant_block(source)
            self.assertEqual(block["format"], "exl3")
            self.assertEqual(block["k"], 4)
            self.assertEqual(block["codebook"], "mul1")
            self.assertNotIn("mcg_multiplier", block)

    def test_convert_extras_ride_beside_the_admitted_keys(self):
        block = expert_quant_block("convert", quantizer="ldlq", calibration="imatrix-diagonal")
        self.assertEqual(block["codebook"], "mul1")
        self.assertEqual(block["quantizer"], "ldlq")
        self.assertNotIn("mcg_multiplier", block)


class LayoutTests(unittest.TestCase):
    def test_synthetic_pack_layout_and_hardlinks(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            hf = td / "hf"
            pack = td / "pack"
            dst = td / "out"
            hf.mkdir()
            pack.mkdir()
            e, hidden, inter = 2, 128, 128
            gate_up = rng.standard_normal((e, 2 * inter, hidden)).astype(np.float32)
            down = rng.standard_normal((e, hidden, inter)).astype(np.float32)
            write_safetensors_raw(str(hf / "model.safetensors"), {
                "model.language_model.layers.0.mlp.experts.gate_up_proj": (
                    "F32", gate_up.shape, np.ascontiguousarray(gate_up).tobytes()),
                "model.language_model.layers.0.mlp.experts.down_proj": (
                    "F32", down.shape, np.ascontiguousarray(down).tobytes()),
            })
            (hf / "config.json").write_text(json.dumps({
                "model_type": "qwen4_exp",
                "text_config": {"num_hidden_layers": 1, "hidden_size": hidden,
                                "moe_intermediate_size": inter, "num_experts": e},
            }))
            (hf / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {
                    "model.language_model.layers.0.mlp.experts.gate_up_proj": "model.safetensors",
                    "model.language_model.layers.0.mlp.experts.down_proj": "model.safetensors",
                }
            }))
            other = rng.standard_normal((4,)).astype(np.float32)
            write_safetensors_raw(str(pack / "model-00001.safetensors"), {
                "language_model.model.embed_tokens.weight": (
                    "F32", other.shape, np.ascontiguousarray(other).tobytes()),
            })
            dummy = np.zeros((e, 8), dtype=np.uint32)
            write_safetensors_raw(str(pack / "model-00002.safetensors"), {
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": (
                    "U32", dummy.shape, dummy.tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales": (
                    "F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases": (
                    "F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": (
                    "U32", dummy.shape, dummy.tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.scales": (
                    "F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.up_proj.biases": (
                    "F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
            })
            router = rng.standard_normal((e, hidden)).astype(np.float32)
            write_safetensors_raw(str(pack / "model-00003.safetensors"), {
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": (
                    "U32", dummy.shape, dummy.tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales": (
                    "F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.switch_mlp.down_proj.biases": (
                    "F16", (e, 2), np.zeros((e, 2), np.float16).tobytes()),
                "language_model.model.layers.0.mlp.gate.weight": (
                    "F32", router.shape, np.ascontiguousarray(router).tobytes()),
            })
            (pack / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 1},
                "weight_map": {
                    "language_model.model.embed_tokens.weight": "model-00001.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.up_proj.scales": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.up_proj.biases": "model-00002.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": "model-00003.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales": "model-00003.safetensors",
                    "language_model.model.layers.0.mlp.switch_mlp.down_proj.biases": "model-00003.safetensors",
                    "language_model.model.layers.0.mlp.gate.weight": "model-00003.safetensors",
                },
            }))
            (pack / "config.json").write_text(json.dumps({
                "model_type": "qwen4_exp",
                "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
                "text_config": {"num_hidden_layers": 1, "hidden_size": hidden,
                                "moe_intermediate_size": inter, "num_experts": e},
            }))
            (pack / "tokenizer.json").write_text("{}")
            convert_pack(hf, pack, dst, quantizer="direct")
            cfg = json.loads((dst / "config.json").read_text())
            self.assertEqual(cfg["expert_quant"]["format"], "exl3")
            self.assertEqual(cfg["expert_quant"]["k"], 4)
            self.assertEqual(cfg["expert_quant"]["codebook"], "mul1")
            self.assertEqual(
                os.stat(dst / "model-00001.safetensors").st_ino,
                os.stat(pack / "model-00001.safetensors").st_ino,
            )
            self.assertEqual(
                os.stat(dst / "tokenizer.json").st_ino,
                os.stat(pack / "tokenizer.json").st_ino,
            )
            self.assertFalse((dst / "model-00002.safetensors").exists())
            idx = json.loads((dst / "model.safetensors.index.json").read_text())
            wm = idx["weight_map"]
            for proj in ("gate", "up", "down"):
                base = f"language_model.model.layers.0.mlp.switch_mlp.{proj}_proj"
                for suffix in (".trellis", ".suh", ".svh"):
                    self.assertIn(base + suffix, wm)
                self.assertNotIn(base + ".weight", wm)
            self.assertIn("language_model.model.layers.0.mlp.gate.weight", wm)
            self.assertEqual(wm["language_model.model.embed_tokens.weight"], "model-00001.safetensors")
            expert_file = wm["language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis"]
            header, _data_off = read_header(dst / expert_file)
            self.assertEqual(
                tuple(header["language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis"]["shape"]),
                (e, hidden // 16, inter // 16, PACKED_K4),
            )
            mix_file = wm["language_model.model.layers.0.mlp.gate.weight"]
            mix_header, mix_off = read_header(dst / mix_file)
            self.assertNotIn("language_model.model.layers.0.mlp.switch_mlp.down_proj.weight", mix_header)
            got = read_raw(dst / mix_file, mix_off, mix_header["language_model.model.layers.0.mlp.gate.weight"])
            np.testing.assert_array_equal(got, router)


class Bf16NgramTableTests(unittest.TestCase):
    """The bf16 `.bin` is the shards concatenated: one row must survive the trip."""

    def _shards(self, td, rows=(3, 5), dim=4):
        rng = np.random.default_rng(7)
        src = Path(td)
        wm = {}
        blocks = []
        for i, r in enumerate(rows):
            bits = rng.integers(0, 1 << 16, size=(r, dim), dtype=np.uint16)
            blocks.append(bits)
            key = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight"
            fname = f"model-{i:05d}.safetensors"
            write_safetensors_raw(str(src / fname), {key: ("BF16", (r, dim), bits.tobytes())})
            wm[key] = fname
        (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
        return np.concatenate(blocks, axis=0)

    def test_every_row_including_the_shard_boundary_is_bit_exact(self):
        with tempfile.TemporaryDirectory() as td:
            want = self._shards(td)
            out = Path(td) / "ngram_table.bin"
            info = write_bf16_ngram_table(td, out)
            self.assertEqual(info["rows"], want.shape[0])
            self.assertEqual(info["dim"], want.shape[1])
            # rows 2 and 3 straddle the shard-0/shard-1 boundary.
            for r in range(want.shape[0]):
                np.testing.assert_array_equal(read_ngram_bin_row(out, r), want[r])

    def test_the_header_is_the_bits_16_contract_the_loader_parses(self):
        with tempfile.TemporaryDirectory() as td:
            self._shards(td)
            out = Path(td) / "ngram_table.bin"
            write_bf16_ngram_table(td, out)
            with open(out, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(hlen))
            self.assertEqual(header["__metadata__"]["format"], "mlx-serve-ngram")
            self.assertEqual(header["__metadata__"]["bits"], "16")
            self.assertEqual(header["weight"]["dtype"], "BF16")
            self.assertEqual(header["weight"]["data_offsets"][0], 0)

    def test_a_missing_shard_index_is_named_not_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            self._shards(td)
            idx = Path(td) / "model.safetensors.index.json"
            wm = json.loads(idx.read_text())["weight_map"]
            wm.pop("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight")
            idx.write_text(json.dumps({"weight_map": wm}))
            with self.assertRaises(RuntimeError) as cm:
                write_bf16_ngram_table(td, Path(td) / "x.bin")
            self.assertIn("shard", str(cm.exception))


class ComposeDenseTests(unittest.TestCase):
    """Dense source wins per MODULE; routed experts come from the EXL3 pack."""

    def test_the_mtp_prefix_is_normalized_to_the_one_the_loader_reads(self):
        self.assertEqual(normalize_dense_key("mtp.fc_hidden.weight"), "language_model.mtp.fc_hidden.weight")
        self.assertEqual(normalize_dense_key("language_model.lm_head.weight"), "language_model.lm_head.weight")

    def test_plan_takes_dense_from_the_dense_source_and_experts_from_the_pack(self):
        dense = {
            "language_model.lm_head.weight": "d1", "language_model.lm_head.scales": "d1",
            "language_model.model.layers.0.mlp.gate.weight": "d1",
            "mtp.fc_hidden.weight": "d2",
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": "d2",
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight": "d3",
        }
        expert = {
            "language_model.lm_head.weight": "p1", "language_model.lm_head.scales": "p1",
            "language_model.model.layers.0.mlp.gate.weight": "p1",
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis": "p2",
            "model.visual.patch_embed.proj.weight": "p3",
        }
        plan = plan_compose(dense, expert)
        self.assertEqual(plan["dense"]["language_model.mtp.fc_hidden.weight"], ("mtp.fc_hidden.weight", "d2"))
        self.assertNotIn("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight", plan["dense"])
        self.assertNotIn("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight", plan["dense"])
        self.assertEqual(sorted(plan["experts"]), ["language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis"])
        self.assertEqual(sorted(plan["carried"]), ["model.visual.patch_embed.proj.weight"])

    def test_a_quantized_module_the_dense_source_carries_replaces_the_pack_whole(self):
        # ours quantizes `gate` (weight+scales+biases), the dense source keeps it dense:
        # the module comes from one side or the other, never half from each.
        dense = {"language_model.model.layers.0.mlp.gate.weight": "d1"}
        expert = {
            "language_model.model.layers.0.mlp.gate.weight": "p1",
            "language_model.model.layers.0.mlp.gate.scales": "p1",
            "language_model.model.layers.0.mlp.gate.biases": "p1",
        }
        plan = plan_compose(dense, expert)
        self.assertEqual(plan["carried"], {})
        self.assertEqual(sorted(plan["dense"]), ["language_model.model.layers.0.mlp.gate.weight"])

    def test_a_geometry_the_loader_cannot_take_is_a_named_refusal(self):
        ok = affine_admits(w_cols=640, s_cols=40, bits=8, gs=64)          # in 2560, 8-bit gs64
        self.assertTrue(ok)
        self.assertTrue(affine_admits(w_cols=320, s_cols=40, bits=4, gs=64))
        self.assertFalse(affine_admits(w_cols=640, s_cols=40, bits=7, gs=64))   # width mx.quantize never ships
        self.assertFalse(affine_admits(w_cols=640, s_cols=160, bits=8, gs=16))  # group size the solver rejects
        self.assertFalse(affine_admits(w_cols=641, s_cols=40, bits=8, gs=64))   # packed cols do not solve

    def test_a_scales_tensor_with_no_weight_is_named_too(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td)
            sc = np.zeros((4, 1), dtype=np.uint16)
            write_safetensors_raw(str(src / "m.safetensors"), {
                "a.scales": ("BF16", sc.shape, sc.tobytes()),
                "a.biases": ("BF16", sc.shape, sc.tobytes()),
            })
            wm = {k: "m.safetensors" for k in ("a.scales", "a.biases")}
            (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
            (src / "config.json").write_text(json.dumps({"quantization": {"bits": 8, "group_size": 64}}))
            refusals = dense_refusals(src, plan_compose(wm, {}))
            self.assertEqual([r["key"] for r in refusals], ["a"])
            self.assertIn("weight", refusals[0]["reason"])

    def test_refusals_name_the_tensor_and_its_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td)
            good = np.zeros((4, 2), dtype=np.uint32)
            bad = np.zeros((4, 3), dtype=np.uint32)
            sc = np.zeros((4, 1), dtype=np.uint16)
            write_safetensors_raw(str(src / "m.safetensors"), {
                "a.weight": ("U32", good.shape, good.tobytes()),
                "a.scales": ("BF16", sc.shape, sc.tobytes()),
                "a.biases": ("BF16", sc.shape, sc.tobytes()),
                "b.weight": ("U32", bad.shape, bad.tobytes()),
                "b.scales": ("BF16", sc.shape, sc.tobytes()),
                "b.biases": ("BF16", sc.shape, sc.tobytes()),
            })
            wm = {k: "m.safetensors" for k in ("a.weight", "a.scales", "a.biases", "b.weight", "b.scales", "b.biases")}
            (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
            (src / "config.json").write_text(json.dumps({"quantization": {"bits": 2, "group_size": 32, "mode": "affine"}}))
            plan = plan_compose(wm, {})
            bad_names = [r["key"] for r in dense_refusals(src, plan)]
            self.assertEqual(bad_names, ["b"])


class ConventionTests(unittest.TestCase):
    """A tensor neither side quantizes comes from the EXPERT pack, because that copy is
    already in the loader's convention: our converter bakes the +1 of the delta-encoded
    norms into the stored weight, an HF-convention pack does not."""

    def _pair(self, td, ours, theirs, name="n.weight"):
        root = Path(td)
        a, b = root / "a", root / "b"
        for d, arr in ((a, ours), (b, theirs)):
            d.mkdir()
            write_safetensors_raw(str(d / "m.safetensors"), {name: ("BF16", arr.shape, arr.tobytes())})
            (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "m.safetensors"}}))
        (b / "config.json").write_text(json.dumps({"quantization": {"bits": 8, "group_size": 64}}))
        wm = {name: "m.safetensors"}
        return b, a, plan_compose(wm, wm)

    def test_an_identical_dense_tensor_comes_from_the_expert_pack(self):
        v = to_bf16(np.array([0.5, -2.0, 7.625], dtype=np.float32))
        with tempfile.TemporaryDirectory() as td:
            dense_dir, pack, plan = self._pair(td, v, v)
            out = resolve_conventions(dense_dir, pack, plan)
            self.assertEqual(out["refusals"], [])
            self.assertEqual(out["from_expert"], ["n.weight"])
            self.assertEqual(out["delta_norms"], 0)

    def test_the_plus_one_delta_encoding_is_recognized_not_refused(self):
        base = to_bf16(np.array([-0.0635, -5.9375, 6.625], dtype=np.float32))
        ours = to_bf16(bf16_to_f32(base) + 1.0)
        with tempfile.TemporaryDirectory() as td:
            dense_dir, pack, plan = self._pair(td, ours, base)
            out = resolve_conventions(dense_dir, pack, plan)
            self.assertEqual(out["refusals"], [])
            self.assertEqual(out["from_expert"], ["n.weight"])
            self.assertEqual(out["delta_norms"], 1)

    def test_any_other_disagreement_is_a_named_refusal(self):
        base = to_bf16(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        ours = to_bf16(np.array([1.0, 2.0, 3.5], dtype=np.float32))
        with tempfile.TemporaryDirectory() as td:
            dense_dir, pack, plan = self._pair(td, ours, base)
            out = resolve_conventions(dense_dir, pack, plan)
            self.assertEqual([r["key"] for r in out["refusals"]], ["n.weight"])
            self.assertIn("neither", out["refusals"][0]["reason"])

    def test_a_tensor_only_the_dense_source_has_stays_with_the_dense_source(self):
        v = to_bf16(np.array([1.0, 2.0], dtype=np.float32))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            d = root / "d"; d.mkdir()
            write_safetensors_raw(str(d / "m.safetensors"), {"n.weight": ("BF16", v.shape, v.tobytes())})
            (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"n.weight": "m.safetensors"}}))
            (d / "config.json").write_text(json.dumps({"quantization": {"bits": 8, "group_size": 64}}))
            pack = root / "p"; pack.mkdir()
            out = resolve_conventions(d, pack, plan_compose({"n.weight": "m.safetensors"}, {}))
            self.assertEqual(out["from_expert"], [])
            self.assertEqual(out["refusals"], [])

    def test_compose_puts_the_pack_norm_in_the_output(self):
        u32 = np.zeros((4, 4), dtype=np.uint32).tobytes()
        bfz = np.zeros((4, 1), dtype=np.uint16).tobytes()
        base = to_bf16(np.array([-0.0635, -5.9375], dtype=np.float32))
        ours = to_bf16(bf16_to_f32(base) + 1.0)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dense, pack, dst = root / "dense", root / "pack", root / "out"
            for d, norm, shard in ((dense, base, "d.safetensors"), (pack, ours, "p.safetensors")):
                d.mkdir()
                write_safetensors_raw(str(d / shard), {
                    "language_model.lm_head.weight": ("U32", (4, 4), u32),
                    "language_model.lm_head.scales": ("BF16", (4, 1), bfz),
                    "language_model.lm_head.biases": ("BF16", (4, 1), bfz),
                    "language_model.model.layers.0.q_norm.weight": ("BF16", norm.shape, norm.tobytes()),
                })
                wm = {k: shard for k in ("language_model.lm_head.weight", "language_model.lm_head.scales",
                                         "language_model.lm_head.biases",
                                         "language_model.model.layers.0.q_norm.weight")}
                (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
            (dense / "config.json").write_text(json.dumps({"quantization": {"bits": 4, "group_size": 32, "mode": "affine"}}))
            (pack / "config.json").write_text(json.dumps({"expert_quant": expert_quant_block("restack", k=4)}))
            ng = root / "ng"; ng.mkdir()
            bits = np.arange(4, dtype=np.uint16).reshape(1, 4)
            key = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
            write_safetensors_raw(str(ng / "s.safetensors"), {key: ("BF16", (1, 4), bits.tobytes())})
            (ng / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "s.safetensors"}}))
            out = compose_pack(dense, pack, dst, ngram_src=ng)
            self.assertEqual(out["delta_norms"], 1)
            got = _read_tensor(dst, "language_model.model.layers.0.q_norm.weight")
            np.testing.assert_array_equal(got, ours)


class DensePrecedenceTests(unittest.TestCase):
    """For a module both packs carry, the DENSE copy wins: it is the more precise one,
    and our fused hyper-connection read declines a quantized `block_inject_weight`
    (`inject_flat` is set only for a dense inject), which costs 44% of decode."""

    def test_the_pack_keeps_a_module_only_the_dense_source_quantizes(self):
        dense = {"l.0.attn_hyper_connection.block_inject_weight." + x: "d"
                 for x in ("weight", "scales", "biases")}
        expert = {"l.0.attn_hyper_connection.block_inject_weight.weight": "p"}
        plan = plan_compose(dense, expert)
        self.assertEqual(sorted(plan["carried"]), ["l.0.attn_hyper_connection.block_inject_weight.weight"])
        self.assertEqual(plan["dense"], {})

    def test_the_dense_source_keeps_a_module_only_the_pack_quantizes(self):
        dense = {"l.0.mlp.gate.weight": "d"}
        expert = {"l.0.mlp.gate." + x: "p" for x in ("weight", "scales", "biases")}
        plan = plan_compose(dense, expert)
        self.assertEqual(sorted(plan["dense"]), ["l.0.mlp.gate.weight"])
        self.assertEqual(plan["carried"], {})

    def test_both_quantized_still_comes_from_the_dense_source(self):
        dense = {"l.0.o_proj." + x: "d" for x in ("weight", "scales", "biases")}
        expert = {"l.0.o_proj." + x: "p" for x in ("weight", "scales", "biases")}
        plan = plan_compose(dense, expert)
        self.assertEqual(len(plan["dense"]), 3)
        self.assertEqual(plan["carried"], {})


class ComposeLayoutTests(unittest.TestCase):
    def _pack(self, root: Path, keys: dict, cfg: dict) -> None:
        root.mkdir(parents=True, exist_ok=True)
        wm = {}
        by_file: dict[str, dict] = {}
        for key, (fname, dtype, shape, raw) in keys.items():
            by_file.setdefault(fname, {})[key] = (dtype, shape, raw)
            wm[key] = fname
        for fname, named in by_file.items():
            write_safetensors_raw(str(root / fname), named)
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
        (root / "config.json").write_text(json.dumps(cfg))

    def test_compose_writes_one_pack_from_two(self):
        u32 = np.zeros((4, 4), dtype=np.uint32).tobytes()  # 4-bit gs32: in 32, packed 4 u32
        bf = np.zeros((4, 1), dtype=np.uint16).tobytes()
        tre = np.zeros((2, 4, 4), dtype=np.uint16).tobytes()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dense, pack, dst = root / "dense", root / "pack", root / "out"
            self._pack(dense, {
                "language_model.lm_head.weight": ("d-00001.safetensors", "U32", (4, 4), u32),
                "language_model.lm_head.scales": ("d-00001.safetensors", "BF16", (4, 1), bf),
                "language_model.lm_head.biases": ("d-00001.safetensors", "BF16", (4, 1), bf),
                "mtp.fc_hidden.weight": ("d-00002.safetensors", "U32", (4, 4), u32),
                "mtp.fc_hidden.scales": ("d-00002.safetensors", "BF16", (4, 1), bf),
                "mtp.fc_hidden.biases": ("d-00002.safetensors", "BF16", (4, 1), bf),
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight":
                    ("d-00002.safetensors", "U32", (4, 4), u32),
            }, {"quantization": {"bits": 4, "group_size": 32, "mode": "affine",
                                 "mtp.fc_hidden": {"bits": 4, "group_size": 32, "mode": "affine"},
                                 "language_model.model.layers.0.mlp.switch_mlp.gate_proj":
                                     {"bits": 4, "group_size": 32, "mode": "affine"}}})
            self._pack(pack, {
                "language_model.lm_head.weight": ("p-00001.safetensors", "U32", (4, 4), u32),
                "language_model.lm_head.scales": ("p-00001.safetensors", "BF16", (4, 1), bf),
                "language_model.lm_head.biases": ("p-00001.safetensors", "BF16", (4, 1), bf),
                "model.visual.patch_embed.proj.weight": ("p-00001.safetensors", "BF16", (4, 1), bf),
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis":
                    ("p-exl3.safetensors", "U16", (2, 4, 4), tre),
            }, {"expert_quant": expert_quant_block("restack", k=4), "text_config": {"x": 1},
                "ngram_table": {"file": "ngram_table.bin", "bits": 4, "group_size": 32}})
            (pack / "tokenizer.json").write_text("{}")
            ng = root / "ngram_src"
            ng.mkdir()
            bits = np.arange(8, dtype=np.uint16).reshape(2, 4)
            key = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
            write_safetensors_raw(str(ng / "s.safetensors"), {key: ("BF16", (2, 4), bits.tobytes())})
            (ng / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "s.safetensors"}}))

            out = compose_pack(dense, pack, dst, ngram_src=ng)
            wm = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
            self.assertEqual(wm["language_model.lm_head.weight"], "d-00001.safetensors")
            self.assertEqual(wm["language_model.mtp.fc_hidden.weight"], "d-00002.safetensors")
            self.assertEqual(wm["language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis"],
                             "p-exl3.safetensors")
            self.assertEqual(wm["model.visual.patch_embed.proj.weight"], "p-00001.safetensors")
            self.assertNotIn("mtp.fc_hidden.weight", wm)
            self.assertNotIn("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight", wm)
            # the unchanged dense shard is a hard link, the renamed one a rewrite
            self.assertEqual(os.stat(dst / "d-00001.safetensors").st_ino,
                             os.stat(dense / "d-00001.safetensors").st_ino)
            self.assertNotEqual(os.stat(dst / "d-00002.safetensors").st_ino,
                                os.stat(dense / "d-00002.safetensors").st_ino)
            self.assertEqual(os.stat(dst / "p-exl3.safetensors").st_ino,
                             os.stat(pack / "p-exl3.safetensors").st_ino)
            self.assertTrue((dst / "tokenizer.json").exists())
            cfg = json.loads((dst / "config.json").read_text())
            self.assertEqual(cfg["expert_quant"], expert_quant_block("restack", k=4))
            self.assertEqual(cfg["text_config"], {"x": 1})
            self.assertEqual(cfg["ngram_table"], {"file": "ngram_table.bin", "bits": 16, "group_size": 0})
            self.assertEqual(cfg["quantization"]["bits"], 4)
            self.assertIn("language_model.mtp.fc_hidden", cfg["quantization"])
            self.assertNotIn("mtp.fc_hidden", cfg["quantization"])
            self.assertNotIn("language_model.model.layers.0.mlp.switch_mlp.gate_proj", cfg["quantization"])
            self.assertEqual(out["carried"], ["model.visual.patch_embed.proj.weight"])
            np.testing.assert_array_equal(read_ngram_bin_row(dst / "ngram_table.bin", 1), bits[1])


class AffineExpertPackTests(unittest.TestCase):
    """ddalcu's affine 4/8 pack: routed experts are affine `switch_mlp` tensors, and
    the pack ships its own quantized n-gram table."""

    def _write_bin(self, path: Path, *, bits: str, gs: str) -> None:
        rows, dim = 2, 32
        wcols, scols = dim * int(bits) // 32, dim // int(gs)
        w = np.zeros((rows, wcols), dtype=np.uint32).tobytes()
        sc = np.zeros((rows, scols), dtype=np.uint16).tobytes()
        header = {"__metadata__": {"format": "mlx-serve-ngram", "bits": bits, "group_size": gs},
                  "weight": {"dtype": "U32", "shape": [rows, wcols], "data_offsets": [0, len(w)]},
                  "scales": {"dtype": "BF16", "shape": [rows, scols],
                             "data_offsets": [len(w), len(w) + len(sc)]},
                  "biases": {"dtype": "BF16", "shape": [rows, scols],
                             "data_offsets": [len(w) + len(sc), len(w) + 2 * len(sc)]}}
        hjson = json.dumps(header).encode()
        hjson += b" " * ((8 - (len(hjson) % 8)) % 8)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hjson)))
            f.write(hjson)
            f.write(w)
            f.write(sc)
            f.write(sc)

    def _compose(self, td: str, *, ngram_bits: str = "4", ngram_gs: str = "32") -> dict:
        root = Path(td)
        dense, pack, dst = root / "dense", root / "pack", root / "out"
        u32_8 = np.zeros((4, 16), dtype=np.uint32).tobytes()   # 8-bit gs64: in 64
        u32_4 = np.zeros((4, 8), dtype=np.uint32).tobytes()    # 4-bit gs64: in 64
        bf = np.zeros((4, 1), dtype=np.uint16).tobytes()
        expert = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
        layout = ComposeLayoutTests()
        layout._pack(dense, {
            "language_model.lm_head.weight": ("d-00001.safetensors", "U32", (4, 16), u32_8),
            "language_model.lm_head.scales": ("d-00001.safetensors", "BF16", (4, 1), bf),
            "language_model.lm_head.biases": ("d-00001.safetensors", "BF16", (4, 1), bf),
        }, {"quantization": {"bits": 8, "group_size": 64, "mode": "affine"}})
        layout._pack(pack, {
            "language_model.lm_head.weight": ("p-00001.safetensors", "U32", (4, 16), u32_8),
            "language_model.lm_head.scales": ("p-00001.safetensors", "BF16", (4, 1), bf),
            "language_model.lm_head.biases": ("p-00001.safetensors", "BF16", (4, 1), bf),
            expert + ".weight": ("p-00002.safetensors", "U32", (4, 8), u32_4),
            expert + ".scales": ("p-00002.safetensors", "BF16", (4, 1), bf),
            expert + ".biases": ("p-00002.safetensors", "BF16", (4, 1), bf),
        }, {"quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
            "text_config": {"x": 1},
            "ngram_table": {"file": "ngram_table.bin", "bits": 4, "group_size": 32}})
        binp = root / "table.bin"
        self._write_bin(binp, bits=ngram_bits, gs=ngram_gs)
        compose_pack(dense, pack, dst, ngram_bin=binp)
        return json.loads((dst / "config.json").read_text())

    def test_the_ngram_block_describes_the_table_that_landed(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self._compose(td, ngram_bits="4", ngram_gs="32")
        self.assertEqual(cfg["ngram_table"],
                         {"file": "ngram_table.bin", "bits": 4, "group_size": 32})

    def test_the_routed_experts_keep_the_packs_own_width(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self._compose(td)
        expert = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
        self.assertEqual(cfg["quantization"]["bits"], 8)
        self.assertEqual(cfg["quantization"][expert],
                         {"bits": 4, "group_size": 64, "mode": "affine"})


class ComponentOutputTests(unittest.TestCase):
    @staticmethod
    def _write_component_source(root: Path) -> Path:
        root.mkdir()
        tensors = {
            "language_model.model.embed_tokens.weight":
                ("U8", (5,), b"embed"),
            "language_model.lm_head.weight":
                ("U8", (4,), b"head"),
            "language_model.model.hyper_connection_mixer.weight":
                ("U8", (3,), b"mix"),
            "language_model.mtp.layers.0.norm.weight":
                ("U8", (3,), b"mtp"),
            "model.visual.weight":
                ("U8", (6,), b"vision"),
        }
        for layer in range(4):
            tensors[f"language_model.model.layers.{layer}.norm.weight"] = (
                "U8", (4,), b"norm"
            )
            for proj in ("gate", "up", "down"):
                for suffix in ("trellis", "suh", "svh"):
                    key = (
                        f"language_model.model.layers.{layer}."
                        f"mlp.switch_mlp.{proj}_proj.{suffix}"
                    )
                    tensors[key] = ("U8", (6,), b"expert")
        tensors["language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.trellis"] = (
            "U8", (10,), b"mtp-expert"
        )
        shard = "source.safetensors"
        write_safetensors_raw(str(root / shard), tensors)
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": sum(len(raw) for _, _, raw in tensors.values())},
            "weight_map": {key: shard for key in tensors},
        }))
        (root / "config.json").write_text(json.dumps({
            "model_type": "qwen4_exp",
            "expert_quant": {"format": "exl3", "k": 3},
            "ngram_table": {"file": "ngram_table.bin", "bits": 16},
        }))
        (root / "ngram_table.bin").write_bytes(b"ngram table bytes")
        (root / "tokenizer.json").write_text("{}")
        (root / "config.json.orig-262k").write_text("original config")
        return root

    @staticmethod
    def _tensor_signature(root: Path, key: str) -> tuple:
        index = json.loads((root / "model.safetensors.index.json").read_text())
        filename = index["weight_map"][key]
        header, data_off = read_header(root / filename)
        meta = header[key]
        raw = read_raw(root / filename, data_off, meta)
        return meta["dtype"], tuple(meta["shape"]), np.ascontiguousarray(raw).tobytes()

    def test_real_component_repack_preserves_bytes_and_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._write_component_source(root / "staged")
            destination = root / "components"
            repack_component_output(source, destination)
            self.assertEqual(
                {path.name for path in destination.glob("*.safetensors")},
                {
                    "model-embed.safetensors",
                    "model-lm-head.safetensors",
                    "model-trunk-00001-of-00002.safetensors",
                    "model-trunk-00002-of-00002.safetensors",
                    *(f"model-experts-L{layer:02}.safetensors" for layer in range(4)),
                    "model-mtp.safetensors",
                    "model-vision.safetensors",
                },
            )
            source_index = json.loads(
                (source / "model.safetensors.index.json").read_text()
            )
            destination_index = json.loads(
                (destination / "model.safetensors.index.json").read_text()
            )
            self.assertEqual(source_index["weight_map"].keys(),
                             destination_index["weight_map"].keys())
            for key in source_index["weight_map"]:
                self.assertEqual(
                    self._tensor_signature(source, key),
                    self._tensor_signature(destination, key),
                    key,
                )
            self.assertEqual(
                (destination / "ngram_table.bin").read_bytes(),
                b"ngram table bytes",
            )
            self.assertEqual(
                (destination / "config.json.orig-262k").read_text(),
                "original config",
            )

    def test_component_repack_forwards_new_destination_and_share_pack(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "staged"
            destination = root / "components"
            share = root / "existing-components"
            source.mkdir()
            share.mkdir()
            repacker = Mock()
            with patch.object(sys.modules[__name__], "_load_component_repacker",
                              return_value=repacker):
                repack_component_output(source, destination, share)
            repacker.repack.assert_called_once_with(source, destination, share)

    def test_component_repack_refuses_to_reuse_destination(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "staged"
            destination = root / "components"
            source.mkdir()
            destination.mkdir()
            with self.assertRaisesRegex(RuntimeError, "destination must not already exist"):
                repack_component_output(source, destination)

    def test_cli_rejects_existing_component_output_before_staging(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            staged = root / "staged"
            destination = root / "components"
            staged.mkdir()
            destination.mkdir()
            with patch.object(sys, "argv", [
                "convert_qwen38_flash_next_exl3.py",
                "--hf", "hf", "--pack", "pack", "--dst", str(staged),
                "--component-output", str(destination),
            ]), patch.object(sys.modules[__name__], "convert_pack") as convert:
                with self.assertRaises(SystemExit) as cm:
                    main()
            self.assertEqual(cm.exception.code, 2)
            convert.assert_not_called()

    def test_normal_convert_repacks_after_staging(self):
        with patch.object(sys, "argv", [
            "convert_qwen38_flash_next_exl3.py",
            "--hf", "hf", "--pack", "pack", "--dst", "staged",
            "--component-output", "components", "--share-with", "existing",
        ]), patch.object(sys.modules[__name__], "convert_pack") as convert, \
                patch.object(sys.modules[__name__], "repack_component_output") as repack:
            convert.return_value = {"search": search_stats_snapshot(), "skipped": 0,
                                    "rewritten": [], "weighted_rel_err": {}}
            self.assertEqual(main(), 0)
        convert.assert_called_once_with(
            "hf", "pack", "staged", quantizer="direct", k=4, codebook="mul1",
            window=WINDOW_DEFAULT, calibration=None, imatrix=None, imatrix_sha=None,
            batch_size=BATCH_EXPERTS_DEFAULT, scratch_gb=SCRATCH_GB_DEFAULT,
            g_scale=True, quality=True, resume=True,
        )
        repack.assert_called_once_with("staged", "components", "existing")

    def test_restack_repacks_after_staging(self):
        with patch.object(sys, "argv", [
            "convert_qwen38_flash_next_exl3.py",
            "--from-exl3", "source", "--pack", "pack", "--dst", "staged",
            "--component-output", "components",
        ]), patch.object(sys.modules[__name__], "restack_from_exl3") as restack, \
                patch.object(sys.modules[__name__], "repack_component_output") as repack:
            self.assertEqual(main(), 0)
        restack.assert_called_once_with("source", "pack", "staged")
        repack.assert_called_once_with("staged", "components", None)

    def test_dense_compose_repacks_after_staging(self):
        composed = {"weight_map": {}, "bytes": 0, "carried": []}
        with patch.object(sys, "argv", [
            "convert_qwen38_flash_next_exl3.py",
            "--dense", "dense", "--pack", "pack", "--dst", "staged",
            "--ngram-bin", "ngram.bin", "--component-output", "components",
        ]), patch.object(sys.modules[__name__], "compose_pack",
                         return_value=composed) as compose, \
                patch.object(sys.modules[__name__], "repack_component_output") as repack:
            self.assertEqual(main(), 0)
        compose.assert_called_once_with(
            "dense", "pack", "staged", ngram_src=None, ngram_bin="ngram.bin",
        )
        repack.assert_called_once_with("staged", "components", None)

    def test_share_with_requires_component_output(self):
        with patch.object(sys, "argv", [
            "convert_qwen38_flash_next_exl3.py",
            "--hf", "hf", "--pack", "pack", "--dst", "staged",
            "--share-with", "existing",
        ]):
            with self.assertRaises(SystemExit) as cm:
                main()
        self.assertEqual(cm.exception.code, 2)


# The rotated-Hessian / global-scale / window battery, at this checkpoint's own geometry:
# a gate or up expert is public [hidden, inter] = [2560, 640] and a down expert is its
# transpose, so both axes are multiples of the 128-point Hadamard block. The synthetic
# shapes below keep that 4:1 aspect at a size the GPU search finishes in milliseconds.

SYN_ROWS = 2 * HAD_BLOCK
SYN_COLS = HAD_BLOCK // 2


def _regularized(rows=SYN_ROWS, cols=4 * SYN_COLS, seed=41):
    _ensure_lib()
    from ponyexl3.convert.regularize import regularize_public_weight
    rng = np.random.default_rng(seed)
    return regularize_public_weight(rng.standard_normal((rows, cols), dtype=np.float32), seed=1)


class RotatedCalibrationTests(unittest.TestCase):
    """The imatrix diagonal is per PUBLIC input channel; the search runs on the inner
    matrix. Handing LDLQ the unrotated diagonal silently discards the calibration."""

    def test_the_rotated_hessian_is_what_ponyexl3_builds_from_the_same_statistic(self):
        _ensure_lib()
        from ponyexl3.convert.hessian import capture_hessian, public_activations_to_inner
        rng = np.random.default_rng(31)
        h = np.abs(rng.standard_normal(SYN_ROWS, dtype=np.float32)) + 0.05
        suh = rng.standard_normal(SYN_ROWS, dtype=np.float32)
        want = capture_hessian(public_activations_to_inner(np.diag(np.sqrt(h)).astype(np.float32), suh),
                               normalize=False)
        got = _common.inner_hessian_blocks(h, suh)
        self.assertEqual(got.shape, (SYN_ROWS // HAD_BLOCK, HAD_BLOCK, HAD_BLOCK))
        for b in range(got.shape[0]):
            lo = b * HAD_BLOCK
            self.assertTrue(np.allclose(got[b], want[lo:lo + HAD_BLOCK, lo:lo + HAD_BLOCK],
                                        rtol=1e-4, atol=1e-4), msg=f"block {b}")

    def test_the_unrotated_public_diagonal_would_have_had_no_feedback_at_all(self):
        """Red-on-revert for the basis bug this converter shipped: `diag_hessian(v)` ->
        `prepare_hessian_for_ldl` -> `block_ldl` factors to exactly the identity, so the
        old `--quantizer ldlq` was a direct search wearing a calibration's name."""
        rng = np.random.default_rng(33)
        v = np.abs(rng.standard_normal(HAD_BLOCK, dtype=np.float32)) + 0.05
        self.assertTrue(_common.ldl_is_feedbackless(_common.ldl_factor(diag_hessian(v))))
        blocks = _common.inner_hessian_blocks(v, _regularized(rows=HAD_BLOCK).suh)
        self.assertFalse(_common.ldl_is_feedbackless(_common.inner_ldl_blocks(blocks)))

    def test_rotated_feedback_beats_the_direct_search_on_the_calibrated_objective(self):
        import mlx.core as mx
        from ponyexl3.ref.reconstruct import reconstruct_inner
        rng = np.random.default_rng(35)
        regs, blocks, diags = [], [], []
        for i in range(4):
            reg = _regularized(rows=HAD_BLOCK, cols=4 * HAD_BLOCK, seed=200 + i)
            h = np.abs(rng.standard_normal(HAD_BLOCK, dtype=np.float32)) + 0.05
            regs.append(reg)
            diags.append(h)
            blocks.append(_common.imatrix_ldl_blocks(h, reg.suh))
        cb = codebook_mode(CODEBOOK)
        fed = _common.ldlq_group_mlx(
            mx.array(np.stack([r.inner for r in regs]), dtype=mx.float32),
            mx.array(np.stack(blocks), dtype=mx.float32),
            k=3, cb=cb, window=WINDOW_DEFAULT, scratch_bytes=1 << 29)
        better = 0
        for i, reg in enumerate(regs):
            plain = _common.quantize_inner_direct(reg.inner, k=3, cb=cb,
                                                  window=WINDOW_DEFAULT, chunk=4096)[0]
            hb = _common.inner_hessian_blocks(diags[i], reg.suh)[0]

            def weighted(packed):
                d = np.asarray(reconstruct_inner(packed, 3, mul1=True), np.float32) - reg.inner
                return float(np.sum(d * (hb @ d)))
            better += weighted(fed[i]) < weighted(plain)
        self.assertGreaterEqual(better, 3, "feedback did not help on 3 of 4 experts")

    def test_the_calibration_reaches_regularize_public_weight(self):
        """`regularize_public_weight(hessian_diag=...)` is what keeps a cancelling gate's
        suppressed outputs negative; the converter used to call it without one."""
        _ensure_lib()
        from ponyexl3.convert.regularize import regularize_public_weight
        rng = np.random.default_rng(711)
        public = rng.standard_normal((HAD_BLOCK, HAD_BLOCK), dtype=np.float32)
        public *= np.linspace(0.1, 3.0, HAD_BLOCK, dtype=np.float32)[None, :]
        skewed = np.ones(HAD_BLOCK, dtype=np.float32)
        skewed[0] = 1e8
        for quantizer in ("direct", "ldlq"):
            for cal in (None, np.ones(HAD_BLOCK, np.float32), skewed):
                want = regularize_public_weight(public, seed=713, hessian_diag=cal)
                got = prepare_expert_bank([public], [713], [cal], quantizer=quantizer)
                np.testing.assert_array_equal(got.inner[0], want.inner)
                np.testing.assert_array_equal(got.suh[0], want.suh)
                np.testing.assert_array_equal(got.svh[0], want.svh)
        self.assertFalse(np.array_equal(
            regularize_public_weight(public, seed=713).inner,
            regularize_public_weight(public, seed=713, hessian_diag=skewed).inner))

    def test_an_expert_no_token_reached_takes_the_rotated_flat_prior(self):
        rng = np.random.default_rng(51)
        suh = rng.standard_normal(HAD_BLOCK, dtype=np.float32)
        prior = _common.prior_ldl_blocks(suh)
        self.assertGreater(float(np.abs(prior).max()), 0.05,
                           "a varying suh makes even a FLAT public Hessian dense")
        cals = bank_calibrations(2, 4, calibration=None,
                                 imatrix_flat=np.ones(8, np.float32),
                                 routed_rows=np.array([5.0, 0.0], np.float32),
                                 zero_routed=(zeros := []), layer_key="L0.gate")
        self.assertIsNotNone(cals[0])
        self.assertIsNone(cals[1])
        self.assertEqual(zeros, ["L0.gate#1"])


class GlobalScaleTests(unittest.TestCase):
    """`regularize_public_weight` divides by MCG's measured RMS, so every other codebook
    lands off its own scale unless the bounded golden-section search corrects for it."""

    def test_the_search_is_bounded_and_brackets_this_converters_codebook(self):
        self.assertIn(CODEBOOK, _common.G_SCALE_BRACKET)
        for low, high in _common.G_SCALE_BRACKET.values():
            self.assertLessEqual(_common.g_scale_iterations(low, high) + 2,
                                 _common.G_SCALE_MAX_EVALS)

    def test_the_found_scale_beats_one_on_the_sampled_tiles(self):
        import mlx.core as mx
        reg = _regularized(rows=HAD_BLOCK, cols=2 * HAD_BLOCK)
        rows, cols = reg.inner.shape
        index = _common.sample_tile_index(rows, cols, count=64)
        tiles = _common.sample_tiles_mlx(mx.array(reg.inner[None], dtype=mx.float32), index)
        cb = codebook_mode(CODEBOOK)
        g = float(np.array(_common.g_scale_search_mlx(
            tiles, k=3, cb=cb, codebook=CODEBOOK, window=WINDOW_DEFAULT,
            scratch_bytes=1 << 29))[0])
        from ponyexl3.convert.direct import _TENSOR_CORE_PERM
        sample = np.array(tiles)[0][:, _TENSOR_CORE_PERM]

        def mse(scale):
            _s, decoded = _common.search_tiles(np.ascontiguousarray(sample * np.float32(scale)),
                                               k=3, cb=cb, window=WINDOW_DEFAULT, chunk=4096,
                                               want_decoded=True)
            return float(np.mean((decoded / np.float32(scale) - sample) ** 2, dtype=np.float64))
        self.assertLess(mse(g), mse(1.0), f"g={g} did not beat the unscaled search")


class WindowTests(unittest.TestCase):
    def _searched(self, window, k=3, rows=HAD_BLOCK, cols=HAD_BLOCK):
        cb = codebook_mode(CODEBOOK)
        reg = _regularized(rows=rows, cols=cols, seed=77)
        to_tiles, from_tiles = _common._tile_helpers()
        states, decoded = _common.search_tiles(to_tiles(reg.inner), k=k, cb=cb, window=window,
                                               chunk=4096, want_decoded=True)
        packed = _common.pack_states(states, k, rows // 16, cols // 16)
        return packed, from_tiles(decoded, rows, cols)

    def test_a_w8_pack_decodes_to_exactly_what_the_search_chose(self):
        from ponyexl3.ref.reconstruct import reconstruct_inner
        packed, want = self._searched(8)
        got = np.asarray(reconstruct_inner(packed, 3, mul1=True, window=8), np.float32)
        self.assertTrue(np.allclose(got, want, rtol=0, atol=1e-3),
                        msg=f"max |diff| {float(np.abs(got - want).max())}")

    def test_a_w8_pack_read_at_w16_is_noise(self):
        """Red-on-revert for `expert_quant.window`: the window is a DECODE parameter."""
        from ponyexl3.ref.reconstruct import reconstruct_inner
        packed, want = self._searched(8)
        right = np.asarray(reconstruct_inner(packed, 3, mul1=True, window=8), np.float32)
        wrong = np.asarray(reconstruct_inner(packed, 3, mul1=True, window=16), np.float32)
        self.assertLess(float(np.mean((right - want) ** 2)), 1e-5)
        self.assertGreater(float(np.mean((wrong - want) ** 2)), 0.1)

    def test_the_window_must_sit_in_the_searchs_own_range(self):
        for good in (8, 12, 16):
            self.assertEqual(validate_window(good, 3), good)
        for bad in (3, 2, 25):
            with self.assertRaises(RuntimeError):
                validate_window(bad, 3)

    def test_the_pack_records_the_window_it_was_searched_at(self):
        rng = np.random.default_rng(91)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            hf, pack, dst = td / "hf", td / "pack", td / "out"
            _write_resume_fixture(hf, pack, rng)
            convert_pack(hf, pack, dst, quantizer="direct", k=3, window=8, quality=False)
            cfg = json.loads((dst / "config.json").read_text())
            self.assertEqual(cfg["expert_quant"]["window"], 8)
            self.assertEqual(cfg["expert_quant"]["k"], 3)
            self.assertEqual(read_stamp(dst / layer_proj_shard(0, "gate"))["window"], "8")


class ConvertedPackTests(unittest.TestCase):
    """Converter-level red-on-revert: each of these passed on the pre-port converter only
    because the calibration and the scale search never reached the weights."""

    @staticmethod
    def _payload(path) -> bytes:
        header, off = read_header(path)
        return b"".join(_common.read_raw(path, off, header[key]).tobytes()
                        for key in sorted(header))

    @staticmethod
    def _imatrix(rows_vec, e=2, dim=128, *, skew: bool = True) -> dict:
        """A real capture is SKEWED — a few input channels carry most of the activation
        energy, which is what makes `regularize_public_weight` drop the output scales."""
        rng = np.random.default_rng(21)
        p = "model.language_model.layers.0.mlp.experts."
        store = {}
        for name in ("gate_up_proj", "down_proj"):
            v = np.abs(rng.standard_normal(e * dim, dtype=np.float32)) + 0.1
            if skew:
                v.reshape(e, dim)[:, :2] = 1e6
            store[p + name] = v
        store[p + "gate_up_proj.rows"] = np.asarray(rows_vec, dtype=np.float32)
        return store

    def _three(self, td, **kw):
        td = Path(td)
        hf, pack = td / "hf", td / "pack"
        if not hf.exists():
            _write_resume_fixture(hf, pack, np.random.default_rng(9))
        out = td / kw.pop("tag")
        args = dict(quantizer="direct", k=3, quality=False)
        args.update(kw)
        convert_pack(hf, pack, out, **args)
        return [self._payload(out / layer_proj_shard(0, p)) for p in PROJECTIONS]

    def test_an_imatrix_changes_the_weights_even_on_the_direct_path(self):
        """`regularize_public_weight` takes the calibration diagonal, so an imatrix moves
        the inner matrix before any search runs; the converter used to drop it there."""
        with tempfile.TemporaryDirectory() as td:
            plain = self._three(td, tag="plain")
            calibrated = self._three(td, tag="cal", imatrix=self._imatrix([8.0, 8.0]),
                                     imatrix_sha="abc")
            self.assertEqual([a != b for a, b in zip(plain, calibrated)], [True] * 3)

    def test_ldlq_does_not_collapse_into_the_direct_pack(self):
        """Feeding LDLQ the raw public diagonal produced an identity factor and a pack
        byte-identical to `--quantizer direct`. Under the rotated Hessian they differ:
        that difference IS the calibration."""
        with tempfile.TemporaryDirectory() as td:
            imat = self._imatrix([8.0, 8.0])
            direct = self._three(td, tag="direct", imatrix=imat, imatrix_sha="abc")
            ldlq = self._three(td, tag="ldlq", quantizer="ldlq", imatrix=imat, imatrix_sha="abc")
            self.assertEqual([a != b for a, b in zip(direct, ldlq)], [True] * 3)

    def test_the_global_scale_search_changes_the_weights(self):
        with tempfile.TemporaryDirectory() as td:
            searched = self._three(td, tag="gss")
            at_one = self._three(td, tag="one", g_scale=False)
            self.assertEqual([a != b for a, b in zip(searched, at_one)], [True] * 3)

    def test_the_pack_reconstructs_the_source_bank(self):
        from ponyexl3.ref.reconstruct import reconstruct_public_weights
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            hf, pack, dst = td / "hf", td / "pack", td / "out"
            _write_resume_fixture(hf, pack, np.random.default_rng(23))
            convert_pack(hf, pack, dst, quantizer="ldlq", k=4, window=12,
                         imatrix=self._imatrix([8.0, 0.0]), imatrix_sha="abc")
            src_header, src_off = read_header(hf / "model.safetensors")
            key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
            want = np.ascontiguousarray(
                read_raw(hf / "model.safetensors", src_off, src_header[key])[0, :128].T)
            shard = dst / layer_proj_shard(0, "gate")
            header, off = read_header(shard)
            base = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
            got = np.asarray(reconstruct_public_weights(
                _common.read_raw(shard, off, header[base + ".trellis"])[0],
                _common.read_raw(shard, off, header[base + ".suh"])[0],
                _common.read_raw(shard, off, header[base + ".svh"])[0], 4, mul1=True,
                window=12), dtype=np.float32)
            rel = float(np.sqrt(np.sum((got - want) ** 2) / np.sum(want ** 2)))
            self.assertTrue(np.all(np.isfinite(got)))
            self.assertLess(rel, 0.6, msg=f"relative reconstruction error {rel}")


class StampedResumeTests(unittest.TestCase):
    def _convert(self, td, **kw):
        td = Path(td)
        hf, pack, dst = td / "hf", td / "pack", td / "out"
        if not hf.exists():
            _write_resume_fixture(hf, pack, np.random.default_rng(5))
        args = dict(quantizer="direct", k=4, quality=False)
        args.update(kw)
        return hf, pack, dst, convert_pack(hf, pack, dst, **args)

    def test_every_shard_carries_the_settings_that_wrote_it(self):
        with tempfile.TemporaryDirectory() as td:
            _hf, _pack, dst, _plan = self._convert(td, k=3, window=12, codebook="tiny")
            got = read_stamp(dst / layer_proj_shard(0, "down"))
            self.assertEqual(got, shard_stamp(k=3, codebook="tiny", window=12,
                                              quantizer="direct", imatrix_sha=None))
            self.assertEqual(got["converter"], CONVERTER_VERSION)
            self.assertNotIn("/", CONVERTER_VERSION)

    def test_a_shard_of_another_codebook_window_or_quantizer_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as td:
            hf, pack, dst, first = self._convert(td, k=3, window=16)
            self.assertEqual(first["skipped"], 0)
            same = convert_pack(hf, pack, dst, quantizer="direct", k=3, window=16, quality=False)
            self.assertEqual(same["skipped"], 3)
            self.assertEqual(same["rewritten"], [])
            for label, kw, needle in (
                ("codebook", dict(codebook="tiny"), "codebook=mul1 != tiny"),
                ("window", dict(window=8), "window=16 != 8"),
                ("quantizer", dict(quantizer="ldlq"), "quantizer=direct != ldlq-rotated"),
                ("g_scale", dict(g_scale=False), "g_scale=gss != one"),
            ):
                args = dict(quantizer="direct", k=3, window=16, quality=False)
                args.update(kw)
                plan = convert_pack(hf, pack, dst, **args)
                self.assertEqual(plan["skipped"], 0, label)
                self.assertEqual(len(plan["rewritten"]), 3, label)
                self.assertIn(needle, plan["rewritten"][0], label)
                # restore the baseline so the next arm differs in ONE setting
                convert_pack(hf, pack, dst, quantizer="direct", k=3, window=16, quality=False)

    def test_an_unstamped_shard_is_named_not_adopted(self):
        with tempfile.TemporaryDirectory() as td:
            hf, pack, dst, _ = self._convert(td, k=3)
            shard = dst / layer_proj_shard(0, "gate")
            header, off = read_header(shard)
            write_safetensors_raw(str(shard), {
                key: (meta["dtype"], tuple(meta["shape"]),
                      _common.read_raw(shard, off, meta).tobytes())
                for key, meta in header.items()})
            plan = convert_pack(hf, pack, dst, quantizer="direct", k=3, quality=False)
            self.assertEqual(plan["skipped"], 2)
            self.assertEqual(plan["rewritten"],
                             [f"{layer_proj_shard(0, 'gate')}: no stamp"])

    def test_no_resume_rewrites_a_matching_shard(self):
        with tempfile.TemporaryDirectory() as td:
            hf, pack, dst, _ = self._convert(td, k=3)
            plan = convert_pack(hf, pack, dst, quantizer="direct", k=3, quality=False,
                                resume=False)
            self.assertEqual(plan["skipped"], 0)
            self.assertEqual(plan["rewritten"], [])

    def test_an_imatrix_of_another_capture_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as td:
            hf, pack, dst, _ = self._convert(td, k=3)
            imat = {
                "model.language_model.layers.0.mlp.experts.gate_up_proj": np.ones(2 * 128, np.float32),
                "model.language_model.layers.0.mlp.experts.down_proj": np.ones(2 * 128, np.float32),
                "model.language_model.layers.0.mlp.experts.gate_up_proj.rows": np.array([4.0, 4.0], np.float32),
            }
            plan = convert_pack(hf, pack, dst, quantizer="direct", k=3, quality=False,
                                imatrix=imat, imatrix_sha="deadbeef")
            self.assertEqual(plan["skipped"], 0)
            self.assertIn("imatrix_sha256=none != deadbeef", plan["rewritten"][0])
            self.assertEqual(read_stamp(dst / layer_proj_shard(0, "gate"))["imatrix_sha256"],
                             "deadbeef")

    def test_the_seed_is_stable_per_bank_layer_expert_and_projection(self):
        """A seed that counted emissions made `--resume` write a DIFFERENT pack from a
        single pass, and made the MTP head's experts reuse trunk layer 0's seeds."""
        self.assertEqual(expert_seed(3, 9, "up"), expert_seed(3, 9, "up"))
        distinct = {expert_seed(l, e, p, mtp=m)
                    for l in (0, 3) for e in (0, 9) for p in PROJECTIONS for m in (False, True)}
        self.assertEqual(len(distinct), 2 * 2 * 3 * 2)
        for seed in distinct:
            self.assertTrue(0 <= seed <= 0x7FFF_FFFF)

    def test_the_mtp_bank_does_not_collide_with_trunk_layer_zero(self):
        """`mtp.layers.0` carries a MoE bank at the trunk's exact [E, 2I, H] geometry, so
        a shard named by (layer, projection) alone would adopt the trunk's."""
        self.assertNotEqual(layer_proj_shard(0, "gate"), layer_proj_shard(0, "gate", mtp=True))
        self.assertTrue(layer_proj_shard(0, "gate", mtp=True).startswith("model-exl3-mtp-"))
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            hf, pack, dst = td / "hf", td / "pack", td / "out"
            _write_resume_fixture(hf, pack, rng, with_mtp=True)
            plan = convert_pack(hf, pack, dst, quantizer="direct", k=3, quality=False)
            self.assertEqual(plan["skipped"], 0)
            shards = sorted(p.name for p in dst.glob("model-exl3-*.safetensors"))
            self.assertEqual(len(shards), 6)
            wm = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
            self.assertEqual(wm["language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.trellis"],
                             layer_proj_shard(0, "gate", mtp=True))
            self.assertEqual(wm["language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis"],
                             layer_proj_shard(0, "gate"))
            again = convert_pack(hf, pack, dst, quantizer="direct", k=3, quality=False)
            self.assertEqual(again["skipped"], 6)


def pick_real_experts(hf_dir: str | Path, imatrix_path: str | Path, layer: int = 0) -> int:
    import time
    _ensure_lib()
    from ponyexl3.convert.direct import quantize_inner_matrix_direct
    from ponyexl3.convert.hessian import block_ldl, ldlq_inner_matrix, prepare_hessian_for_ldl
    from ponyexl3.convert.regularize import regularize_public_weight
    from ponyexl3.ref.codebook import CodebookMode
    from ponyexl3.ref.reconstruct import reconstruct_public_weights
    from convert_dsv4_weights import mlx_affine_dequant_f32, mlx_affine_quant
    import mlx.core as mx

    hf_dir = Path(hf_dir)
    imat = load_imatrix(imatrix_path)
    rows_key = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj.rows"
    gu_key = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
    rows = np.asarray(imat[rows_key], dtype=np.float32)
    gu_flat = np.asarray(imat[gu_key], dtype=np.float32)
    order = np.argsort(rows)
    cold, med, hot = int(order[0]), int(order[len(order) // 2]), int(order[-1])
    picks = [("cold", cold), ("median", med), ("hot", hot)]
    print(f"layer {layer} routed tokens: cold={rows[cold]:.0f} e={cold}  median={rows[med]:.0f} e={med}  hot={rows[hot]:.0f} e={hot}", flush=True)
    idx = json.loads((hf_dir / "model.safetensors.index.json").read_text())
    hf_file = hf_dir / idx["weight_map"][gu_key]
    header, data_off = read_header(hf_file)
    bank = read_raw(hf_file, data_off, header[gu_key])
    if bank.dtype == np.uint16:
        from convert_dsv4_weights import bf16_to_f32
        bank = bf16_to_f32(bank)
    bank = np.asarray(bank, dtype=np.float32)
    half = bank.shape[1] // 2
    hidden = bank.shape[2]
    cb = CodebookMode.MCG

    def score(w, what, v):
        return _weighted_row_err(w, what, v), _output_err(w, what, v)

    for tag, ei in picks:
        w = np.ascontiguousarray(bank[ei, :half].T)
        v = imatrix_expert_vector(gu_flat, ei, hidden)
        v = np.maximum(v, 1e-8)
        mx.set_default_device(mx.cpu)
        wq, sc, bi = mlx_affine_quant(w.T, 4, 64)
        w_aff = mlx_affine_dequant_f32(
            np.frombuffer(wq[2], dtype=np.uint32).reshape(wq[1]),
            np.frombuffer(sc[2], dtype=np.uint16).reshape(sc[1]),
            np.frombuffer(bi[2], dtype=np.uint16).reshape(bi[1]),
            4, 64,
        ).T
        mx.set_default_device(mx.gpu)
        q_aff = score(w, w_aff, v)
        reg = regularize_public_weight(w, seed=1)
        t0 = time.perf_counter()
        packed, _, _ = quantize_inner_matrix_direct(reg.inner, k=4, cb=cb, search_backend="metal", return_states=False)
        t_direct = time.perf_counter() - t0
        w_direct = reconstruct_public_weights(packed, reg.suh.astype(np.float16), reg.svh.astype(np.float16), 4, mcg=True).astype(np.float32)
        q_direct = score(w, w_direct, v)
        t1 = time.perf_counter()
        prep = prepare_hessian_for_ldl(diag_hessian(v))
        ldl = block_ldl(prep.hessian)
        res = ldlq_inner_matrix(reg.inner, ldl.l, k=4, cb=cb, hessian=prep.hessian, search_backend="metal", collect_states=False, compute_proxy=False)
        t_ldlq = time.perf_counter() - t1
        w_ldlq = reconstruct_public_weights(res.packed, reg.suh.astype(np.float16), reg.svh.astype(np.float16), 4, mcg=True).astype(np.float32)
        q_ldlq = score(w, w_ldlq, v)
        print(f"  {tag} e={ei} tokens={rows[ei]:.0f}", flush=True)
        print(f"    affine-4-g64     wRMS={q_aff[0]:.5f} out={q_aff[1]:.5f}", flush=True)
        print(f"    direct           wRMS={q_direct[0]:.5f} out={q_direct[1]:.5f} {t_direct:.3f}s", flush=True)
        print(f"    ldlq-diag        wRMS={q_ldlq[0]:.5f} out={q_ldlq[1]:.5f} {t_ldlq:.3f}s", flush=True)

    n = 16
    regs = []
    for i in range(n):
        ei = int(order[-(i + 1)])
        w = np.ascontiguousarray(bank[ei, :half].T)
        regs.append(regularize_public_weight(w, seed=2 + i))
    inners = [r.inner for r in regs]
    vs = [imatrix_expert_vector(gu_flat, int(order[-(i + 1)]), hidden) for i in range(n)]
    t2 = time.perf_counter()
    _quantize_direct_batch(inners, 4, cb)
    dt_direct = time.perf_counter() - t2
    t3 = time.perf_counter()
    for inner, vv in zip(inners, vs):
        prep = prepare_hessian_for_ldl(diag_hessian(np.maximum(vv, 1e-8)))
        ldl = block_ldl(prep.hessian)
        ldlq_inner_matrix(inner, ldl.l, k=4, cb=cb, hessian=prep.hessian, search_backend="metal", collect_states=False, compute_proxy=False)
    dt_ldlq = time.perf_counter() - t3
    print(f"batch N={n} direct {n / dt_direct:.2f} proj/s  ldlq-serial {n / dt_ldlq:.2f} proj/s", flush=True)
    fat = np.concatenate(inners, axis=1)
    vmean = np.mean(np.stack(vs, 0), 0)
    t4 = time.perf_counter()
    prep = prepare_hessian_for_ldl(diag_hessian(np.maximum(vmean, 1e-8)))
    ldl = block_ldl(prep.hessian)
    ldlq_inner_matrix(fat, ldl.l, k=4, cb=cb, hessian=prep.hessian, search_backend="metal", collect_states=False, compute_proxy=False)
    dt_fat = time.perf_counter() - t4
    print(f"batch N={n} ldlq-concat-out {n / dt_fat:.2f} proj/s ({dt_fat:.2f}s wall)", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--pick-experts", action="store_true")
    ap.add_argument("--from-exl3", default=None)
    ap.add_argument("--dense", default=None)
    ap.add_argument("--ngram-src", default=None)
    ap.add_argument("--ngram-bin", default=None)
    ap.add_argument("--ngram-out", default=None)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--hf", default=None)
    ap.add_argument("--pack", default=None)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--component-output", default=None,
                    help="write canonical component shards to this new directory after conversion")
    ap.add_argument("--share-with", default=None,
                    help="existing canonical component pack whose unchanged files may be shared")
    ap.add_argument("--quantizer", default="direct", choices=("ldlq", "direct"))
    ap.add_argument("--k", default=str(K), help="trellis rate, any multiple of 1/16 in [2, 8]")
    ap.add_argument("--codebook", default=CODEBOOK, choices=("mul1", "tiny", "mcg"))
    ap.add_argument("--window", type=int, default=WINDOW_DEFAULT,
                    help="codeword window the search hashes, 8..16 served; stamped into the pack")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--imatrix", default=None)
    ap.add_argument("--batch-size", type=int, default=BATCH_EXPERTS_DEFAULT,
                    help="experts quantized per GPU batch")
    ap.add_argument("--scratch-gb", type=float, default=SCRATCH_GB_DEFAULT,
                    help="Metal search scratch budget; sets the tiles per launch")
    ap.add_argument("--no-g-scale", action="store_true",
                    help="skip the global codebook-scale search (regularize at g=1)")
    ap.add_argument("--no-quality", action="store_true",
                    help="skip the per-expert imatrix-weighted reconstruction error")
    ap.add_argument("--no-resume", action="store_true",
                    help="rewrite every shard, even one whose stamp already matches")
    args = ap.parse_args()
    if args.share_with and not args.component_output:
        ap.error("--share-with requires --component-output")
    if args.component_output and (args.bench or args.pick_experts or args.ngram_out or args.self_test):
        ap.error("--component-output is only supported for conversion, restack, or dense compose")
    if args.component_output:
        component_path = Path(args.component_output)
        if component_path.exists() or component_path.is_symlink():
            ap.error("--component-output must be a new directory")
        if args.dst:
            staged_path = Path(args.dst).resolve()
            component_real = component_path.resolve()
            if (
                staged_path == component_real
                or staged_path in component_real.parents
                or component_real in staged_path.parents
            ):
                ap.error("--component-output must be outside the --dst staging directory")
    if args.bench:
        return bench_batch_quality()
    if args.ngram_out:
        if not args.ngram_src:
            ap.error("--ngram-out needs --ngram-src")
        print(write_bf16_ngram_table(args.ngram_src, args.ngram_out), flush=True)
        return 0
    if args.dense:
        if not (args.pack and args.dst):
            ap.error("--dense needs --pack and --dst")
        if bool(args.ngram_src) == bool(args.ngram_bin):
            ap.error("--dense needs exactly one of --ngram-src, --ngram-bin")
        out = compose_pack(args.dense, args.pack, args.dst,
                           ngram_src=args.ngram_src, ngram_bin=args.ngram_bin)
        print(f"composed {len(out['weight_map'])} tensors, {out['bytes'] / 1e9:.1f} GB", flush=True)
        print(f"carried from the expert pack: {len(out['carried'])} tensors", flush=True)
        if args.component_output:
            repack_component_output(args.dst, args.component_output, args.share_with)
        return 0
    if args.from_exl3:
        if not (args.pack and args.dst):
            ap.error("--from-exl3 needs --pack and --dst")
        restack_from_exl3(args.from_exl3, args.pack, args.dst)
        if args.component_output:
            repack_component_output(args.dst, args.component_output, args.share_with)
        return 0
    if args.pick_experts:
        if not (args.hf and args.imatrix):
            ap.error("--pick-experts needs --hf and --imatrix")
        return pick_real_experts(args.hf, args.imatrix, layer=args.layer)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    if not (args.hf and args.pack and args.dst):
        ap.error("--hf --pack --dst are required unless --self-test")
    cal = None
    if args.calibration:
        cal = np.load(os.path.expanduser(args.calibration))
    imat = imat_sha = None
    if args.imatrix:
        imatrix_path = os.path.expanduser(args.imatrix)
        imat = load_imatrix(imatrix_path)
        imat_sha = file_sha256(imatrix_path)
    t0 = time.perf_counter()
    plan = convert_pack(
        args.hf, args.pack, args.dst, quantizer=args.quantizer, k=parse_k(args.k),
        codebook=args.codebook, window=args.window, calibration=cal, imatrix=imat,
        imatrix_sha=imat_sha, batch_size=args.batch_size, scratch_gb=args.scratch_gb,
        g_scale=not args.no_g_scale, quality=not args.no_quality, resume=not args.no_resume,
    )
    wall = time.perf_counter() - t0
    se = plan["search"]
    print(f"pack {args.dst}: {plan['skipped']} shards skipped, {len(plan['rewritten'])} "
          f"rewritten, {wall:.1f}s wall; search {se['seconds']:.1f}s GPU "
          f"({100.0 * se['seconds'] / max(wall, 1e-9):.1f}% of wall), {se['launches']} "
          f"launches, {se['tiles']} tiles", flush=True)
    werr = plan.get("weighted_rel_err") or {}
    if werr:
        allv = [x for v in werr.values() for x in v]
        print(f"imatrix-weighted relative reconstruction error: "
              f"{len(werr)} shards, mean {sum(allv) / len(allv):.5f}", flush=True)
    if args.component_output:
        repack_component_output(args.dst, args.component_output, args.share_with)
    return 0


if __name__ == "__main__":
    sys.exit(main())
