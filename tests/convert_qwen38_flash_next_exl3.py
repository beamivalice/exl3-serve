#!/usr/bin/env python3
"""Replace routed-expert affine banks in a qwen4_exp 4/8 pack with stacked EXL3 K4.

Input: bf16 HF checkpoint + the existing mixed 4/8 pack. Output: a new pack dir
where every routed gate/up/down bank is EXL3 K4 (MUL1), stacked [E, ...] per
projection so gather kernels index expert e on axis 0. Every non-expert file
from the 4/8 pack is hard-linked; mixed shards are rewritten with the expert
tensors dropped and remaining tensors copied as raw bytes. config.json carries
expert_quant = {format: exl3, k: 4, codebook: mul1}.

Default quantization is LDLQ with a captured Hessian; --quantizer direct is
the synthetic-test path. Calibration rows, when captured, are the MLP input
(hidden) for gate/up and the SwiGLU activation (intermediate) for down.

  python3 tests/convert_qwen38_flash_next_exl3.py --self-test
  python3 tests/convert_qwen38_flash_next_exl3.py \\
      --hf /Users/beam/llm/models/Qwen/Qwen3.8-Flash-Next \\
      --pack /Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit \\
      --dst /path/to/out
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_dsv4_weights import write_safetensors_raw  # noqa: E402
from convert_qwen38_flash_next import read_header, read_raw, rename  # noqa: E402

K = 4
CODEBOOK = "mul1"
PACKED_K4 = 256 * K // 16


def packed_hw(k: int) -> int:
    return 256 * k // 16


def k_from_packed(last: int) -> int:
    if last % 16 != 0:
        raise RuntimeError(f"packed dim {last} not divisible by 16")
    k = last // 16
    if k not in (2, 3, 4):
        raise RuntimeError(f"unsupported packed K={k} from last dim {last}")
    return k
HAD = 128
MCG_MULT = 0xCBAC1FED

SWITCH = ".mlp.switch_mlp."
HF_GATE_UP = ".mlp.experts.gate_up_proj"
HF_DOWN = ".mlp.experts.down_proj"


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


def diag_hessian(v: np.ndarray) -> np.ndarray:
    vec = np.asarray(v, dtype=np.float32).reshape(-1)
    return np.diag(vec)


def imatrix_expert_vector(flat: np.ndarray, expert: int, dim: int) -> np.ndarray:
    arr = np.asarray(flat, dtype=np.float32).reshape(-1)
    start = expert * dim
    return arr[start : start + dim].copy()


def calib_for_expert(routed_tokens: int, moments: np.ndarray) -> tuple[str, np.ndarray | None]:
    if routed_tokens <= 0:
        return "ldlq-gaussian-256", None
    return "imatrix-diagonal", np.asarray(moments, dtype=np.float32).reshape(-1)


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


def _quantize_public(
    public: np.ndarray,
    *,
    k: int,
    codebook: str,
    quantizer: str,
    calibration: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _ensure_lib()
    from ponyexl3.convert.direct import quantize_inner_matrix_direct
    from ponyexl3.convert.hessian import block_ldl, capture_hessian, ldlq_inner_matrix, prepare_hessian_for_ldl
    from ponyexl3.convert.regularize import regularize_public_weight
    from ponyexl3.ref.codebook import CodebookMode

    cb = CodebookMode.MCG if codebook == "mcg" else CodebookMode.MUL1
    reg = regularize_public_weight(public.astype(np.float32), seed=seed)
    suh = reg.suh.astype(np.float16)
    svh = reg.svh.astype(np.float16)
    if quantizer == "direct":
        packed, _, _ = quantize_inner_matrix_direct(
            reg.inner, k=k, cb=cb, search_backend="metal", return_states=False
        )
        return packed.astype(np.uint16), suh, svh
    rows = public.shape[0]
    if calibration is None:
        rng = np.random.default_rng(seed + 17)
        acts = rng.standard_normal((256, rows), dtype=np.float32)
        hessian = capture_hessian(acts)
    elif calibration.ndim == 1:
        if calibration.shape[0] != rows:
            raise ValueError(f"calibration features {calibration.shape[0]} != {rows}")
        hessian = diag_hessian(calibration)
    else:
        acts = np.asarray(calibration, dtype=np.float32)
        if acts.shape[1] != rows:
            raise ValueError(f"calibration features {acts.shape[1]} != {rows}")
        hessian = capture_hessian(acts)
    prepared = prepare_hessian_for_ldl(hessian)
    ldl = block_ldl(prepared.hessian)
    result = ldlq_inner_matrix(
        reg.inner,
        ldl.l,
        k=k,
        cb=cb,
        hessian=prepared.hessian,
        search_backend="metal",
        collect_states=False,
        compute_proxy=False,
    )
    return result.packed.astype(np.uint16), suh, svh


def _stack_experts(
    bank: np.ndarray,
    *,
    k: int,
    codebook: str,
    quantizer: str,
    calibration: np.ndarray | None,
    seed: int,
    routed_rows: np.ndarray | None = None,
    imatrix_flat: np.ndarray | None = None,
    zero_routed: list | None = None,
    layer_key: str = "",
    batch_size: int = 32,
) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    _ensure_lib()
    from ponyexl3.convert.regularize import regularize_public_weight
    from ponyexl3.ref.codebook import CodebookMode
    cb = CodebookMode.MCG if codebook == "mcg" else CodebookMode.MUL1
    e, out_dim, in_dim = bank.shape
    publics = []
    cals = []
    for ei in range(e):
        publics.append(np.ascontiguousarray(bank[ei].T))
        cal = calibration
        if imatrix_flat is not None:
            moments = imatrix_expert_vector(imatrix_flat, ei, in_dim)
            ntok = int(routed_rows[ei]) if routed_rows is not None else 1
            mode, vec = calib_for_expert(ntok, moments)
            if mode != "imatrix-diagonal":
                if zero_routed is not None:
                    zero_routed.append(f"{layer_key}#{ei}")
                cal = None
            else:
                cal = vec
        cals.append(cal)
    trellis_list, suh_list, svh_list = [], [], []
    if quantizer == "direct":
        regs = [regularize_public_weight(p.astype(np.float32), seed=seed + ei) for ei, p in enumerate(publics)]
        inners = [r.inner for r in regs]
        packed_parts: list[np.ndarray] = []
        bs = max(1, int(batch_size))
        for start in range(0, e, bs):
            packed_parts.extend(_quantize_direct_batch(inners[start : start + bs], k, cb))
        for r, packed in zip(regs, packed_parts):
            trellis_list.append(packed.astype(np.uint16))
            suh_list.append(r.suh.astype(np.float16))
            svh_list.append(r.svh.astype(np.float16))
    else:
        for ei, public in enumerate(publics):
            packed, suh, svh = _quantize_public(
                public, k=k, codebook=codebook, quantizer=quantizer, calibration=cals[ei], seed=seed + ei
            )
            trellis_list.append(packed)
            suh_list.append(suh)
            svh_list.append(svh)
    trellis = np.stack(trellis_list, axis=0)
    suh = np.stack(suh_list, axis=0)
    svh = np.stack(svh_list, axis=0)
    return {
        "trellis": ("U16", trellis.shape, np.ascontiguousarray(trellis).tobytes()),
        "suh": ("F16", suh.shape, np.ascontiguousarray(suh).tobytes()),
        "svh": ("F16", svh.shape, np.ascontiguousarray(svh).tobytes()),
    }


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


def expert_quant_block(source: str, **extra) -> dict:
    """The `expert_quant` block the server admits: exl3, K, mul1 — nothing else."""
    block = {"format": "exl3", "k": int(K), "codebook": CODEBOOK, "out_scales": "svh", "source": source}
    block.update(extra)
    return block


def load_imatrix(path: str | Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file
    return load_file(str(path))


def convert_pack(
    hf_dir: str | Path,
    pack_dir: str | Path,
    dst: str | Path,
    *,
    quantizer: str = "direct",
    k: int = K,
    codebook: str = CODEBOOK,
    calibration: np.ndarray | None = None,
    imatrix: dict[str, np.ndarray] | None = None,
    batch_size: int = 32,
) -> dict:
    hf_dir = Path(hf_dir)
    pack_dir = Path(pack_dir)
    dst = Path(dst)
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
    seed = 0
    zero_routed: list[str] = []
    skipped = 0

    def emit(layer: int, proj: str, base: str, bank: np.ndarray, imat_flat, rows_vec, lkey: str):
        nonlocal seed, skipped
        e, out_dim, in_dim = bank.shape
        shard = layer_proj_shard(layer, proj)
        dest = dst / shard
        if shard_is_valid(dest, e, in_dim, out_dim, k):
            skipped += 1
            print(f"skip {shard}", flush=True)
            for suffix in (".trellis", ".suh", ".svh"):
                weight_map[base + suffix] = shard
            return
        t0 = __import__("time").perf_counter()
        stacked = _stack_experts(
            bank, k=k, codebook=codebook, quantizer=quantizer, calibration=calibration, seed=seed,
            routed_rows=rows_vec, imatrix_flat=imat_flat, zero_routed=zero_routed, layer_key=lkey,
            batch_size=batch_size,
        )
        seed += e
        named = {f"{base}.{suffix}": triple for suffix, triple in stacked.items()}
        write_safetensors_raw(str(dest), named)
        for key in named:
            weight_map[key] = shard
        print(f"wrote {shard}  {__import__('time').perf_counter() - t0:.1f}s  e={e} in={in_dim} out={out_dim}", flush=True)

    for hf_key, hf_file in hf_map.items():
        if hf_key.endswith("experts.gate_up_proj"):
            header, data_off = read_header(hf_dir / hf_file)
            arr = read_raw(hf_dir / hf_file, data_off, header[hf_key])
            if arr.dtype == np.uint16:
                from convert_dsv4_weights import bf16_to_f32
                arr = bf16_to_f32(arr)
            arr = np.asarray(arr, dtype=np.float32)
            half = arr.shape[1] // 2
            gate = np.ascontiguousarray(arr[:, :half])
            up = np.ascontiguousarray(arr[:, half:])
            gu_flat = None if imatrix is None else imatrix.get(hf_key)
            gu_rows = None if imatrix is None else imatrix.get(hf_key + ".rows")
            layer = parse_layer_from_hf_key(hf_key)
            for proj, bank in (("gate", gate), ("up", up)):
                base = mlx_switch_base(hf_key, proj)
                emit(layer, proj, base, bank, gu_flat, gu_rows, hf_key + "." + proj)
        elif hf_key.endswith("experts.down_proj"):
            header, data_off = read_header(hf_dir / hf_file)
            arr = read_raw(hf_dir / hf_file, data_off, header[hf_key])
            if arr.dtype == np.uint16:
                from convert_dsv4_weights import bf16_to_f32
                arr = bf16_to_f32(arr)
            arr = np.asarray(arr, dtype=np.float32)
            base = mlx_switch_base(hf_key, "down")
            parent = hf_key.replace("experts.down_proj", "experts.gate_up_proj")
            dn_flat = None if imatrix is None else imatrix.get(hf_key)
            gu_rows = None if imatrix is None else imatrix.get(parent + ".rows")
            layer = parse_layer_from_hf_key(hf_key)
            emit(layer, "down", base, arr, dn_flat, gu_rows, hf_key)
    total = 0
    for fname in sorted(set(weight_map.values())):
        total += os.path.getsize(dst / fname)
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2
    ))
    cfg = json.loads((pack_dir / "config.json").read_text())
    if imatrix is not None:
        cal_tag = "imatrix-diagonal"
    elif calibration is not None:
        cal_tag = "captured-rows"
    elif quantizer == "ldlq":
        cal_tag = "ldlq-gaussian-256"
    else:
        cal_tag = "none-direct"
    cfg["expert_quant"] = expert_quant_block("convert", k=int(k), codebook=codebook, quantizer=quantizer, calibration=cal_tag)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    plan["zero_routed"] = zero_routed
    plan["calibration"] = cal_tag
    plan["skipped"] = skipped
    return plan


def layer_proj_shard(layer: int, proj: str) -> str:
    return f"model-exl3-L{layer:02d}-{proj}.safetensors"


def parse_layer_from_hf_key(hf_key: str) -> int:
    import re
    m = re.search(r"\.layers\.(\d+)\.", hf_key)
    if not m:
        raise ValueError(hf_key)
    return int(m.group(1))


def shard_is_valid(path: str | Path, n_experts: int, in_dim: int, out_dim: int, k: int) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    try:
        header, _ = read_header(p)
    except Exception:
        return False
    trellis_keys = [k for k in header if k.endswith(".trellis")]
    if len(trellis_keys) != 1:
        return False
    sh = list(header[trellis_keys[0]]["shape"])
    if len(sh) != 4:
        return False
    if sh[:3] != [n_experts, in_dim // 16, out_dim // 16]:
        return False
    return sh[3] == packed_hw(k)


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


def _write_resume_fixture(hf: Path, pack: Path, rng) -> None:
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
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--hf", default=None)
    ap.add_argument("--pack", default=None)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--quantizer", default="direct", choices=("ldlq", "direct"))
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--imatrix", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    if args.bench:
        return bench_batch_quality()
    if args.from_exl3:
        if not (args.pack and args.dst):
            ap.error("--from-exl3 needs --pack and --dst")
        restack_from_exl3(args.from_exl3, args.pack, args.dst)
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
    imat = None
    if args.imatrix:
        imat = load_imatrix(os.path.expanduser(args.imatrix))
    convert_pack(
        args.hf, args.pack, args.dst, quantizer=args.quantizer, calibration=cal, imatrix=imat,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
