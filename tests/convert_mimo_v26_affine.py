#!/usr/bin/env python3
"""MiMo-V2.6-Flash-RL -> a RESIDENT mixed-width imatrix-weighted affine pack.

The source's routed experts are MXFP4 (`model.layers.{L}.mlp.experts.{E}.
{gate,up,down}_proj.{weight,weight_scale}`, e2m1 nibbles + e8m0 block-32
scales). This converter dequantizes them and re-quantizes each expert with
`dsv4_imatrix.weighted_affine_quant` at the per-layer (bits, group_size) that
`tests/mimo_v26_iq_allocate.py allocate` picked, then stacks the 256 experts
into the engine's expert-store banks

    model.layers.{L}.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}

one shard per layer. Everything that is not a routed expert is hard-linked from
the source unchanged; the source expert keys leave the index and the MXFP4
claim leaves `quantization_config`, so the pack cannot be re-read as MXFP4.
The engine solves (bits, group_size) per bank from the packed shapes
(`expert_quant.affineGeomFromShapes`); `expert_quant` in config.json records
the same table for humans.

Quantization is pure numpy across a fork pool — nothing here touches the GPU.

  python3 tests/convert_mimo_v26_affine.py --src <hf dir> \
      --imatrix im.safetensors --alloc alloc.json --dst <pack> [--resume] \
      [--layers 1-3] [--jobs 16] [--verify]
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import struct
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mimo_v26_iq_allocate import dequant_mxfp4  # noqa: E402
with warnings.catch_warnings():
    # convert_dsv4_weights' e8m0 LUT build overflows exp2 at code 254+ on import.
    warnings.simplefilter("ignore", RuntimeWarning)
    from dsv4_imatrix import dequant_np, weighted_affine_quant  # noqa: E402
    from convert_dsv4_weights import mx  # noqa: E402

# Bump on any change to what the quantizer writes — never for a comment or a
# test. A shard's `__metadata__` carries it and `--resume` refuses a shard whose
# stamp differs, so an editorial change must not cost a rerun.
CONVERTER_VERSION = "mimo-affine-1"
QUANTIZER = "weighted-affine"

PREFIX = "model.layers."
PROJECTIONS = ("gate", "up", "down")
COMPONENTS = ("weight", "scales", "biases")
NPDTYPE = {"U32": np.uint32, "BF16": np.uint16}
EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate|up|down)_proj\.(weight|weight_scale)$")


def wkey(a):
    """A width as the stamp, the config and the allocation table spell it."""
    return f"{int(a['bits'])}x{int(a['group_size'])}"


def role_of(proj):
    return "down" if proj == "down" else "gate_up"


def bank_dims(cfg, proj):
    """(out_dim, in_dim) of one expert's projection matrix."""
    hidden, inter = int(cfg["hidden_size"]), int(cfg["moe_intermediate_size"])
    return (hidden, inter) if proj == "down" else (inter, hidden)


def moe_layers(cfg):
    freq = cfg.get("moe_layer_freq")
    layers = int(cfg["num_hidden_layers"])
    if not isinstance(freq, list) or len(freq) != layers:
        raise RuntimeError(f"moe_layer_freq must list {layers} entries")
    return [i for i, f in enumerate(freq) if int(f) != 0]


def parse_layers(text, available):
    if not text:
        return list(available)
    allowed, picked = set(available), []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        for l in range(int(lo), int(hi or lo) + 1):
            if l not in allowed:
                raise RuntimeError(f"layer {l} carries no routed experts")
            picked.append(l)
    return sorted(set(picked))


def layer_shard(layer):
    return f"model-affine-L{layer:02d}.safetensors"


def switch_base(layer, proj):
    return f"{PREFIX}{layer}.mlp.switch_mlp.{proj}_proj"


# ------------------------------------------------------------------ source I/O

_H = {}


def header(path):
    path = str(path)
    if path not in _H:
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            _H[path] = (json.loads(f.read(hlen)), 8 + hlen)
    return _H[path]


def raw_memmap(src, index, name):
    path = Path(src) / index["weight_map"][name]
    hdr, data_off = header(path)
    meta = hdr[name]
    if meta["dtype"] != "U8":
        raise RuntimeError(f"{name}: {meta['dtype']}, expected U8")
    b, _ = meta["data_offsets"]
    return np.memmap(path, dtype=np.uint8, mode="r", offset=data_off + b,
                     shape=tuple(meta["shape"]))


def expert_f32(src, index, layer, expert, proj):
    stem = f"{PREFIX}{layer}.mlp.experts.{expert}.{proj}_proj"
    return dequant_mxfp4(raw_memmap(src, index, stem + ".weight"),
                         raw_memmap(src, index, stem + ".weight_scale"))


def plan_source(index):
    """A shard holding ONLY routed experts is dropped; a shard carrying anything
    else is hard-linked whole and its expert keys simply leave the index."""
    files = {}
    for key, fname in index["weight_map"].items():
        slot = files.setdefault(fname, {"expert": 0, "other": 0})
        slot["expert" if EXPERT_RE.match(key) else "other"] += 1
    return {
        "drop": sorted(f for f, s in files.items() if not s["other"]),
        "mixed": sorted(f for f, s in files.items() if s["other"] and s["expert"]),
        "keep_keys": [k for k in index["weight_map"] if not EXPERT_RE.match(k)],
    }


def stage_non_expert(src, dst, dropped):
    linked = 0
    for root, dirs, names in os.walk(src):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        rel_root = Path(root).relative_to(src)
        for name in names:
            if name.startswith("."):
                continue
            rel = rel_root / name if str(rel_root) != "." else Path(name)
            if str(rel) in dropped or str(rel) in (
                    "model.safetensors.index.json", "config.json"):
                continue
            source = Path(root) / name
            if not source.is_file() or source.is_symlink():
                continue
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            try:
                os.link(source, target)
            except OSError as exc:
                raise RuntimeError(
                    f"cannot hard-link {source} -> {target} ({exc.strerror}); the "
                    f"destination must sit on the source's filesystem") from None
            linked += 1
    return linked


# ------------------------------------------------------------------ shard I/O

def write_shard(path, tensors, metadata):
    """`tensors` maps name -> (dtype_str, ndarray); written without a bytes copy."""
    hdr, off = {"__metadata__": metadata}, 0
    for name, (dt, arr) in tensors.items():
        hdr[name] = {"dtype": dt, "shape": list(arr.shape),
                     "data_offsets": [off, off + arr.nbytes]}
        off += arr.nbytes
    hjson = json.dumps(hdr).encode()
    hjson += b" " * ((8 - (len(hjson) % 8)) % 8)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for _, (_, arr) in tensors.items():
            f.write(np.ascontiguousarray(arr).data)
    os.replace(tmp, path)


def shard_stamp(*, layer_alloc, imatrix_sha, source_sha):
    """What a shard was made with; every value a string, since safetensors
    `__metadata__` is a string map and `--resume` compares the whole dict."""
    return {
        "format": "affine",
        "gate_up": wkey(layer_alloc["gate_up"]),
        "down": wkey(layer_alloc["down"]),
        "quantizer": QUANTIZER,
        "imatrix_sha256": imatrix_sha,
        "source_sha256": source_sha,
        "converter": CONVERTER_VERSION,
    }


def width_of(layer_alloc, proj):
    a = layer_alloc[role_of(proj)]
    return int(a["bits"]), int(a["group_size"])


def bank_shapes(cfg, layer_alloc, n_experts, proj):
    out_dim, in_dim = bank_dims(cfg, proj)
    bits, gs = width_of(layer_alloc, proj)
    return {
        "weight": ("U32", [n_experts, out_dim, in_dim * bits // 32]),
        "scales": ("BF16", [n_experts, out_dim, in_dim // gs]),
        "biases": ("BF16", [n_experts, out_dim, in_dim // gs]),
    }


def shard_reuse_refusal(path, cfg, n_experts, layer, layer_alloc, stamp):
    """None when the shard on disk may be kept, else why it may not."""
    if not Path(path).is_file():
        return "absent"
    try:
        hdr, data_off = header(path)
    except Exception as exc:                                  # noqa: BLE001
        return f"unreadable header ({exc})"
    for proj in PROJECTIONS:
        for comp, (dt, shape) in bank_shapes(cfg, layer_alloc, n_experts, proj).items():
            meta = hdr.get(f"{switch_base(layer, proj)}.{comp}")
            if meta is None:
                return f"missing {proj}_proj.{comp}"
            if meta["dtype"] != dt or list(meta["shape"]) != shape:
                return (f"{proj}_proj.{comp} is {meta['dtype']}{list(meta['shape'])}, "
                        f"expected {dt}{shape}")
    end = max(m["data_offsets"][1] for k, m in hdr.items() if k != "__metadata__")
    if Path(path).stat().st_size < data_off + end:
        return "truncated payload"
    have = hdr.get("__metadata__") or {}
    differing = [f"{k}={have.get(k, '<absent>')} != {v}"
                 for k, v in sorted(stamp.items()) if have.get(k) != v]
    return "stamp " + ", ".join(differing) if differing else None


# ------------------------------------------------------- per-expert quantizer

_G = {}


def _quant_expert(task):
    layer, e = task
    out = {}
    for proj in PROJECTIONS:
        role = role_of(proj)
        a = _G["alloc"][f"layers.{layer}.{role}"]
        bits, gs = int(a["bits"]), int(a["group_size"])
        w = expert_f32(_G["src"], _G["index"], layer, e, proj)
        in_dim = w.shape[1]
        ch = _G["im"][f"{PREFIX}{layer}.mlp.experts.{role}_proj"]
        triples = weighted_affine_quant(w, bits, gs, ch[e * in_dim:(e + 1) * in_dim])
        out[proj] = tuple(np.frombuffer(raw, dtype=NPDTYPE[dt]).reshape(shape)
                          for dt, shape, raw in triples)
    return e, out


# ------------------------------------------------------------------- converter

def convert(src, dst, *, imatrix, imatrix_sha, alloc, layers=None, jobs=8,
            resume=False, verify=0, verbose=True):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    index = json.loads((src / "model.safetensors.index.json").read_text())
    n_experts = int(cfg["n_routed_experts"])
    all_moe = moe_layers(cfg)
    picked = parse_layers(layers, all_moe)
    plan = plan_source(index)
    source_sha = source_stamp(src, index)

    t0 = time.time()
    linked = stage_non_expert(src, dst, set(plan["drop"]))
    if verbose:
        print(f"staged {linked} files in {time.time() - t0:.1f}s, dropped "
              f"{len(plan['drop'])} expert-only shards, {len(plan['mixed'])} mixed "
              f"shard(s) linked with their expert keys left out of the index",
              flush=True)

    weight_map = {k: index["weight_map"][k] for k in plan["keep_keys"]}
    _G.update(src=str(src), index=index, im=imatrix, alloc=alloc)
    stats = {"shards": 0, "skipped": 0, "seconds": 0.0, "expert_bytes": 0,
             "layers": picked, "partial": picked != all_moe, "verify": []}

    def run_verify(layer, layer_alloc):
        rec = verify_layer(dst, src, index, imatrix, layer_alloc, cfg, layer,
                           n_experts, verify)
        stats["verify"].append(rec)
        if verbose:
            print(f"  verify layer {layer}: experts {rec['experts']} "
                  f"exact_vs_quantizer={rec['exact']} "
                  f"mlx_rel_rms={rec['mlx_rel_rms']:.3e} "
                  f"rel_err gate_up={rec['gate_up_rel_err']:.6f} "
                  f"down={rec['down_rel_err']:.6f}", flush=True)

    pool = mp.get_context("fork").Pool(jobs)
    try:
        for n, layer in enumerate(picked):
            layer_alloc = {r: alloc[f"layers.{layer}.{r}"] for r in ("gate_up", "down")}
            stamp = shard_stamp(layer_alloc=layer_alloc, imatrix_sha=imatrix_sha,
                                source_sha=source_sha)
            shard = layer_shard(layer)
            keys = [f"{switch_base(layer, p)}.{c}"
                    for p in PROJECTIONS for c in COMPONENTS]
            if resume:
                refusal = shard_reuse_refusal(dst / shard, cfg, n_experts, layer,
                                              layer_alloc, stamp)
                if refusal is None:
                    stats["skipped"] += 1
                    stats["expert_bytes"] += os.path.getsize(dst / shard)
                    for key in keys:
                        weight_map[key] = shard
                    if verbose:
                        print(f"skip {shard}", flush=True)
                    if verify:
                        run_verify(layer, layer_alloc)
                    continue
                if refusal != "absent" and verbose:
                    print(f"rewrite {shard}: {refusal}", flush=True)

            t_layer = time.time()
            banks = {}
            for proj in PROJECTIONS:
                banks[proj] = {
                    comp: np.empty(shape, dtype=NPDTYPE[dt]) for comp, (dt, shape)
                    in bank_shapes(cfg, layer_alloc, n_experts, proj).items()
                }
            done = 0
            for e, per_proj in pool.imap_unordered(
                    _quant_expert, [(layer, e) for e in range(n_experts)]):
                for proj, triple in per_proj.items():
                    for comp, arr in zip(COMPONENTS, triple):
                        banks[proj][comp][e] = arr
                done += 1
            if done != n_experts:
                raise RuntimeError(f"layer {layer}: {done}/{n_experts} experts")

            tensors = {}
            for proj in PROJECTIONS:
                for comp in COMPONENTS:
                    tensors[f"{switch_base(layer, proj)}.{comp}"] = (
                        "U32" if comp == "weight" else "BF16", banks[proj][comp])
            write_shard(dst / shard, tensors, stamp)
            _H.pop(str(dst / shard), None)
            for key in keys:
                weight_map[key] = shard
            size = os.path.getsize(dst / shard)
            dt = time.time() - t_layer
            stats["shards"] += 1
            stats["seconds"] += dt
            stats["expert_bytes"] += size
            del banks, tensors
            if verbose:
                print(f"wrote {shard}  {dt:.1f}s  gate_up {stamp['gate_up']} "
                      f"down {stamp['down']}  {size / 1e9:.2f} GB  "
                      f"[{n + 1}/{len(picked)}, {(time.time() - t0) / 60:.1f} min]",
                      flush=True)
            if verify:
                run_verify(layer, layer_alloc)
    finally:
        pool.close()
        pool.join()

    total = sum(os.path.getsize(dst / f) for f in sorted(set(weight_map.values()))
                if (dst / f).is_file())
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    (dst / "config.json").write_text(json.dumps(
        rewrite_config(cfg, alloc, picked, imatrix_sha), indent=2))
    (dst / "alloc.json").write_text(json.dumps(alloc, indent=1))
    stats["total_size"] = total
    stats["expert_params"] = n_experts * len(picked) * sum(
        o * i for o, i in (bank_dims(cfg, p) for p in PROJECTIONS))
    stats["index_errors"] = index_errors(dst, weight_map)
    return stats


def source_stamp(src, index):
    """One hash over every source shard the conversion reads (name, size, mtime)."""
    h = hashlib.sha256()
    for fname in sorted(set(index["weight_map"].values())):
        st = (Path(src) / fname).stat()
        h.update(f"{fname}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def index_errors(dst, weight_map):
    errs = []
    for key, fname in weight_map.items():
        if not (Path(dst) / fname).is_file():
            errs.append(f"{key} -> missing {fname}")
    if any(EXPERT_RE.match(k) for k in weight_map):
        errs.append("source expert keys survived into the index")
    return errs[:20]


def rewrite_config(config, alloc, picked, imatrix_sha):
    """Carry the affine allocation and strip the MXFP4 claim: nothing routed in
    the output pack is MXFP4 any more, so `store_dtype` and the block size must
    not survive to be read as an expert layout."""
    cfg = json.loads(json.dumps(config))
    quant = cfg.get("quantization_config")
    replaced = None
    if isinstance(quant, dict):
        replaced = quant.pop("store_dtype", None)
        quant.pop("mxfp4_block_size", None)
    block = {
        "format": "affine",
        "calibration": "imatrix-weighted affine (iQ-MLX)",
        "quantizer": QUANTIZER,
        "imatrix_sha256": imatrix_sha,
        "converter": CONVERTER_VERSION,
        "widths": dict(sorted(Counter(
            wkey(alloc[f"layers.{l}.{r}"])
            for l in picked for r in ("gate_up", "down")).items())),
        "layers": {str(l): {r: dict(alloc[f"layers.{l}.{r}"])
                            for r in ("gate_up", "down")} for l in picked},
    }
    if replaced is not None:
        block["replaced_store_dtype"] = replaced
    cfg["expert_quant"] = block
    return cfg


# ---------------------------------------------------------------- verification

def verify_layer(dst, src, index, imatrix, layer_alloc, cfg, layer, n_experts,
                 count, seed=20260922):
    """Round-trip a sample of the written experts: the stored bytes must BE what
    `weighted_affine_quant` produced, CPU `mx.dequantize` must read the same
    numbers out of them, and the imatrix-weighted relative error must land where
    mimo_v26_iq_allocate measured it (`errors.json` averages per-expert ratios,
    and a gate_up ratio is over gate and up together)."""
    m = mx()
    path = str(dst / layer_shard(layer))
    hdr, data_off = header(path)
    experts = sorted(random.Random(seed + layer).sample(range(n_experts), count))
    ratios = {"gate_up": [], "down": []}
    exact, mlx_num, mlx_den = True, 0.0, 0.0

    def stored(proj, comp, e):
        meta = hdr[f"{switch_base(layer, proj)}.{comp}"]
        return np.asarray(np.memmap(
            path, dtype=NPDTYPE[meta["dtype"]], mode="r",
            offset=data_off + meta["data_offsets"][0],
            shape=tuple(meta["shape"]))[e])

    for e in experts:
        acc = {"gate_up": [0.0, 0.0], "down": [0.0, 0.0]}
        for proj in PROJECTIONS:
            role = role_of(proj)
            bits, gs = width_of(layer_alloc, proj)
            w = expert_f32(src, index, layer, e, proj)
            in_dim = w.shape[1]
            ch = imatrix[f"{PREFIX}{layer}.mlp.experts.{role}_proj"]
            ch = np.asarray(ch[e * in_dim:(e + 1) * in_dim], dtype=np.float32)
            ref = [np.frombuffer(raw, dtype=NPDTYPE[dt]).reshape(shape) for dt, shape, raw
                   in weighted_affine_quant(w, bits, gs, ch)]
            got = [stored(proj, c, e) for c in COMPONENTS]
            exact &= all(np.array_equal(a, b) for a, b in zip(ref, got))

            deq = dequant_np(got[0], got[1], got[2], bits, gs).reshape(w.shape)
            mdeq = np.array(m.dequantize(
                m.array(got[0]), m.array(got[1]).view(m.bfloat16),
                m.array(got[2]).view(m.bfloat16),
                group_size=gs, bits=bits).astype(m.float32))
            mlx_num += float(((mdeq - deq) ** 2).sum())
            mlx_den += float((deq * deq).sum())

            wt = (ch + float(ch.mean()) * 1e-4 + 1e-30)[None, :]
            resid = deq - w
            acc[role][0] += float((wt * resid * resid).sum())
            acc[role][1] += float((wt * w * w).sum())
        for role, (num, den) in acc.items():
            ratios[role].append(num / max(den, 1e-30))
    return {
        "layer": layer, "experts": experts, "exact": bool(exact),
        "mlx_rel_rms": (mlx_num / max(mlx_den, 1e-30)) ** 0.5,
        "gate_up_rel_err": float(np.mean(ratios["gate_up"])),
        "down_rel_err": float(np.mean(ratios["down"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--alloc", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--layers", default=None, help="subset, e.g. 1 or 1,3-5")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="round-trip N random experts of every written layer")
    args = ap.parse_args()

    from safetensors.numpy import load_file
    imatrix_path = os.path.expanduser(args.imatrix)
    imatrix = load_file(imatrix_path)
    alloc = json.loads(Path(os.path.expanduser(args.alloc)).read_text())
    t0 = time.time()
    stats = convert(os.path.expanduser(args.src), os.path.expanduser(args.dst),
                    imatrix=imatrix, imatrix_sha=file_sha256(imatrix_path),
                    alloc=alloc, layers=args.layers, jobs=args.jobs,
                    resume=args.resume, verify=args.verify)
    wall = time.time() - t0
    bpw = stats["expert_bytes"] * 8 / max(stats["expert_params"], 1)
    print(f"done: {stats['shards']} shards written, {stats['skipped']} kept, "
          f"expert bytes {stats['expert_bytes'] / 1e9:.2f} GB, pack "
          f"{stats['total_size'] / 1e9:.2f} GB"
          + f", {bpw:.3f} bpw"
          + f" in {wall / 60:.1f} min", flush=True)
    if stats["partial"]:
        print(f"PARTIAL pack: only layers {stats['layers']} carry affine experts")
    for err in stats["index_errors"]:
        print(f"INDEX ERROR: {err}")
    return 1 if stats["index_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
