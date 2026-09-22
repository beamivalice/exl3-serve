#!/usr/bin/env python3
"""Per-layer expert width allocation for a MiMo-V2.6-Flash affine pack.

Same two phases as tests/qwen38_flash_next_iq_allocate.py, against the MXFP4
source checkpoint: routed experts are stored per expert as
`model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.{weight,weight_scale}`
(e2m1 nibbles + e8m0 block-32 scales), so every measured matrix is dequantized
from those bytes first. Layer 0 is dense and owns no experts.

  measure   For every (layer, gate_up|down) group: quantize a random sample of
            experts at each candidate (bits, group_size) with the imatrix-
            weighted search (dsv4_imatrix.weighted_affine_quant) and record the
            weighted RELATIVE reconstruction error, plus the per-candidate
            seconds a full-bank conversion would be priced from.
  allocate  Spend an expert byte budget greedily from the floor, best
            error-reduction-per-byte first, with `--tail-layers` pinned 4x64.

  python3 tests/mimo_v26_iq_allocate.py measure --src <hf dir> \
      --imatrix im.safetensors --out errors.json
  python3 tests/mimo_v26_iq_allocate.py allocate --errors errors.json \
      --budget-gb 107.25 --out alloc.json
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv4_imatrix import weighted_affine_quant  # noqa: E402

CANDIDATES = ((2, 64), (2, 128), (3, 64), (3, 128), (4, 64))
PREFIX = "model.layers."
BLOCK = 32
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def ckey(bits, gs):
    return f"{bits}x{gs}"


def bytes_per_param(bits, gs):
    return (bits * 8 + 256 // gs) / 64.0   # packed bits + bf16 scale/bias per group, in bytes


def e2m1_value(codes):
    c = np.asarray(codes, dtype=np.uint8)
    mag = E2M1[c & 0x7]
    return np.where((c & 0x8) != 0, -mag, mag).astype(np.float32)


def e8m0_value(codes):
    c = np.asarray(codes, dtype=np.uint8).astype(np.int32)
    with np.errstate(over="ignore"):
        out = np.exp2((c - 127).astype(np.float32))
    out = np.where(c == 0, np.float32(0.0), out)
    return np.where(c == 255, np.float32(np.inf), out).astype(np.float32)


def dequant_mxfp4(weight, scale, block=BLOCK):
    """MXFP4 [out, in/2] + [out, in/block] -> float32 [out, in]."""
    w = np.asarray(weight, dtype=np.uint8)
    s = np.asarray(scale, dtype=np.uint8)
    out_rows, half = w.shape
    in_dim = half * 2
    assert s.shape == (out_rows, in_dim // block), f"scale {s.shape} vs {w.shape}"
    nibbles = np.empty((out_rows, in_dim), dtype=np.uint8)
    nibbles[:, 0::2] = w & 0x0F
    nibbles[:, 1::2] = w >> 4
    return e2m1_value(nibbles) * np.repeat(e8m0_value(s), block, axis=1)


def raw_memmap(src, index, name):
    path = Path(src) / index["weight_map"][name]
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    meta = hdr[name]
    assert meta["dtype"] == "U8", f"{name}: {meta['dtype']}"
    b, _ = meta["data_offsets"]
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8 + hlen + b,
                     shape=tuple(meta["shape"]))


def expert_f32(src, index, layer, expert, proj):
    stem = f"{PREFIX}{layer}.mlp.experts.{expert}.{proj}_proj"
    return dequant_mxfp4(raw_memmap(src, index, stem + ".weight"),
                         raw_memmap(src, index, stem + ".weight_scale"))


_G = {}


def _measure(task):
    layer, role, experts = task
    name = f"{PREFIX}{layer}.mlp.experts.{'gate_up_proj' if role == 'gate_up' else 'down_proj'}"
    ch_all = _G["im"][name]
    E = _G["n_experts"]
    in_dim = ch_all.shape[0] // E
    assert ch_all.shape == (E * in_dim,)
    cands = [c for c in CANDIDATES if in_dim % c[1] == 0]
    err = {ckey(*c): 0.0 for c in cands}
    secs = {ckey(*c): 0.0 for c in cands}
    zero_ch, nonfinite = 0, 0
    out_dim = None
    for e in experts:
        if role == "gate_up":
            w = np.concatenate([expert_f32(_G["src"], _G["index"], layer, e, p)
                                for p in ("gate", "up")], axis=0)
        else:
            w = expert_f32(_G["src"], _G["index"], layer, e, "down")
        out_dim = w.shape[0]
        nonfinite += int((~np.isfinite(w)).sum())
        ch = ch_all[e * in_dim:(e + 1) * in_dim]
        zero_ch += int((ch == 0).sum())
        for bits, gs in cands:
            t0 = time.time()
            _, st = weighted_affine_quant(w, bits, gs, ch, return_stats=True)
            secs[ckey(bits, gs)] += time.time() - t0
            err[ckey(bits, gs)] += st["weighted_rel_err"] / len(experts)
    return f"layers.{layer}.{role}", {
        "err": err, "secs_per_expert": {k: v / len(experts) for k, v in secs.items()},
        "params": int(E * out_dim * in_dim), "zero_ch": zero_ch, "nonfinite": nonfinite}


def cmd_measure(args):
    index = json.loads((Path(args.src) / "model.safetensors.index.json").read_text())
    cfg = json.loads((Path(args.src) / "config.json").read_text())
    n_layers, n_experts = cfg["num_hidden_layers"], cfg["n_routed_experts"]
    from safetensors.numpy import load_file
    _G["src"], _G["index"] = args.src, index
    _G["im"], _G["n_experts"] = load_file(args.imatrix), n_experts

    never = {}
    for l in range(1, n_layers):
        rows = _G["im"][f"{PREFIX}{l}.mlp.experts.gate_up_proj.rows"]
        n0 = int((rows == 0).sum())
        if n0:
            never[l] = n0
    print(f"never-routed experts: {never or 'none'}", flush=True)

    rng = random.Random(args.seed)
    tasks = [(l, role, sorted(rng.sample(range(n_experts), args.experts)))
             for l in range(1, n_layers) for role in ("gate_up", "down")]
    out, t0 = {}, time.time()
    with mp.get_context("fork").Pool(args.jobs) as pool:
        for i, (k, rec) in enumerate(pool.imap_unordered(_measure, tasks)):
            out[k] = rec
            print(f"[{i+1}/{len(tasks)}] {k} "
                  + " ".join(f"{c}={v:.4f}" for c, v in rec["err"].items())
                  + (f" zero_ch={rec['zero_ch']}" if rec["zero_ch"] else "")
                  + (f" NONFINITE={rec['nonfinite']}" if rec["nonfinite"] else ""), flush=True)
    wall = time.time() - t0
    print(f"wall {wall:.1f}s on {args.jobs} jobs", flush=True)
    Path(args.out).write_text(json.dumps(
        {"experts_sampled": args.experts, "seed": args.seed, "jobs": args.jobs,
         "wall_secs": wall, "n_experts": n_experts, "groups": out}, indent=1))


def cmd_allocate(args):
    doc = json.loads(Path(args.errors).read_text())
    groups = doc["groups"]
    layers = sorted({int(k.split(".")[1]) for k in groups})
    n_layers = 1 + layers[-1]
    pinned = {k for k in groups if int(k.split(".")[1]) >= n_layers - args.tail_layers}
    total_params = sum(g["params"] for g in groups.values())

    def cost(k, c):
        return groups[k]["params"] * bytes_per_param(*c)

    def err(k, c):
        return groups[k]["err"][ckey(*c)] * groups[k]["params"]   # error weighted by size

    for c in CANDIDATES:
        if all(ckey(*c) in g["err"] for g in groups.values()):
            b = sum(cost(k, c) for k in groups)
            e = sum(err(k, c) for k in groups) / total_params
            print(f"uniform {ckey(*c)}: {b/1e9:.2f} GB  rel err {e:.5f}")

    def floor_for(k):
        if k.endswith(".down") and args.down_floor:
            b, g = args.down_floor.split("x")
            return int(b), int(g)
        return (2, 128) if ckey(2, 128) in groups[k]["err"] else (2, 64)

    state = {k: ((4, 64) if k in pinned else floor_for(k)) for k in groups}
    spent = sum(cost(k, c) for k, c in state.items())
    budget = args.budget_gb * 1e9
    while True:
        best = None
        for k in groups:
            if k in pinned:
                continue
            cur = state[k]
            for c in CANDIDATES:
                if ckey(*c) not in groups[k]["err"]:
                    continue
                dc = cost(k, c) - cost(k, cur)
                if dc <= 0 or spent + dc > budget:
                    continue
                gain = (err(k, cur) - err(k, c)) / dc
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, k, c)
        if best is None:
            break
        _, k, c = best
        spent += cost(k, c) - cost(k, state[k])
        state[k] = c
    total_err = sum(err(k, c) for k, c in state.items()) / total_params
    print(f"spent {spent/1e9:.2f} GB of {args.budget_gb} GB "
          f"({spent * 8 / total_params:.3f} bpw), size-weighted rel err {total_err:.5f}")
    print("widths:", dict(Counter(ckey(*c) for c in state.values())))
    for l in layers:
        print(f"  layer {l:2d}: gate_up {ckey(*state[f'layers.{l}.gate_up'])}"
              f"  down {ckey(*state[f'layers.{l}.down'])}")
    Path(args.out).write_text(json.dumps(
        {k: {"bits": c[0], "group_size": c[1]} for k, c in sorted(state.items())}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--src", required=True)
    m.add_argument("--imatrix", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--experts", type=int, default=16)
    m.add_argument("--seed", type=int, default=20260922)
    m.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    m.set_defaults(fn=cmd_measure)
    a = sub.add_parser("allocate")
    a.add_argument("--errors", required=True)
    a.add_argument("--budget-gb", type=float, required=True, help="expert bytes incl. the pinned tail")
    a.add_argument("--tail-layers", type=int, default=2)
    a.add_argument("--down-floor", default=None, help="e.g. 3x128: never below this on the down projections")
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_allocate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
