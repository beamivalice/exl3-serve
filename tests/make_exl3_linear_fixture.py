#!/usr/bin/env python3
"""Generate an EXL3 linear fixture for src/fixtures/ from PonyExl3.

The trellis is packed by PonyExl3 (the converter side of the format); `inner`
and `public` come from PonyExl3's reference decode, reduced in the exact order
of `expert_exl3.reconstructPublic` so the Zig test can assert byte identity.

  python tests/make_exl3_linear_fixture.py --k 2.5 --codebook tiny \
      --out src/fixtures/exl3_k2p5_tiny_linear.safetensors
  python tests/make_exl3_linear_fixture.py --k 2.5 --codebook tiny --window 12 \
      --out src/fixtures/exl3_k2p5_tiny_w12_linear.safetensors
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

HAD = 128


def had_matrix() -> np.ndarray:
    idx = np.arange(HAD)
    parity = np.array([[bin(r & c).count("1") & 1 for c in idx] for r in idx])
    return np.where(parity == 0, 1.0, -1.0).astype(np.float32) * np.float32(1.0 / math.sqrt(HAD))


def serve_public(inner: np.ndarray, suh: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """reconstructPublic's reduction order: one rounded multiply-add per k."""
    h = had_matrix()
    w = inner.astype(np.float32)
    rows, cols = w.shape
    for rb in range(0, rows, HAD):
        blk = w[rb : rb + HAD]
        acc = np.zeros_like(blk)
        for k in range(HAD):
            acc += h[:, k][:, None] * blk[k][None, :]
        w[rb : rb + HAD] = acc
    w *= suh.astype(np.float32)[:, None]
    for cb in range(0, cols, HAD):
        blk = w[:, cb : cb + HAD]
        acc = np.zeros_like(blk)
        for k in range(HAD):
            acc += blk[:, k][:, None] * h[:, k][None, :]
        w[:, cb : cb + HAD] = acc
    w *= svh.astype(np.float32)[None, :]
    return w.astype(np.float16)


def write_safetensors(
    path: str, tensors: dict[str, np.ndarray], metadata: dict[str, str]
) -> None:
    header: dict[str, object] = {"__metadata__": metadata}
    blob = bytearray()
    for name, arr in tensors.items():
        dtype = {np.dtype(np.uint16): "U16", np.dtype(np.float16): "F16"}[arr.dtype]
        raw = arr.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [len(blob), len(blob) + len(raw)],
        }
        blob += raw
    head = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(head)) % 8
    head += b" " * pad
    with open(path, "wb") as f:
        f.write(len(head).to_bytes(8, "little"))
        f.write(head)
        f.write(bytes(blob))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--codebook", choices=["mul1", "mcg", "tiny"], required=True)
    # The codeword width the search hashes. The bitstream is the same width at
    # every window; only the value each codeword decodes to changes, so a pack
    # and its reference decode must name the same one.
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--in-features", type=int, default=128)
    ap.add_argument("--out-features", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pony", default="/Users/beam/llm/ponyexl3")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.pony)
    from ponyexl3.convert.metal_search import quantize_tiles_mlx
    from ponyexl3.ref.codebook import CodebookMode
    from ponyexl3.ref.perm import tensor_core_perm
    from ponyexl3.ref.reconstruct import reconstruct_inner
    from ponyexl3.ref.trellis import pack_trellis, packed_halfwords

    k = int(args.k) if float(args.k).is_integer() else args.k
    mode = {"mul1": CodebookMode.MUL1, "mcg": CodebookMode.MCG, "tiny": CodebookMode.TINY}[
        args.codebook
    ]
    rng = np.random.default_rng(args.seed)
    in_f, out_f = args.in_features, args.out_features
    in_tiles, out_tiles = in_f // 16, out_f // 16

    # Unit variance: the trellis codebooks are unit-scaled, and EXL3 feeds them
    # a Hadamard-rotated block whose scale the converter has already removed.
    target = rng.standard_normal((in_f, out_f)).astype(np.float32)
    perm = tensor_core_perm()
    tiles = np.stack(
        [
            target[tk * 16 : tk * 16 + 16, tn * 16 : tn * 16 + 16].reshape(256)[perm]
            for tk in range(in_tiles)
            for tn in range(out_tiles)
        ]
    )
    _, states = quantize_tiles_mlx(tiles, k, mode, window=args.window)
    fresh = np.array(states).astype(np.uint16, copy=False)
    packed = pack_trellis(fresh.reshape(in_tiles, out_tiles, 256), k)
    assert packed.shape == (in_tiles, out_tiles, packed_halfwords(k)), packed.shape

    signs = np.where(rng.random(in_f) > 0.5, -1.0, 1.0)
    suh = (signs * (0.7 + 0.35 * rng.random(in_f))).astype(np.float16)
    signs = np.where(rng.random(out_f) > 0.5, -1.0, 1.0)
    svh = (signs * (0.7 + 0.35 * rng.random(out_f))).astype(np.float16)

    flags = {args.codebook: True}
    inner = reconstruct_inner(np.asarray(packed), k, window=args.window, **flags)
    public = serve_public(inner, suh, svh)
    write_safetensors(
        args.out,
        {
            "inner": inner,
            "public": public,
            "suh": suh,
            "svh": svh,
            "trellis": np.asarray(packed, dtype=np.uint16),
        },
        {"k": str(k), "codebook": args.codebook, "window": str(args.window)},
    )
    print(
        f"{args.out}: K={k} n={packed_halfwords(k)} codebook={args.codebook} "
        f"window={args.window}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
