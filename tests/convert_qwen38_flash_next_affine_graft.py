#!/usr/bin/env python3
"""Graft fresh affine routed experts onto an existing Qwen3.8 MLX pack.

The donor supplies the non-expert trunk, MTP trunk, vision weights, tokenizer
files, runtime config, and raw BF16 n-gram table.  Only the routed expert
banks are read from the original BF16 checkpoint and quantized with the
existing MLX CPU affine helper.  The destination must not exist.

Example:

    /Users/beam/llm/ponyexl3/.venv/bin/python \
      tests/convert_qwen38_flash_next_affine_graft.py \
      --src /Users/beam/llm/models/Qwen/Qwen3.8-Flash-Next \
      --donor /Users/beam/llm/models/Qwen3.8-Flash-Next-EXL3-K3 \
      --dst /Users/beam/llm/models/Qwen3.8-Flash-Next-Affine3-G128

The route quantizer works in bounded expert batches.  `--batch-experts 64`
keeps the large BF16 source array resident while keeping MLX CPU temporaries
well below a whole 512-expert bank.
"""

import argparse
import copy
import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_qwen38_flash_next as affine  # noqa: E402
from convert_dsv4_weights import mx as mlx_runtime  # noqa: E402


AFFINE_SUFFIXES = ("weight", "scales", "biases")
SOURCE_GATE_UP = "gate_up_proj"
SOURCE_DOWN = "down_proj"
_TRUNK_SOURCE_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|down_proj)$"
)
_MTP_SOURCE_RE = re.compile(
    r"^mtp\.layers\.0\.mlp\.experts\.(gate_up_proj|down_proj)$"
)


def destination_guard(dst: str | Path) -> Path:
    """Refuse an existing destination, including a dangling symlink."""
    path = Path(dst).expanduser()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"destination already exists: {path}")
    return path


def _route_base(kind: str, layer: int) -> str:
    if kind == "trunk":
        return f"language_model.model.layers.{layer}.mlp.switch_mlp"
    if kind == "mtp":
        if layer != 0:
            raise ValueError(f"unsupported MTP layer {layer}")
        return "language_model.mtp.layers.0.mlp.switch_mlp"
    raise ValueError(f"unknown route kind {kind!r}")


def expected_route_specs(
    n_layers: int, *, bits: int, group_size: int
) -> dict[str, dict[str, int | str]]:
    """Return the loader config for every routed projection."""
    spec = {"bits": bits, "group_size": group_size, "mode": "affine"}
    out = {}
    for layer in range(n_layers):
        base = _route_base("trunk", layer)
        for role in ("gate", "up", "down"):
            out[f"{base}.{role}_proj"] = dict(spec)
    base = _route_base("mtp", 0)
    for role in ("gate", "up", "down"):
        out[f"{base}.{role}_proj"] = dict(spec)
    return out


def build_affine_config(
    donor_config: dict, n_layers: int, *, bits: int, group_size: int
) -> dict:
    """Replace EXL3 routing metadata while retaining donor runtime metadata."""
    cfg = copy.deepcopy(donor_config)
    cfg.pop("expert_quant", None)
    route_specs = expected_route_specs(
        n_layers, bits=bits, group_size=group_size
    )
    for block_name in ("quantization", "quantization_config"):
        block = copy.deepcopy(cfg.get(block_name) or {})
        # A donor may already carry a per-module route block.  It describes
        # the old payload and must not survive beside the fresh affine bank.
        for key in list(block):
            if ".switch_mlp." in key:
                block.pop(key)
        block["calibration"] = "uncalibrated affine from BF16 source"
        block.update(copy.deepcopy(route_specs))
        cfg[block_name] = block
    return cfg


def _source_route_info(name: str, n_layers: int) -> tuple[str, int, str] | None:
    match = _TRUNK_SOURCE_RE.fullmatch(name)
    if match:
        layer, role = int(match.group(1)), match.group(2)
        if layer >= n_layers:
            raise RuntimeError(f"source route layer outside config: {name}")
        return "trunk", layer, role
    match = _MTP_SOURCE_RE.fullmatch(name)
    if match:
        return "mtp", 0, match.group(1)
    return None


def _source_route_name(kind: str, layer: int, role: str) -> str:
    if kind == "trunk":
        return f"model.language_model.layers.{layer}.mlp.experts.{role}"
    if kind == "mtp" and layer == 0:
        return f"mtp.layers.0.mlp.experts.{role}"
    raise ValueError((kind, layer, role))


def _expected_donor_route_keys(n_layers: int) -> set[str]:
    out = set()
    for base in expected_route_specs(n_layers, bits=3, group_size=128):
        for suffix in ("trellis", "suh", "svh"):
            out.add(f"{base}.{suffix}")
    return out


def _read_index(path: Path) -> dict:
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")
    return json.loads(index_path.read_text())


def _validate_source_index(src: Path, index: dict, n_layers: int) -> dict[str, str]:
    weight_map = index.get("weight_map") or {}
    expected = {
        _source_route_name(kind, layer, role)
        for kind, count in (("trunk", n_layers), ("mtp", 1))
        for layer in range(count)
        for role in (SOURCE_GATE_UP, SOURCE_DOWN)
    }
    actual = {
        name for name in weight_map if _source_route_info(name, n_layers) is not None
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"BF16 source routed tensor set mismatch: missing={missing}, extra={extra}"
        )
    for name in expected:
        if not (src / weight_map[name]).is_file():
            raise FileNotFoundError(f"source shard for {name}: {src / weight_map[name]}")
    return {name: weight_map[name] for name in expected}


def _validate_donor_index(donor: Path, index: dict, n_layers: int) -> set[str]:
    weight_map = index.get("weight_map") or {}
    actual = {name for name in weight_map if ".switch_mlp." in name}
    expected = _expected_donor_route_keys(n_layers)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"EXL3 donor routed tensor set mismatch: missing={missing}, extra={extra}"
        )
    for name in expected:
        if not (donor / weight_map[name]).is_file():
            raise FileNotFoundError(f"donor shard for {name}: {donor / weight_map[name]}")
    return actual


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link immutable donor payloads, with a cross-device fallback."""
    try:
        os.link(src, dst)
    except OSError as exc:
        if getattr(exc, "errno", None) != 18:  # EXDEV
            raise
        shutil.copy2(src, dst)


def _copy_donor_payload(
    donor: Path,
    dst: Path,
    donor_index: dict,
    donor_route_keys: set[str],
) -> tuple[dict[str, str], set[str]]:
    """Link donor files that contain no routed tensors."""
    weight_map = donor_index["weight_map"]
    route_files = {weight_map[name] for name in donor_route_keys}
    skipped = {
        "config.json",
        "model.safetensors.index.json",
        "README.md",
    }
    for source in sorted(donor.iterdir()):
        if not source.is_file() or source.name in skipped:
            continue
        if source.name in route_files:
            continue
        _link_or_copy(source, dst / source.name)
    out_map = {
        name: filename
        for name, filename in weight_map.items()
        if name not in donor_route_keys
    }
    return out_map, route_files


def _read_nonroute_tensors(
    path: Path, route_keys: set[str]
) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    """Read only non-route tensors from the donor's mixed MTP shard."""
    header, data_offset = affine.read_header(str(path))
    out = {}
    for name, meta in header.items():
        if name in route_keys:
            continue
        raw = affine.read_raw(str(path), data_offset, meta)
        out[name] = (
            meta["dtype"],
            tuple(meta["shape"]),
            np.ascontiguousarray(raw).tobytes(),
        )
    return out


def _quant_bank(
    arr_u16: np.ndarray,
    bits: int,
    group_size: int,
    batch_experts: int,
) -> tuple[tuple[str, tuple[int, ...], bytes], ...]:
    """Quantize a 3-D [expert, out, in] bank in bounded batches."""
    if arr_u16.ndim != 3:
        raise RuntimeError(f"expert bank must be 3-D, got {arr_u16.shape}")
    experts = arr_u16.shape[0]
    if experts == 0:
        raise RuntimeError("expert bank is empty")
    chunks: list[list[bytes]] = [[], [], []]
    shapes: list[tuple[int, ...] | None] = [None, None, None]
    for start in range(0, experts, batch_experts):
        stop = min(start + batch_experts, experts)
        batch = np.ascontiguousarray(arr_u16[start:stop])
        triples = affine.quant(batch, bits, group_size)
        for i, (dtype, shape, raw) in enumerate(triples):
            if shapes[i] is None:
                shapes[i] = tuple(shape[1:])
            elif tuple(shape[1:]) != shapes[i]:
                raise RuntimeError(f"quantized bank shape changed at batch {start}")
            chunks[i].append(raw)
        del batch, triples
        gc.collect()
    assert all(shape is not None for shape in shapes)
    return tuple(
        (
            ("U32" if i == 0 else "BF16"),
            (experts,) + shapes[i],  # type: ignore[operator]
            b"".join(chunks[i]),
        )
        for i in range(3)
    )


def _check_source_shape(
    arr: np.ndarray,
    kind: str,
    role: str,
    expert_count: int,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    expected = (
        (expert_count, intermediate_size * 2, hidden_size)
        if role == SOURCE_GATE_UP
        else (expert_count, hidden_size, intermediate_size)
    )
    if tuple(arr.shape) != expected:
        raise RuntimeError(
            f"{kind} {role} BF16 geometry {tuple(arr.shape)} != expected {expected}"
        )


def _convert_route_unit(
    src: Path,
    source_weight_map: dict[str, str],
    kind: str,
    layer: int,
    *,
    bits: int,
    group_size: int,
    batch_experts: int,
    expert_count: int,
    hidden_size: int,
    intermediate_size: int,
) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    """Read and freshly quantize one trunk/MTP routed layer."""
    gate_name = _source_route_name(kind, layer, SOURCE_GATE_UP)
    down_name = _source_route_name(kind, layer, SOURCE_DOWN)
    gate_path = src / source_weight_map[gate_name]
    down_path = src / source_weight_map[down_name]

    gate_header, gate_offset = affine.read_header(str(gate_path))
    gate_meta = gate_header[gate_name]
    if gate_meta["dtype"] != "BF16":
        raise RuntimeError(f"{gate_name} is {gate_meta['dtype']}, expected BF16")
    gate_arr = affine.read_raw(str(gate_path), gate_offset, gate_meta)
    _check_source_shape(
        gate_arr,
        kind,
        SOURCE_GATE_UP,
        expert_count,
        hidden_size,
        intermediate_size,
    )
    split = gate_arr.shape[1] // 2
    gate = _quant_bank(
        gate_arr[:, :split], bits, group_size, batch_experts
    )
    up = _quant_bank(
        gate_arr[:, split:], bits, group_size, batch_experts
    )
    del gate_arr
    gc.collect()

    down_header, down_offset = affine.read_header(str(down_path))
    down_meta = down_header[down_name]
    if down_meta["dtype"] != "BF16":
        raise RuntimeError(f"{down_name} is {down_meta['dtype']}, expected BF16")
    down_arr = affine.read_raw(str(down_path), down_offset, down_meta)
    _check_source_shape(
        down_arr,
        kind,
        SOURCE_DOWN,
        expert_count,
        hidden_size,
        intermediate_size,
    )
    down = _quant_bank(down_arr, bits, group_size, batch_experts)
    del down_arr
    gc.collect()

    base = _route_base(kind, layer)
    out: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for projection, triples in (
        ("gate_proj", gate),
        ("up_proj", up),
        ("down_proj", down),
    ):
        for suffix, triple in zip(AFFINE_SUFFIXES, triples):
            out[f"{base}.{projection}.{suffix}"] = triple
    return out


def _write_route_shard(
    dst: Path,
    filename: str,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> dict[str, str]:
    affine.write_safetensors_raw(str(dst / filename), tensors)
    return {name: filename for name in tensors}


def _expected_route_shape(
    name: str,
    *,
    expert_count: int,
    hidden_size: int,
    intermediate_size: int,
    bits: int,
    group_size: int,
) -> tuple[str, tuple[int, ...]]:
    suffix = name.rsplit(".", 1)[-1]
    module = name[: -(len(suffix) + 1)]
    projection = module.rsplit(".", 1)[-1]
    if projection in ("gate_proj", "up_proj"):
        out_dim, in_dim = intermediate_size, hidden_size
    elif projection == "down_proj":
        out_dim, in_dim = hidden_size, intermediate_size
    else:
        raise RuntimeError(f"not an affine routed projection: {name}")
    if suffix == "weight":
        shape = (expert_count, out_dim, in_dim * bits // 32)
        dtype = "U32"
    else:
        shape = (expert_count, out_dim, in_dim // group_size)
        dtype = "BF16"
    return dtype, shape


def audit_pack(
    dst: Path,
    index: dict,
    config: dict,
    *,
    n_layers: int,
    bits: int,
    group_size: int,
) -> dict:
    """Audit index coverage, affine route headers, and completion metadata."""
    weight_map = index.get("weight_map") or {}
    if "expert_quant" in config:
        raise RuntimeError("EXL3 expert_quant remains in output config")
    route_specs = expected_route_specs(
        n_layers, bits=bits, group_size=group_size
    )
    route_keys = {
        f"{module}.{suffix}"
        for module in route_specs
        for suffix in AFFINE_SUFFIXES
    }
    if not route_keys.issubset(weight_map):
        raise RuntimeError(
            f"output index is missing routed keys: {sorted(route_keys - set(weight_map))}"
        )
    file_headers: dict[str, dict] = {}
    for filename in sorted(set(weight_map.values())):
        path = dst / filename
        if not path.is_file():
            raise RuntimeError(f"index names missing shard: {path}")
        file_headers[filename], _ = affine.read_header(str(path))
    for name in route_keys:
        filename = weight_map[name]
        meta = file_headers[filename].get(name)
        if meta is None:
            raise RuntimeError(f"routed key absent from named shard: {name}")
        dtype, shape = _expected_route_shape(
            name,
            expert_count=int(config["text_config"]["num_experts"]),
            hidden_size=int(config["text_config"]["hidden_size"]),
            intermediate_size=int(config["text_config"]["moe_intermediate_size"]),
            bits=bits,
            group_size=group_size,
        )
        if meta["dtype"] != dtype or tuple(meta["shape"]) != shape:
            raise RuntimeError(
                f"routed header {name}: {(meta['dtype'], meta['shape'])} != {(dtype, shape)}"
            )
    exl3_names = {
        name
        for name in weight_map
        if name.endswith((".trellis", ".suh", ".svh"))
    }
    if exl3_names:
        raise RuntimeError(f"EXL3 routed keys remain: {sorted(exl3_names)}")
    total = sum((dst / filename).stat().st_size for filename in set(weight_map.values()))
    if int(index.get("metadata", {}).get("total_size", -1)) != total:
        raise RuntimeError(
            f"index total_size {index.get('metadata', {}).get('total_size')} != {total}"
        )
    return {
        "tensors": len(weight_map),
        "shards": len(set(weight_map.values())),
        "routed_tensors": len(route_keys),
        "total_bytes": total,
    }


def _write_readme(dst: Path, *, bits: int, group_size: int) -> None:
    (dst / "README.md").write_text(
        f"""# Qwen3.8-Flash-Next affine expert graft

This mlx-serve pack keeps the immutable non-expert weights and raw BF16
`ngram_table.bin` from the donor pack. All routed trunk and MTP experts were
freshly quantized from `Qwen/Qwen3.8-Flash-Next` BF16 with plain affine
{bits}-bit, group-{group_size} quantization on the MLX CPU path.

The routed quantization is deliberately **uncalibrated**: no activation
calibration or imatrix was used. The donor's runtime context, RoPE, vision,
tokenizer, and remaining quantization metadata are retained.
"""
    )


def convert(args: argparse.Namespace) -> dict:
    src = Path(args.src).expanduser()
    donor = Path(args.donor).expanduser()
    dst = destination_guard(args.dst)
    if not src.is_dir():
        raise NotADirectoryError(src)
    if not donor.is_dir():
        raise NotADirectoryError(donor)
    if args.bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError(f"unsupported affine bits {args.bits}")
    if args.expert_gs <= 0 or args.batch_experts <= 0:
        raise ValueError("expert-gs and batch-experts must be positive")
    for name, value in (
        ("VECLIB_MAXIMUM_THREADS", args.cpu_threads),
        ("OMP_NUM_THREADS", args.cpu_threads),
        ("OPENBLAS_NUM_THREADS", args.cpu_threads),
        ("MKL_NUM_THREADS", args.cpu_threads),
    ):
        os.environ[name] = str(value)

    src_cfg = json.loads((src / "config.json").read_text())
    donor_cfg = json.loads((donor / "config.json").read_text())
    src_text = src_cfg.get("text_config", src_cfg)
    donor_text = donor_cfg.get("text_config", donor_cfg)
    n_layers = int(donor_text["num_hidden_layers"])
    if int(src_text["num_hidden_layers"]) != n_layers:
        raise RuntimeError("BF16 source and donor layer counts differ")
    for key in ("num_experts", "hidden_size", "moe_intermediate_size"):
        if int(src_text[key]) != int(donor_text[key]):
            raise RuntimeError(f"BF16 source and donor {key} differ")

    src_index = _read_index(src)
    donor_index = _read_index(donor)
    source_weight_map = _validate_source_index(src, src_index, n_layers)
    donor_route_keys = _validate_donor_index(donor, donor_index, n_layers)
    dst.mkdir(parents=False)
    out_map, route_files = _copy_donor_payload(
        donor, dst, donor_index, donor_route_keys
    )
    if "model-mtp.safetensors" not in route_files:
        raise RuntimeError("donor MTP routed tensors are not in model-mtp.safetensors")

    # mlx_affine_quant() sets the default device to CPU.  Touch the lazy
    # runtime here so the conversion cannot accidentally inherit a GPU device.
    m = mlx_runtime()
    m.set_default_device(m.cpu)
    print(
        f"[affine-graft] CPU quantization bits={args.bits} gs={args.expert_gs} "
        f"batch_experts={args.batch_experts} threads={args.cpu_threads}",
        flush=True,
    )
    t0 = time.time()
    expert_count = int(donor_text["num_experts"])
    hidden_size = int(donor_text["hidden_size"])
    intermediate_size = int(donor_text["moe_intermediate_size"])

    for layer in range(n_layers):
        tensors = _convert_route_unit(
            src,
            source_weight_map,
            "trunk",
            layer,
            bits=args.bits,
            group_size=args.expert_gs,
            batch_experts=args.batch_experts,
            expert_count=expert_count,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        filename = f"model-experts-L{layer:02d}.safetensors"
        out_map.update(_write_route_shard(dst, filename, tensors))
        print(
            f"[affine-graft] layer {layer + 1}/{n_layers} wrote {filename} "
            f"({(time.time() - t0) / 60:.1f} min)",
            flush=True,
        )

    mtp_tensors = _convert_route_unit(
        src,
        source_weight_map,
        "mtp",
        0,
        bits=args.bits,
        group_size=args.expert_gs,
        batch_experts=args.batch_experts,
        expert_count=expert_count,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    mtp_path = donor / "model-mtp.safetensors"
    mtp_named = _read_nonroute_tensors(mtp_path, donor_route_keys)
    mtp_named.update(mtp_tensors)
    out_map.update(
        _write_route_shard(dst, "model-mtp.safetensors", mtp_named)
    )
    print(
        f"[affine-graft] MTP route grafted ({(time.time() - t0) / 60:.1f} min)",
        flush=True,
    )

    cfg = build_affine_config(
        donor_cfg, n_layers, bits=args.bits, group_size=args.expert_gs
    )
    _write_readme(dst, bits=args.bits, group_size=args.expert_gs)

    # These are intentionally the final two files.  A directory without them
    # is an incomplete pack and cannot be mistaken for a ready model.
    total = sum((dst / filename).stat().st_size for filename in set(out_map.values()))
    cfg_path = dst / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    index = {
        "metadata": {"total_size": total},
        "weight_map": dict(sorted(out_map.items())),
    }
    idx_path = dst / "model.safetensors.index.json"
    idx_path.write_text(json.dumps(index, indent=2) + "\n")
    audit = audit_pack(
        dst,
        index,
        cfg,
        n_layers=n_layers,
        bits=args.bits,
        group_size=args.expert_gs,
    )
    print(
        f"[affine-graft] header audit: {audit['routed_tensors']} routed tensors, "
        f"{audit['tensors']} indexed tensors, {audit['shards']} shards, "
        f"{audit['total_bytes'] / 1e9:.2f} GB indexed",
        flush=True,
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="original BF16 HF checkpoint")
    parser.add_argument("--donor", required=True, help="existing MLX donor pack")
    parser.add_argument("--dst", required=True, help="new destination; must be absent")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--expert-gs", type=int, default=128)
    parser.add_argument(
        "--batch-experts",
        type=int,
        default=64,
        help="expert rows per MLX CPU quantization call",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=min(4, os.cpu_count() or 4),
        help="thread cap for BLAS/Accelerate helpers",
    )
    args = parser.parse_args()
    convert(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
