#!/usr/bin/env python3
"""Byte-preserving Qwen4 EXL3 component packs, with optional verified hardlinks.

Inputs must remain immutable during repacking. Outputs never replace an existing
directory. Hardlinked files must be replaced, not modified in place.
"""
import argparse
from collections import defaultdict
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile


INDEX = "model.safetensors.index.json"
CHUNK = 8 << 20
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8,
}
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def flat_name(name):
    if not isinstance(name, str) or name in ("", ".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"expected a flat filename, got {name!r}")
    return name


def fingerprint(path):
    s = path.stat()
    # Adding a hardlink changes ctime without changing the file's contents.
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def catalog(root):
    """Validate indexed shards and return byte ranges without materializing tensors."""
    index = json.loads((root / INDEX).read_text(), object_pairs_hook=unique_object)
    wm = index["weight_map"]
    if not isinstance(wm, dict) or not wm:
        raise ValueError("empty or invalid weight_map")
    tensors, stamps = {}, {}
    for filename in sorted(set(wm.values())):
        path = root / flat_name(filename)
        stamps[path] = fingerprint(path)
        size = path.stat().st_size
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(f"truncated safetensors header: {path}")
            n, = struct.unpack("<Q", raw)
            if n > min(64 << 20, size - 8):
                raise ValueError(f"invalid safetensors header length: {path}")
            header = json.loads(f.read(n), object_pairs_hook=unique_object)
        cursor = 0
        entries = [(key, value) for key, value in header.items() if key != "__metadata__"]
        for key, info in sorted(entries, key=lambda item: item[1]["data_offsets"]):
            start, end = info["data_offsets"]
            shape, dtype = info["shape"], info["dtype"]
            if (not all(type(x) is int and x >= 0 for x in shape)
                    or type(start) is not int or type(end) is not int
                    or start != cursor or end < start or end > size - 8 - n
                    or dtype not in DTYPE_BYTES
                    or end - start != math.prod(shape) * DTYPE_BYTES[dtype]):
                raise ValueError(f"invalid tensor geometry: {path}: {key}")
            if key in tensors or wm.get(key) != filename:
                raise ValueError(f"index/header mismatch: {path}: {key}")
            tensors[key] = {
                "path": path, "offset": 8 + n + start, "size": end - start,
                "dtype": dtype, "shape": shape,
            }
            cursor = end
        if cursor != size - 8 - n:
            raise ValueError(f"unaccounted safetensors payload: {path}")
    if set(tensors) != set(wm):
        raise ValueError("index names tensors absent from shard headers")
    return tensors, stamps


def plan_shards(tensors):
    groups, trunk = defaultdict(list), defaultdict(list)
    for key in sorted(tensors):
        layer = LAYER.search(key)
        if key.startswith("mtp.") or ".mtp." in key:
            name = "model-mtp.safetensors"
        elif key.startswith("visual.") or ".visual." in key:
            name = "model-vision.safetensors"
        elif ".embed_tokens." in key:
            name = "model-embed.safetensors"
        elif key.startswith("lm_head.") or ".lm_head." in key:
            name = "model-lm-head.safetensors"
        elif ".mlp.switch_mlp." in key:
            if layer is None:
                raise ValueError(f"expert tensor has no layer: {key}")
            name = f"model-experts-L{int(layer[1]):02}.safetensors"
        else:
            trunk[int(layer[1]) if layer else -1].append(key)
            continue
        groups[name].append(key)
    units = [trunk[layer] for layer in sorted(trunk)]
    if units:
        sizes = [sum(tensors[key]["size"] for key in unit) for unit in units]
        # A layer is indivisible; choose the nearest boundary to half the bytes.
        split = min(range(1, len(units)), key=lambda i: abs(2 * sum(sizes[:i]) - sum(sizes))) if len(units) > 1 else 1
        halves = [units[:split], units[split:]] if split < len(units) else [units]
        for i, half in enumerate(halves, 1):
            groups[f"model-trunk-{i:05}-of-{len(halves):05}.safetensors"] = [
                key for unit in half for key in unit
            ]
    return {name: sorted(keys) for name, keys in sorted(groups.items())}


def transfer_hash(src, length, dst=None):
    digest = hashlib.sha256()
    while length:
        data = src.read(min(CHUNK, length))
        if not data:
            raise ValueError("unexpected EOF while reading tensor payload")
        digest.update(data)
        if dst is not None:
            dst.write(data)
        length -= len(data)
    return digest.hexdigest()


def file_hash(path):
    with path.open("rb") as f:
        return transfer_hash(f, path.stat().st_size)


def identical(a, b):
    if not b.is_file() or a.stat().st_size != b.stat().st_size:
        return False
    return os.path.samefile(a, b) or file_hash(a) == file_hash(b)


def write_shard(path, keys, tensors):
    header, size = {}, 0
    for key in keys:
        info = tensors[key]
        header[key] = {
            "dtype": info["dtype"], "shape": info["shape"],
            "data_offsets": [size, size + info["size"]],
        }
        size += info["size"]
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    hashes = {}
    with path.open("xb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for key in keys:
            info = tensors[key]
            with info["path"].open("rb") as src:
                src.seek(info["offset"])
                hashes[key] = transfer_hash(src, info["size"], out)
        out.flush()
        os.fsync(out.fileno())
    with path.open("rb") as check:
        check.seek(8 + len(raw))
        for key in keys:
            if transfer_hash(check, tensors[key]["size"]) != hashes[key]:
                raise ValueError(f"payload verification failed: {key}")


def repack(source, destination, share_with=None, *, progress=None):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    share_with = Path(share_with).resolve() if share_with is not None else None
    if share_with is not None and not share_with.is_dir():
        raise ValueError("--share-with must name an existing component pack directory")
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen4_exp" or config.get("expert_quant", {}).get("format") != "exl3":
        raise ValueError("expected a qwen4_exp EXL3 pack")
    ngram = flat_name(config.get("ngram_table", {}).get("file", "ngram_table.bin"))
    if not (source / ngram).is_file():
        raise ValueError(f"missing n-gram table: {source / ngram}")
    tensors, stamps = catalog(source)
    groups = plan_shards(tensors)
    if ngram in groups or ngram in (INDEX, "config.json") or source / ngram in stamps:
        raise ValueError(f"n-gram filename collides with pack metadata or weights: {ngram}")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.repack-", dir=destination.parent))
    report = {
        "destination": str(destination), "tensor_count": len(tensors),
        "shard_count": len(groups), "tensor_bytes": sum(t["size"] for t in tensors.values()),
        "shared_with_bytes": 0, "source_link_bytes": 0, "shared_files": [],
    }
    reserved = False
    try:
        for name, keys in groups.items():
            if progress:
                progress(f"Writing and verifying {name}")
            out = staging / name
            write_shard(out, keys, tensors)
            if share_with is not None and identical(out, share_with / name):
                out.unlink()
                os.link((share_with / name).resolve(), out)
                report["shared_with_bytes"] += out.stat().st_size
                report["shared_files"].append(name)
        for path in sorted(source.iterdir()):
            if (not path.is_file() or path.name == INDEX
                    or (path.suffix == ".safetensors" and path.name != ngram)):
                continue
            out = staging / path.name
            if out.exists():
                raise ValueError(f"sidecar collides with component shard: {path.name}")
            if path.name == ngram:
                donor = path
                if share_with is not None and identical(path, share_with / ngram):
                    donor = share_with / ngram
                try:
                    os.link(donor.resolve(), out)
                except OSError as exc:
                    if exc.errno != errno.EXDEV or donor != path:
                        raise
                    shutil.copyfile(path, out)
                    if not identical(path, out):
                        raise ValueError("n-gram copy verification failed")
                else:
                    field = "source_link_bytes" if donor == path else "shared_with_bytes"
                    report[field] += out.stat().st_size
                    if donor != path:
                        report["shared_files"].append(ngram)
            else:
                shutil.copyfile(path, out)
        for path, stamp in stamps.items():
            if fingerprint(path) != stamp:
                raise ValueError(f"source changed while repacking: {path}")
        weight_map = {key: name for name, keys in groups.items() for key in keys}
        (staging / INDEX).write_text(json.dumps({
            "metadata": {"total_size": report["tensor_bytes"]},
            "weight_map": dict(sorted(weight_map.items())),
        }, indent=2) + "\n")
        # Reserve the destination exclusively, then atomically replace our empty dir.
        destination.mkdir()
        reserved = True
        staging.rename(destination)
        reserved = False
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if reserved:
            destination.rmdir()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, help="new directory; never overwritten")
    parser.add_argument("--share-with", type=Path, help="hardlink byte-identical component shards and n-gram table")
    args = parser.parse_args()
    try:
        report = repack(args.source, args.destination, args.share_with,
                        progress=lambda line: print(line, file=sys.stderr, flush=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"repack refused: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
