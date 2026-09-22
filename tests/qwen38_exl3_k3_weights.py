#!/usr/bin/env python3
"""Measure sampled EXL3 expert-weight cosine against the BF16 checkpoint.

This is deliberately a CPU-only measurement.  It reads one expert row at a
time from each safetensors file and delegates EXL3 reconstruction to
``ponyexl3.ref.reconstruct.reconstruct_public_weights``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import struct
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np


MatrixName = Literal["gate", "up", "down"]
MATRIX_NAMES: tuple[MatrixName, ...] = ("gate", "up", "down")
NUM_LAYERS = 48
NUM_EXPERTS = 512
HIDDEN_SIZE = 2560
EXPERT_SIZE = 640
EXL3_K = 3
EXL3_CODEBOOK = "mul1"
EXL3_PACKED_SIZE = 256 * EXL3_K // 16

DEFAULT_TARGET = Path("/Users/beam/llm/models/Qwen3.8-Flash-Next-EXL3-K3")
DEFAULT_SOURCE = Path("/Users/beam/llm/models/Qwen/Qwen3.8-Flash-Next")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "measurements" / "qwen38-exl3-k3" / "weights.json"
)

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
_HEADER_LENGTH = struct.Struct("<Q")


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True)
class FileHeader:
    path: Path
    data_offset: int
    tensors: dict[str, TensorInfo]


@dataclass(frozen=True)
class TensorRef:
    path: Path
    info: TensorInfo


def bf16_bytes_to_float32(raw: bytes) -> np.ndarray:
    """Decode little-endian BF16 values without going through a lossy dtype."""
    words = np.frombuffer(raw, dtype="<u2")
    widened = words.astype("<u4") << 16
    return widened.view("<f4")


def _decode_tensor_bytes(raw: bytes, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return bf16_bytes_to_float32(raw)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if dtype == "U16":
        return np.frombuffer(raw, dtype="<u2").copy()
    raise ValueError(f"unsupported safetensors dtype {dtype!r}")


def _read_file_header(path: Path) -> FileHeader:
    with path.open("rb") as handle:
        length_bytes = handle.read(_HEADER_LENGTH.size)
        if len(length_bytes) != _HEADER_LENGTH.size:
            raise ValueError(f"{path}: missing safetensors header length")
        header_length = _HEADER_LENGTH.unpack(length_bytes)[0]
        header_raw = handle.read(header_length)
        if len(header_raw) != header_length:
            raise ValueError(f"{path}: truncated safetensors header")

    payload = json.loads(header_raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    tensors: dict[str, TensorInfo] = {}
    for name, value in payload.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{path}: tensor header for {name!r} is not an object")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if dtype not in _DTYPE_BYTES or not isinstance(shape, list) or not isinstance(offsets, list):
            raise ValueError(f"{path}: malformed tensor header for {name!r}")
        if len(offsets) != 2 or any(not isinstance(x, int) for x in offsets):
            raise ValueError(f"{path}: malformed offsets for {name!r}")
        if any(not isinstance(x, int) or x < 0 for x in shape):
            raise ValueError(f"{path}: malformed shape for {name!r}")
        begin, end = offsets
        if begin < 0 or end < begin:
            raise ValueError(f"{path}: invalid offsets for {name!r}")
        expected_bytes = math.prod(shape) * _DTYPE_BYTES[dtype]
        if end - begin != expected_bytes:
            raise ValueError(
                f"{path}: {name} has {end - begin} payload bytes, expected {expected_bytes}"
            )
        tensors[name] = TensorInfo(
            name=name,
            dtype=dtype,
            shape=tuple(shape),
            begin=begin,
            end=end,
        )
    return FileHeader(path=path, data_offset=_HEADER_LENGTH.size + header_length, tensors=tensors)


class SafetensorsRows:
    """Header/indexed first-axis row reader; tensor banks are never materialized."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index_path = model_dir / "model.safetensors.index.json"
        try:
            index_payload = json.loads(index_path.read_text())
        except FileNotFoundError as exc:
            raise ValueError(f"missing safetensors index: {index_path}") from exc
        weight_map = index_payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"{index_path}: missing weight_map")
        self.index_path = index_path
        self._weight_map = {str(name): str(file_name) for name, file_name in weight_map.items()}
        self._headers: dict[Path, FileHeader] = {}

    def ref(self, name: str) -> TensorRef:
        try:
            file_name = self._weight_map[name]
        except KeyError as exc:
            raise ValueError(f"{self.model_dir}: tensor is not indexed: {name}") from exc
        path = self.model_dir / file_name
        header = self._headers.get(path)
        if header is None:
            header = _read_file_header(path)
            self._headers[path] = header
        try:
            info = header.tensors[name]
        except KeyError as exc:
            raise ValueError(f"{path}: tensor {name!r} is absent from its header") from exc
        return TensorRef(path=path, info=info)

    def shape(self, name: str) -> tuple[int, ...]:
        return self.ref(name).info.shape

    def dtype(self, name: str) -> str:
        return self.ref(name).info.dtype

    def read_first_axis(self, name: str, index: int) -> np.ndarray:
        ref = self.ref(name)
        shape = ref.info.shape
        if not shape:
            raise ValueError(f"{name}: scalar tensor cannot be indexed by expert")
        if index < 0 or index >= shape[0]:
            raise IndexError(f"{name}: expert index {index} outside [0, {shape[0]})")
        row_elements = math.prod(shape[1:])
        item_bytes = _DTYPE_BYTES[ref.info.dtype]
        row_bytes = row_elements * item_bytes
        offset = ref.info.begin + index * row_bytes
        with ref.path.open("rb") as handle:
            handle.seek(self._headers[ref.path].data_offset + offset)
            raw = handle.read(row_bytes)
        if len(raw) != row_bytes:
            raise ValueError(f"{ref.path}: short read for {name} expert {index}")
        decoded = _decode_tensor_bytes(raw, ref.info.dtype)
        return decoded.reshape(shape[1:])


def source_public_matrix(
    source_expert: np.ndarray,
    matrix: MatrixName,
    *,
    gate_width: int = EXPERT_SIZE,
) -> np.ndarray:
    """Convert source checkpoint layout to the public ``[in_features, out]`` layout."""
    if source_expert.ndim != 2:
        raise ValueError(f"{matrix}: source expert row must be 2D, got {source_expert.shape}")
    if matrix in ("gate", "up"):
        expected_rows = gate_width * 2
        if source_expert.shape[0] != expected_rows:
            raise ValueError(
                f"{matrix}: fused source has {source_expert.shape[0]} rows, expected {expected_rows}"
            )
        first_row = 0 if matrix == "gate" else gate_width
        return source_expert[first_row : first_row + gate_width, :].T
    if matrix == "down":
        return source_expert.T
    raise ValueError(f"unknown matrix {matrix!r}")


def stratified_expert_ids(num_experts: int = NUM_EXPERTS, count: int = 4) -> tuple[int, ...]:
    """Choose integer midpoints from equally sized expert strata."""
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if count <= 0 or count > num_experts:
        raise ValueError("count must be in [1, num_experts]")
    ids = tuple(((2 * index + 1) * num_experts - 1) // (2 * count) for index in range(count))
    if len(set(ids)) != count:
        raise ValueError("strata do not produce distinct expert ids")
    return ids


def cosine_moments(source: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """Return float64 dot, source norm squared, and target norm squared."""
    source_flat = np.asarray(source).reshape(-1)
    target_flat = np.asarray(target).reshape(-1)
    if source_flat.shape != target_flat.shape:
        raise ValueError(f"cosine shape mismatch: {source_flat.shape} != {target_flat.shape}")
    if not np.isfinite(source_flat).all():
        raise ValueError("source contains non-finite values")
    if not np.isfinite(target_flat).all():
        raise ValueError("target contains non-finite values")
    source64 = source_flat.astype(np.float64, copy=False)
    target64 = target_flat.astype(np.float64, copy=False)
    dot = float(np.dot(source64, target64))
    source_norm_sq = float(np.dot(source64, source64))
    target_norm_sq = float(np.dot(target64, target64))
    if not math.isfinite(dot) or not math.isfinite(source_norm_sq) or not math.isfinite(target_norm_sq):
        raise ValueError("cosine moments are non-finite")
    if source_norm_sq <= 0.0 or target_norm_sq <= 0.0:
        raise ValueError("cosine rejects a zero-norm matrix")
    return dot, source_norm_sq, target_norm_sq


def cosine_from_moments(dot: float, source_norm_sq: float, target_norm_sq: float) -> float:
    if not all(math.isfinite(value) for value in (dot, source_norm_sq, target_norm_sq)):
        raise ValueError("cosine moments are non-finite")
    if source_norm_sq <= 0.0 or target_norm_sq <= 0.0:
        raise ValueError("cosine rejects a zero-norm matrix")
    return dot / math.sqrt(source_norm_sq * target_norm_sq)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"missing config: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: config is not an object")
    return value


def _target_names(layer: int, matrix: MatrixName) -> tuple[str, str, str]:
    prefix = f"language_model.model.layers.{layer}.mlp.switch_mlp.{matrix}_proj"
    return f"{prefix}.trellis", f"{prefix}.suh", f"{prefix}.svh"


def _source_name(layer: int, matrix: MatrixName) -> str:
    prefix = f"model.language_model.layers.{layer}.mlp.experts"
    if matrix in ("gate", "up"):
        return f"{prefix}.gate_up_proj"
    if matrix == "down":
        return f"{prefix}.down_proj"
    raise ValueError(f"unknown matrix {matrix!r}")


def _expected_public_shape(matrix: MatrixName) -> tuple[int, int]:
    return (HIDDEN_SIZE, EXPERT_SIZE) if matrix in ("gate", "up") else (EXPERT_SIZE, HIDDEN_SIZE)


def _validate_target_layout(store: SafetensorsRows, layer: int, matrix: MatrixName) -> None:
    trellis_name, suh_name, svh_name = _target_names(layer, matrix)
    in_features, out_features = _expected_public_shape(matrix)
    expected_trellis = (NUM_EXPERTS, in_features // 16, out_features // 16, EXL3_PACKED_SIZE)
    expected_scale_shape = (NUM_EXPERTS, in_features)
    expected_output_scale_shape = (NUM_EXPERTS, out_features)
    if store.shape(trellis_name) != expected_trellis:
        raise ValueError(
            f"{trellis_name}: shape {store.shape(trellis_name)} != {expected_trellis}"
        )
    if store.shape(suh_name) != expected_scale_shape:
        raise ValueError(f"{suh_name}: shape {store.shape(suh_name)} != {expected_scale_shape}")
    if store.shape(svh_name) != expected_output_scale_shape:
        raise ValueError(
            f"{svh_name}: shape {store.shape(svh_name)} != {expected_output_scale_shape}"
        )
    if store.dtype(trellis_name) != "U16":
        raise ValueError(f"{trellis_name}: expected U16")
    if store.dtype(suh_name) != "F16" or store.dtype(svh_name) != "F16":
        raise ValueError(f"{layer}/{matrix}: expected F16 suh/svh")


def _validate_source_layout(store: SafetensorsRows, layer: int, matrix: MatrixName) -> None:
    name = _source_name(layer, matrix)
    expected = (
        (NUM_EXPERTS, EXPERT_SIZE * 2, HIDDEN_SIZE)
        if matrix in ("gate", "up")
        else (NUM_EXPERTS, HIDDEN_SIZE, EXPERT_SIZE)
    )
    if store.shape(name) != expected:
        raise ValueError(f"{name}: shape {store.shape(name)} != {expected}")
    if store.dtype(name) != "BF16":
        raise ValueError(f"{name}: expected BF16")


@lru_cache(maxsize=1)
def _reconstructor(ponyexl3_root: str):
    root = str(Path(ponyexl3_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from ponyexl3.ref.reconstruct import reconstruct_public_weights

    return reconstruct_public_weights


def _reconstruct_target(
    trellis: np.ndarray,
    suh: np.ndarray,
    svh: np.ndarray,
    *,
    ponyexl3_root: Path,
) -> np.ndarray:
    reconstruct_public_weights = _reconstructor(str(ponyexl3_root))
    reconstructed = reconstruct_public_weights(
        trellis,
        suh,
        svh,
        EXL3_K,
        mul1=True,
    )
    return np.asarray(reconstructed, dtype=np.float32)


_WORKER_SOURCE_STORE: SafetensorsRows | None = None
_WORKER_TARGET_STORE: SafetensorsRows | None = None
_WORKER_PONYEXL3_ROOT: Path | None = None


def _init_worker(source_dir: str, target_dir: str, ponyexl3_root: str) -> None:
    global _WORKER_SOURCE_STORE, _WORKER_TARGET_STORE, _WORKER_PONYEXL3_ROOT
    _WORKER_SOURCE_STORE = SafetensorsRows(Path(source_dir))
    _WORKER_TARGET_STORE = SafetensorsRows(Path(target_dir))
    _WORKER_PONYEXL3_ROOT = Path(ponyexl3_root)


def _measure_worker(task: tuple[int, int, MatrixName]) -> dict[str, Any]:
    if _WORKER_SOURCE_STORE is None or _WORKER_TARGET_STORE is None or _WORKER_PONYEXL3_ROOT is None:
        raise RuntimeError("measurement worker was not initialized")
    layer, expert_id, matrix = task
    return _measure_one(
        _WORKER_SOURCE_STORE,
        _WORKER_TARGET_STORE,
        layer,
        expert_id,
        matrix,
        ponyexl3_root=_WORKER_PONYEXL3_ROOT,
    )


def _measure_one(
    source_store: SafetensorsRows,
    target_store: SafetensorsRows,
    layer: int,
    expert_id: int,
    matrix: MatrixName,
    *,
    ponyexl3_root: Path,
) -> dict[str, Any]:
    trellis_name, suh_name, svh_name = _target_names(layer, matrix)
    source_name = _source_name(layer, matrix)

    trellis = target_store.read_first_axis(trellis_name, expert_id)
    suh = target_store.read_first_axis(suh_name, expert_id)
    svh = target_store.read_first_axis(svh_name, expert_id)
    source_expert = source_store.read_first_axis(source_name, expert_id)
    source_public = source_public_matrix(source_expert, matrix)
    target_public = _reconstruct_target(trellis, suh, svh, ponyexl3_root=ponyexl3_root)

    expected_shape = _expected_public_shape(matrix)
    if source_public.shape != expected_shape or target_public.shape != expected_shape:
        raise ValueError(
            f"layer {layer} expert {expert_id} {matrix}: source/target shapes "
            f"{source_public.shape}/{target_public.shape} != {expected_shape}"
        )

    dot, source_norm_sq, target_norm_sq = cosine_moments(source_public, target_public)
    cosine = cosine_from_moments(dot, source_norm_sq, target_norm_sq)
    return {
        "layer": layer,
        "expert_id": expert_id,
        "matrix": matrix,
        "weight_count": math.prod(expected_shape),
        "source_shape": list(source_public.shape),
        "target_shape": list(target_public.shape),
        "cosine": cosine,
        "loss": 1.0 - cosine,
        "_dot": dot,
        "_source_norm_sq": source_norm_sq,
        "_target_norm_sq": target_norm_sq,
    }


def _sum_moments(rows: Iterable[dict[str, Any]]) -> tuple[float, float, float]:
    dot = 0.0
    source_norm_sq = 0.0
    target_norm_sq = 0.0
    for row in rows:
        dot += float(row["_dot"])
        source_norm_sq += float(row["_source_norm_sq"])
        target_norm_sq += float(row["_target_norm_sq"])
    return dot, source_norm_sq, target_norm_sq


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty measurement")
    losses = np.asarray([row["loss"] for row in rows], dtype=np.float64)
    moments = _sum_moments(rows)
    cosine = cosine_from_moments(*moments)
    worst = max(rows, key=lambda row: (row["loss"], -row["layer"], -row["expert_id"]))
    return {
        "matrix_count": len(rows),
        "weight_count": sum(int(row["weight_count"]) for row in rows),
        "mean_loss": float(np.mean(losses, dtype=np.float64)),
        "p95_loss": float(np.percentile(losses, 95.0, method="linear")),
        "worst_loss": float(np.max(losses)),
        "mean_cosine": float(np.mean(np.asarray([row["cosine"] for row in rows], dtype=np.float64))),
        "global_cosine": cosine,
        "global_loss": 1.0 - cosine,
        "worst": {
            "layer": worst["layer"],
            "expert_id": worst["expert_id"],
            "matrix": worst["matrix"],
            "cosine": worst["cosine"],
            "loss": worst["loss"],
        },
        "moments_float64": {
            "dot": moments[0],
            "source_norm_sq": moments[1],
            "target_norm_sq": moments[2],
        },
    }


def _validate_configs(source_dir: Path, target_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_config = _config(source_dir)
    target_config = _config(target_dir)
    text = target_config.get("text_config", {})
    if not isinstance(text, dict):
        raise ValueError("target config text_config is not an object")
    for key, expected in (
        ("num_hidden_layers", NUM_LAYERS),
        ("num_experts", NUM_EXPERTS),
        ("hidden_size", HIDDEN_SIZE),
        ("moe_intermediate_size", EXPERT_SIZE),
    ):
        if text.get(key) != expected:
            raise ValueError(f"target config {key}={text.get(key)!r}, expected {expected}")
    expert_quant = target_config.get("expert_quant")
    if not isinstance(expert_quant, dict):
        raise ValueError("target config has no inline expert_quant")
    if expert_quant.get("format") != "exl3" or expert_quant.get("k") != EXL3_K:
        raise ValueError(f"target config expert_quant does not describe EXL3 K3: {expert_quant}")
    if expert_quant.get("codebook") != EXL3_CODEBOOK:
        raise ValueError(f"target config codebook is not {EXL3_CODEBOOK!r}: {expert_quant}")
    if source_config.get("model_type") != "qwen4_exp":
        raise ValueError(f"unexpected source model_type: {source_config.get('model_type')!r}")
    return source_config, target_config


def measure(
    source_dir: Path,
    target_dir: Path,
    output: Path,
    *,
    expert_ids: tuple[int, ...] | None = None,
    ponyexl3_root: Path = Path("/Users/beam/llm/ponyexl3"),
    workers: int = 1,
) -> dict[str, Any]:
    source_dir = source_dir.expanduser().resolve()
    target_dir = target_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    source_config, target_config = _validate_configs(source_dir, target_dir)
    del source_config, target_config
    source_store = SafetensorsRows(source_dir)
    target_store = SafetensorsRows(target_dir)
    selected_experts = expert_ids if expert_ids is not None else stratified_expert_ids()
    if workers <= 0:
        raise ValueError("workers must be positive")
    if len(selected_experts) != len(set(selected_experts)):
        raise ValueError("expert ids must be distinct")
    if any(expert < 0 or expert >= NUM_EXPERTS for expert in selected_experts):
        raise ValueError(f"expert ids must be in [0, {NUM_EXPERTS})")

    for layer in range(NUM_LAYERS):
        for matrix in MATRIX_NAMES:
            _validate_source_layout(source_store, layer, matrix)
            _validate_target_layout(target_store, layer, matrix)

    tasks = [
        (layer, expert_id, matrix)
        for layer in range(NUM_LAYERS)
        for expert_id in selected_experts
        for matrix in MATRIX_NAMES
    ]
    rows: list[dict[str, Any]] = []
    if workers == 1:
        results = (
            _measure_one(
                source_store,
                target_store,
                layer,
                expert_id,
                matrix,
                ponyexl3_root=ponyexl3_root,
            )
            for layer, expert_id, matrix in tasks
        )
    else:
        if "fork" not in multiprocessing.get_all_start_methods():
            raise ValueError("workers > 1 requires a fork-capable Python runtime")
        context = multiprocessing.get_context("fork")
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(str(source_dir), str(target_dir), str(ponyexl3_root.resolve())),
        )
        results = pool.map(_measure_worker, tasks, chunksize=1)
    try:
        for task, result in zip(tasks, results):
            layer, expert_id, matrix = task
            print(f"layer={layer:02d} expert={expert_id:03d} matrix={matrix}", file=sys.stderr, flush=True)
            rows.append(result)
    finally:
        if workers > 1:
            pool.shutdown()

    by_matrix = {
        matrix: _summarize([row for row in rows if row["matrix"] == matrix])
        for matrix in MATRIX_NAMES
    }
    summary = _summarize(rows)
    for row in rows:
        row.pop("_dot", None)
        row.pop("_source_norm_sq", None)
        row.pop("_target_norm_sq", None)

    result = {
        "schema_version": 1,
        "measurement": "cpu_reconstructed_weight_cosine",
        "source_path": str(source_dir),
        "target_path": str(target_dir),
        "source_config_sha256": _sha256(source_dir / "config.json"),
        "target_config_sha256": _sha256(target_dir / "config.json"),
        "source_index_sha256": _sha256(source_store.index_path),
        "target_index_sha256": _sha256(target_store.index_path),
        "method": {
            "cpu_only": True,
            "reconstructor": "ponyexl3.ref.reconstruct.reconstruct_public_weights",
            "ponyexl3_root": str(ponyexl3_root.resolve()),
            "exl3_k": EXL3_K,
            "codebook": EXL3_CODEBOOK,
            "source_dtype": "BF16",
            "bf16_conversion": "little-endian uint16 word shifted left 16 bits into float32",
            "source_orientation": (
                "gate_up source[expert, 0:640, :] and [640:1280, :] transpose to public [2560,640]; "
                "down source[expert, :, :] transpose to public [640,2560]"
            ),
            "target_orientation": (
                "reconstruct_public_weights returns [in_features,out_features]; "
                "suh is the left/input sign vector and svh is the right/output sign vector"
            ),
            "target_layout_validation": (
                "trellis tile dimensions times 16 equal public dimensions and packed size is 48; "
                "suh/svh first-axis rows match input/output dimensions"
            ),
            "raw_safetensors_reads": "header parsed and one first-axis expert row seek-read at a time",
            "accumulation": "float64 dot and squared-norm moments",
            "workers": workers,
            "sample": {
                "layers": list(range(NUM_LAYERS)),
                "expert_ids": list(selected_experts),
                "expert_count_per_layer": len(selected_experts),
                "exhaustive": False,
            },
        },
        "geometry": {
            "layers": NUM_LAYERS,
            "experts": NUM_EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "expert_intermediate_size": EXPERT_SIZE,
            "matrices_per_expert": list(MATRIX_NAMES),
            "sampled_matrices": len(rows),
            "sampled_weights": sum(int(row["weight_count"]) for row in rows),
        },
        "summary": summary,
        "by_matrix": by_matrix,
        "matrices": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _parse_expert_ids(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expert ids must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("expert ids cannot be empty")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expert-ids",
        type=_parse_expert_ids,
        default=None,
        help="comma-separated expert ids; default is four quartile midpoints",
    )
    parser.add_argument(
        "--ponyexl3-root",
        type=Path,
        default=Path(os.environ.get("PONYEXL3_ROOT", "/Users/beam/llm/ponyexl3")),
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    measure(
        args.source,
        args.target,
        args.output,
        expert_ids=args.expert_ids,
        ponyexl3_root=args.ponyexl3_root,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
