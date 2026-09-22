#!/usr/bin/env python3
"""Hermetic tests for the Qwen3.8 BF16 expert graft converter."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


import convert_qwen38_flash_next as affine  # noqa: E402
import convert_qwen38_flash_next_affine_graft as graft  # noqa: E402
from convert_qwen38_flash_next_affine_graft import (  # noqa: E402
    build_affine_config,
    destination_guard,
    expected_route_specs,
)


class AffineGraftConfigTests(unittest.TestCase):
    def test_route_specs_cover_trunk_and_mtp_at_requested_geometry(self):
        specs = expected_route_specs(48, bits=3, group_size=128)

        self.assertEqual(len(specs), 49 * 3)
        self.assertEqual(
            specs["language_model.model.layers.0.mlp.switch_mlp.gate_proj"],
            {"bits": 3, "group_size": 128, "mode": "affine"},
        )
        self.assertEqual(
            specs["language_model.model.layers.47.mlp.switch_mlp.down_proj"],
            {"bits": 3, "group_size": 128, "mode": "affine"},
        )
        self.assertEqual(
            specs["language_model.mtp.layers.0.mlp.switch_mlp.up_proj"],
            {"bits": 3, "group_size": 128, "mode": "affine"},
        )

    def test_config_preserves_donor_metadata_and_drops_exl3(self):
        donor = {
            "model_type": "qwen4_exp",
            "text_config": {
                "num_hidden_layers": 48,
                "rope_parameters": {"rope_type": "yarn", "factor": 4.0},
            },
            "vision_config": {"hidden_size": 1152},
            "quantization": {
                "bits": 8,
                "group_size": 64,
                "mode": "affine",
                "language_model.model.layers.0.self_attn.q_proj": {
                    "bits": 8,
                    "group_size": 128,
                    "mode": "affine",
                },
            },
            "quantization_config": {
                "bits": 8,
                "group_size": 64,
                "mode": "affine",
            },
            "expert_quant": {
                "format": "exl3",
                "k": 3,
                "codebook": "mul1",
            },
            "ngram_table": {"file": "ngram_table.bin", "bits": 16, "group_size": 0},
        }

        got = build_affine_config(donor, 48, bits=3, group_size=128)

        self.assertNotIn("expert_quant", got)
        self.assertEqual(got["text_config"], donor["text_config"])
        self.assertEqual(got["vision_config"], donor["vision_config"])
        self.assertEqual(got["ngram_table"], donor["ngram_table"])
        self.assertEqual(
            got["quantization"]["language_model.model.layers.0.self_attn.q_proj"],
            donor["quantization"]["language_model.model.layers.0.self_attn.q_proj"],
        )
        self.assertEqual(
            got["quantization"]["language_model.model.layers.0.mlp.switch_mlp.gate_proj"],
            {"bits": 3, "group_size": 128, "mode": "affine"},
        )
        self.assertEqual(
            got["quantization_config"]["language_model.mtp.layers.0.mlp.switch_mlp.down_proj"],
            {"bits": 3, "group_size": 128, "mode": "affine"},
        )
        self.assertEqual(
            got["quantization"]["calibration"],
            "uncalibrated affine from BF16 source",
        )

        donor["text_config"]["rope_parameters"]["factor"] = 99.0
        self.assertEqual(
            got["text_config"]["rope_parameters"]["factor"],
            4.0,
        )


class AffineGraftSafetyTests(unittest.TestCase):
    def test_destination_guard_refuses_any_existing_path(self):
        with tempfile.TemporaryDirectory() as td:
            existing = Path(td) / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(FileExistsError, "destination already exists"):
                destination_guard(existing)

    def test_destination_guard_allows_a_new_path_without_creating_it(self):
        with tempfile.TemporaryDirectory() as td:
            new_path = Path(td) / "new"
            destination_guard(new_path)
            self.assertFalse(new_path.exists())


class AffineGraftLayoutTests(unittest.TestCase):
    @staticmethod
    def _fake_quant(arr, bits, group_size):
        shape = tuple(arr.shape)
        lead = shape[:-1]
        packed = shape[-1] * bits // 32
        groups = shape[-1] // group_size
        return (
            ("U32", lead + (packed,), np.zeros((
                *lead, packed
            ), dtype=np.uint32).tobytes()),
            ("BF16", lead + (groups,), np.zeros((
                *lead, groups
            ), dtype=np.uint16).tobytes()),
            ("BF16", lead + (groups,), np.zeros((
                *lead, groups
            ), dtype=np.uint16).tobytes()),
        )

    @staticmethod
    def _fake_shard(path, named):
        affine.write_safetensors_raw(str(path), named)

    def test_conversion_replaces_only_routes_and_audits_affine_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src"
            donor = root / "donor"
            dst = root / "dst"
            src.mkdir()
            donor.mkdir()
            text = {
                "num_hidden_layers": 1,
                "num_experts": 2,
                "hidden_size": 32,
                "moe_intermediate_size": 32,
            }
            source_cfg = {"model_type": "qwen4_exp", "text_config": text}
            donor_cfg = {
                "model_type": "qwen4_exp",
                "text_config": text,
                "vision_config": {"hidden_size": 4},
                "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
                "quantization_config": {
                    "bits": 8,
                    "group_size": 64,
                    "mode": "affine",
                },
                "expert_quant": {"format": "exl3", "k": 3},
                "ngram_table": {"file": "ngram_table.bin", "bits": 16, "group_size": 0},
            }
            (src / "config.json").write_text(json.dumps(source_cfg))
            (donor / "config.json").write_text(json.dumps(donor_cfg))

            source_map = {}
            route_source = {
                "model.language_model.layers.0.mlp.experts.gate_up_proj": (
                    "source-gate.safetensors",
                    (2, 64, 32),
                ),
                "model.language_model.layers.0.mlp.experts.down_proj": (
                    "source-down.safetensors",
                    (2, 32, 32),
                ),
                "mtp.layers.0.mlp.experts.gate_up_proj": (
                    "source-mtp-gate.safetensors",
                    (2, 64, 32),
                ),
                "mtp.layers.0.mlp.experts.down_proj": (
                    "source-mtp-down.safetensors",
                    (2, 32, 32),
                ),
            }
            for name, (filename, shape) in route_source.items():
                self._fake_shard(
                    src / filename,
                    {name: ("BF16", shape, np.zeros(shape, dtype=np.uint16).tobytes())},
                )
                source_map[name] = filename
            (src / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": source_map})
            )

            donor_map = {}
            donor_route = expected_route_specs(1, bits=3, group_size=128)
            for module in donor_route:
                shard = (
                    "model-experts-L00.safetensors"
                    if ".model.layers." in module
                    else "model-mtp.safetensors"
                )
                for suffix in ("trellis", "suh", "svh"):
                    name = f"{module}.{suffix}"
                    donor_map[name] = shard
            self._fake_shard(
                donor / "model-experts-L00.safetensors",
                {
                    name: ("U32", (2, 1), np.zeros((2, 1), dtype=np.uint32).tobytes())
                    for name in donor_map
                    if donor_map[name] == "model-experts-L00.safetensors"
                },
            )
            self._fake_shard(
                donor / "model-mtp.safetensors",
                {
                    **{
                        name: ("U32", (2, 1), np.zeros((2, 1), dtype=np.uint32).tobytes())
                        for name in donor_map
                        if donor_map[name] == "model-mtp.safetensors"
                    },
                    "language_model.mtp.fc_embedding.weight": (
                        "BF16",
                        (1, 32),
                        np.zeros((1, 32), dtype=np.uint16).tobytes(),
                    ),
                },
            )
            nonexpert = donor / "model-trunk.safetensors"
            self._fake_shard(
                nonexpert,
                {
                    "language_model.model.layers.0.self_attn.q_proj.weight": (
                        "BF16",
                        (32, 32),
                        np.zeros((32, 32), dtype=np.uint16).tobytes(),
                    )
                },
            )
            donor_map["language_model.model.layers.0.self_attn.q_proj.weight"] = (
                nonexpert.name
            )
            ngram = donor / "ngram_table.bin"
            ngram.write_bytes(b"raw-bf16-ngram")
            (donor / "tokenizer.json").write_text("{}")
            (donor / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": donor_map})
            )

            fake_mlx = SimpleNamespace(cpu=object(), set_default_device=lambda _: None)
            args = SimpleNamespace(
                src=str(src),
                donor=str(donor),
                dst=str(dst),
                bits=3,
                expert_gs=2,
                batch_experts=1,
                cpu_threads=1,
            )
            with mock.patch.object(graft.affine, "quant", self._fake_quant), \
                    mock.patch.object(graft, "mlx_runtime", return_value=fake_mlx):
                audit = graft.convert(args)

            self.assertEqual(audit["routed_tensors"], 18)
            self.assertNotIn("expert_quant", json.loads((dst / "config.json").read_text()))
            out_index = json.loads((dst / "model.safetensors.index.json").read_text())
            self.assertEqual(
                out_index["weight_map"][
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
                ],
                "model-experts-L00.safetensors",
            )
            self.assertFalse(
                any(
                    name.endswith((".trellis", ".suh", ".svh"))
                    for name in out_index["weight_map"]
                )
            )
            self.assertEqual(
                os.stat(dst / "model-trunk.safetensors").st_ino,
                os.stat(nonexpert).st_ino,
            )
            self.assertEqual(
                os.stat(dst / "ngram_table.bin").st_ino,
                os.stat(ngram).st_ino,
            )
            header, _ = affine.read_header(str(dst / "model-experts-L00.safetensors"))
            self.assertEqual(
                header[
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
                ]["shape"],
                [2, 32, 3],
            )


if __name__ == "__main__":
    unittest.main()
