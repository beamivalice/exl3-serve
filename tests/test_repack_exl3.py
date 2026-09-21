#!/usr/bin/env python3
"""Hermetic byte-preserving component-pack tests; no MLX or numpy required."""
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repack_exl3.py"


def write_shard(path, tensors):
    header, payload = {}, bytearray()
    for key, data in tensors.items():
        dtype, shape = "U8", [len(data)]
        if isinstance(data, tuple):
            dtype, shape, data = data
        header[key] = {
            "dtype": dtype, "shape": shape,
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def read_tensors(root):
    index = json.loads((root / "model.safetensors.index.json").read_text())
    result = {}
    for filename in set(index["weight_map"].values()):
        with (root / filename).open("rb") as f:
            n, = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
            payload = f.read()
        for key, info in header.items():
            if key == "__metadata__":
                continue
            start, end = info["data_offsets"]
            result[key] = (info["dtype"], info["shape"], payload[start:end])
    return result


class ComponentPackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("repack_exl3", SCRIPT)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, name, expert=b"expert"):
        root = self.root / name
        root.mkdir()
        tensors = {
            "language_model.model.embed_tokens.weight": b"embed",
            "language_model.lm_head.weight": b"head",
            "language_model.model.hyper_connection_mixer.weight": b"mix",
            "language_model.mtp.layers.0.norm.weight": b"mtp",
            "model.visual.weight": b"vision",
        }
        for layer in range(4):
            tensors[f"language_model.model.layers.{layer}.norm.weight"] = b"norm"
            for proj in ("gate", "up", "down"):
                for suffix in ("trellis", "suh", "svh"):
                    tensors[f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}_proj.{suffix}"] = expert
        tensors["language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.trellis"] = expert
        write_shard(root / "old.safetensors", tensors)
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 1},
            "weight_map": {key: "old.safetensors" for key in tensors},
        }))
        (root / "config.json").write_text(json.dumps({
            "model_type": "qwen4_exp", "expert_quant": {"format": "exl3", "k": 3},
            "ngram_table": {"file": "ngram_table.bin", "bits": 16},
        }))
        (root / "ngram_table.bin").write_bytes(b"ngram table bytes")
        (root / "tokenizer.json").write_text("{}")
        (root / "config.json.orig-262k").write_text("old config")
        return root

    def test_layout_bytes_index_and_sidecars(self):
        src = self.fixture("src")
        dst = self.root / "dst"
        self.mod.repack(src, dst)
        expected = {
            "model-embed.safetensors", "model-lm-head.safetensors",
            "model-trunk-00001-of-00002.safetensors",
            "model-trunk-00002-of-00002.safetensors",
            "model-mtp.safetensors", "model-vision.safetensors",
            *(f"model-experts-L{i:02}.safetensors" for i in range(4)),
        }
        self.assertEqual({p.name for p in dst.glob("*.safetensors")}, expected)
        self.assertEqual(read_tensors(src), read_tensors(dst))
        idx = json.loads((dst / "model.safetensors.index.json").read_text())
        self.assertEqual(idx["metadata"]["total_size"],
                         sum(len(v[2]) for v in read_tensors(src).values()))
        self.assertTrue(os.path.samefile(src / "ngram_table.bin", dst / "ngram_table.bin"))
        self.assertEqual((dst / "config.json").read_bytes(), (src / "config.json").read_bytes())
        self.assertEqual((dst / "config.json.orig-262k").read_bytes(), b"old config")

    def test_share_identical_only_and_either_pack_can_stand_alone(self):
        src3, src4 = self.fixture("src3"), self.fixture("src4", b"EXPERT")
        dst3, dst4 = self.root / "k3", self.root / "k4"
        self.mod.repack(src3, dst3)
        self.mod.repack(src4, dst4, share_with=dst3)
        for filename in ("model-embed.safetensors", "model-lm-head.safetensors",
                         "model-trunk-00001-of-00002.safetensors",
                         "model-trunk-00002-of-00002.safetensors",
                         "model-vision.safetensors", "ngram_table.bin"):
            self.assertTrue(os.path.samefile(dst3 / filename, dst4 / filename), filename)
        for filename in ("model-experts-L00.safetensors", "model-mtp.safetensors"):
            self.assertFalse(os.path.samefile(dst3 / filename, dst4 / filename), filename)
        expected = read_tensors(src4)
        shutil.rmtree(src3)
        shutil.rmtree(src4)
        shutil.rmtree(dst3)
        self.assertEqual(read_tensors(dst4), expected)
        self.assertEqual((dst4 / "ngram_table.bin").read_bytes(), b"ngram table bytes")

    def test_refuse_existing_destination(self):
        src = self.fixture("src")
        with self.assertRaises(FileExistsError):
            self.mod.repack(src, src)
        self.assertTrue((src / "old.safetensors").exists())

    def test_refuse_empty_directory_file_and_dangling_symlink(self):
        src = self.fixture("src")
        directory, file, link = (self.root / n for n in ("dir", "file", "link"))
        directory.mkdir()
        file.write_bytes(b"keep")
        link.symlink_to(self.root / "missing")
        for dst in (directory, file, link):
            with self.assertRaises(FileExistsError):
                self.mod.repack(src, dst)
        self.assertEqual(file.read_bytes(), b"keep")
        self.assertTrue(link.is_symlink())

    def test_failed_write_cleans_staging_without_publishing(self):
        src, dst = self.fixture("src"), self.root / "dst"
        expected = read_tensors(src)
        with patch.object(self.mod, "write_shard", side_effect=OSError("injected write failure")):
            with self.assertRaises(OSError):
                self.mod.repack(src, dst)
        self.assertFalse(dst.exists())
        self.assertEqual(list(self.root.glob(".dst.repack-*")), [])
        self.assertEqual(read_tensors(src), expected)

    def test_share_with_canonical_source(self):
        src = self.fixture("src")
        canonical, dst = self.root / "canonical", self.root / "dst"
        self.mod.repack(src, canonical)
        self.mod.repack(canonical, dst, share_with=canonical)
        self.assertEqual(read_tensors(canonical), read_tensors(dst))
        self.assertTrue(os.path.samefile(
            canonical / "model-mtp.safetensors", dst / "model-mtp.safetensors"))

    def test_refuse_incomplete_index(self):
        src = self.fixture("src")
        (src / "model.safetensors.index.json").write_text(
            '{"weight_map":{"missing":"old.safetensors"}}')
        with self.assertRaises(ValueError):
            self.mod.repack(src, self.root / "dst")
        self.assertFalse((self.root / "dst").exists())

    def test_refuse_unsafe_shard_path(self):
        src = self.fixture("src")
        (src / "model.safetensors.index.json").write_text(
            '{"weight_map":{"missing":"../old.safetensors"}}')
        with self.assertRaises(ValueError):
            self.mod.repack(src, self.root / "dst")

    def test_refuse_truncated_payload(self):
        src = self.fixture("src")
        path = src / "old.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            self.mod.repack(src, self.root / "dst")
        self.assertFalse((self.root / "dst").exists())

    def test_unreferenced_shards_are_not_carried(self):
        src = self.fixture("src")
        write_shard(src / "stale.safetensors", {"stale": b"obsolete"})
        dst = self.root / "dst"
        self.mod.repack(src, dst)
        self.assertNotIn("stale", read_tensors(dst))
        self.assertFalse((dst / "stale.safetensors").exists())

    def test_multishard_real_dtypes_and_shapes(self):
        src = self.fixture("src")
        tensors = read_tensors(src)
        tensors["language_model.model.embed_tokens.weight"] = (
            "U32", [2, 4], bytes(range(32)))
        base = "language_model.model.layers.0.mlp.switch_mlp.gate_proj."
        tensors[base + "trellis"] = ("U16", [2, 3, 4], bytes(range(48)))
        tensors[base + "suh"] = ("BF16", [2, 4], bytes(range(16)))
        tensors[base + "svh"] = ("F32", [2, 4], bytes(range(32)))
        wm = {}
        for i in range(2):
            subset = dict(list(tensors.items())[i::2])
            filename = f"source-{i}.safetensors"
            write_shard(src / filename, subset)
            wm.update({key: filename for key in subset})
        (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
        dst = self.root / "dst"
        self.mod.repack(src, dst)
        self.assertEqual(tensors, read_tensors(dst))
        wm = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
        for layer in range(4):
            prefix = f"language_model.model.layers.{layer}."
            trunk_files = {name for key, name in wm.items()
                           if key.startswith(prefix) and ".switch_mlp." not in key}
            self.assertEqual(len(trunk_files), 1)

    def test_ngram_symlink_becomes_standalone_regular_file(self):
        src = self.fixture("src")
        table = src / "ngram_table.bin"
        blob = self.root / "blob"
        table.rename(blob)
        table.symlink_to(blob)
        dst = self.root / "dst"
        self.mod.repack(src, dst)
        self.assertFalse((dst / table.name).is_symlink())
        blob.unlink()
        self.assertEqual((dst / table.name).read_bytes(), b"ngram table bytes")

    def test_configured_ngram_filename_is_authoritative(self):
        src = self.fixture("src")
        name = "table.safetensors"
        (src / "ngram_table.bin").rename(src / name)
        config = json.loads((src / "config.json").read_text())
        config["ngram_table"]["file"] = name
        (src / "config.json").write_text(json.dumps(config))
        dst = self.root / "dst"
        self.mod.repack(src, dst)
        self.assertEqual((dst / name).read_bytes(), b"ngram table bytes")
        self.assertTrue(os.path.samefile(src / name, dst / name))

    def test_refuse_ngram_output_collision(self):
        src = self.fixture("src")
        name = "model-embed.safetensors"
        (src / "ngram_table.bin").rename(src / name)
        config = json.loads((src / "config.json").read_text())
        config["ngram_table"]["file"] = name
        (src / "config.json").write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            self.mod.repack(src, self.root / "dst")
        self.assertFalse((self.root / "dst").exists())


if __name__ == "__main__":
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ComponentPackTests))
    if not result.wasSuccessful():
        print(output.getvalue(), file=sys.stderr, end="")
        sys.exit(1)
