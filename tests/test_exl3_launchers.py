#!/usr/bin/env python3
"""Exercise the standalone launchers without loading a model."""
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class Exl3LauncherTests(unittest.TestCase):
    def test_both_launchers_preserve_flags_and_forward_arguments(self):
        with tempfile.TemporaryDirectory(prefix="exl3 launcher ") as td:
            home = Path(td)
            binary = home / "llm/sushi/zig-out/bin/sushi"
            binary.parent.mkdir(parents=True)
            binary.write_text("#!/usr/bin/env bash\nprintf '%s\\0' \"$@\"\n")
            binary.chmod(0o755)
            for k in (3, 4):
                with self.subTest(k=k):
                    result = subprocess.run(
                        ["bash", str(SCRIPTS / f"exl3-qwen38flash-k{k}.sh"),
                         "--port", "11235", "--log-file", "/tmp/log with spaces"],
                        env={**os.environ, "HOME": td}, capture_output=True, check=True,
                    )
                    args = result.stdout.decode().rstrip("\0").split("\0")
                    self.assertEqual(args, [
                        "serve",
                        "--model", str(home / f"llm/models/Qwen3.8-Flash-Next-EXL3-K{k}"),
                        "--host", "127.0.0.1",
                        "--port", "11234",
                        "--ctx-size", "1048576",
                        "--prefill-chunk", "8192",
                        "--max-concurrent", "1",
                        "--kv-quant", "8",
                        "--max-tokens", "64000",
                        "--mtp",
                        "--prefix-cache-mem", "12GB",
                        "--prefix-cache-entries", "1",
                        "--prefix-cache-disk", "100GB",
                        "--ssm-checkpoint-max", "16",
                        "--metrics",
                        "--port", "11235", "--log-file", "/tmp/log with spaces",
                    ])
                    self.assertEqual(result.stderr, b"")

    def test_missing_binary_names_the_build_command(self):
        with tempfile.TemporaryDirectory() as td:
            for k in (3, 4):
                with self.subTest(k=k):
                    result = subprocess.run(
                        ["bash", str(SCRIPTS / f"exl3-qwen38flash-k{k}.sh")],
                        env={**os.environ, "HOME": td}, capture_output=True,
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(b"zig build -Doptimize=ReleaseFast", result.stderr)
                    self.assertEqual(result.stdout, b"")


if __name__ == "__main__":
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Exl3LauncherTests))
    if not result.wasSuccessful():
        print(output.getvalue(), file=sys.stderr, end="")
        sys.exit(1)
