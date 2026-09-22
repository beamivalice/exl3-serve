#!/usr/bin/env python3
"""Collect the Qwen3.8-Flash-Next imatrix from the SERVED bf16 checkpoint.

Same output contract as tests/qwen38_flash_next_imatrix_collect.py — the engine
accumulates it on the GPU while it streams the routed experts off SSD, so there is
no CPU layer-by-layer pass. Arms `MLX_SERVE_IMATRIX_OUT`, replays the SAME corpus
as the CPU collector (imported, never re-spelled) as prefill-only `/v1/completions`
with `max_tokens 1`, then stops the server so its teardown writes the file.

  venv/bin/python tests/qwen38_flash_next_imatrix_serve.py \
      --src ~/llm/models/Qwen/Qwen3.8-Flash-Next \
      --out ~/claude-tmp/imatrix-served.safetensors --ssd-budget-gb 96
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen38_flash_next_imatrix_collect as collect  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(port, deadline, proc):
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                rows = json.loads(r.read())["data"]
                if rows:
                    return rows[0]["id"]
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
            time.sleep(2)
    raise SystemExit("server did not become ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="bf16 Qwen3.8-Flash-Next checkpoint")
    ap.add_argument("--out", required=True, help="imatrix safetensors the server writes")
    ap.add_argument("--binary", default=str(REPO / "zig-out/bin/mlx-serve"))
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--ssd-budget-gb", type=int, default=96)
    ap.add_argument("--ctx-size", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=160_000, help="corpus token budget")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--load-timeout", type=int, default=2400)
    ap.add_argument("--request-timeout", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=0, help="stop after N windows (smoke)")
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out)).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.src)
    corpus_args = SimpleNamespace(seed=args.seed, max_tokens=args.max_tokens, seq_len=args.seq_len)
    seqs, composition, _ = collect.build_corpus(tok, corpus_args)
    # The corpus is built as token windows; the server re-tokenizes, so hand it
    # the decoded window rather than a second, divergent rendering of the text.
    prompts = [tok.decode(s.tolist()) for s in seqs]
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"corpus: {len(prompts)} windows, {sum(len(s) for s in seqs)} tokens {composition}", flush=True)

    env = dict(os.environ)
    env["MLX_SERVE_IMATRIX_OUT"] = str(out)
    cmd = [args.binary, "--model", args.src, "--serve", "--host", "127.0.0.1",
           "--port", str(args.port), "--ssd-budget-gb", str(args.ssd_budget_gb),
           "--ctx-size", str(args.ctx_size), "--no-mtp", "--prefix-cache-entries", "0"]
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env)
    try:
        model = wait_ready(args.port, time.time() + args.load_timeout, proc)
        url = f"http://127.0.0.1:{args.port}/v1/completions"
        t0 = time.time()
        for i, prompt in enumerate(prompts):
            post(url, {"model": model, "prompt": prompt, "max_tokens": 1,
                       "temperature": 0, "stream": False}, args.request_timeout)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(prompts)} windows  {(time.time()-t0)/60:.1f} min", flush=True)
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=1800)

    if not out.is_file():
        raise SystemExit(f"server wrote no imatrix at {out}")
    print(f"wrote {out}: {out.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    sys.exit(main())
