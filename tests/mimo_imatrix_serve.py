#!/usr/bin/env python3
"""Collect the MiMo-V2.6-Flash imatrix from the SERVED original checkpoint.

The twin of tests/qwen38_flash_next_imatrix_serve.py for `model_type mimo_v2`:
the engine accumulates the statistics on the GPU while it streams the routed
MXFP4 experts off SSD, so there is no CPU layer-by-layer pass. Arms
`MLX_SERVE_IMATRIX_OUT`, replays the SAME corpus (imported, never re-spelled) as
prefill-only `/v1/completions` with `max_tokens 1`, then stops the server so its
teardown writes the file, and reconciles the routed counts against the tokens
the server reported.

Keys are MiMo's own expert names (`model.layers.{L}.mlp.experts.…`); layer 0 is
dense and owes no entries.

  python3 tests/mimo_imatrix_serve.py \
      --src ~/llm/models/MiMo-V2.6-Flash-RL \
      --out ~/llm/models/calib/mimo-v2.6-flash-rl-imatrix.safetensors
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen38_imatrix_collect as corpus  # noqa: E402  (pure corpus builders)

REPO = Path(__file__).resolve().parent.parent
KEY = "model.layers.{}.mlp.experts.{}"


def post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def post_window(url, payload, timeout):
    """A window the server REFUSES by name (admission at the memory edge) is
    skipped, not fatal: the capture is a sum over windows and the file is still
    written at shutdown."""
    try:
        return post(url, payload, timeout)
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        print(f"  skipped window: 400 {e.read()[:200]!r}", flush=True)
        return None


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


# The load's own warmup forwards (1 decode + 8 prefill positions) route like
# any other token, and every request runs one decode tick past its prefill.
WARMUP_TOKENS = 9


def verify(path, src, tokens=None, requests=0):
    """Routed counts per MoE layer against tokens x top_k, and the dense layer's
    absence. Returns the per-layer report."""
    import numpy as np
    from safetensors.numpy import load_file
    data = load_file(str(path))
    cfg = json.loads((Path(src) / "config.json").read_text())
    layers = int(cfg["num_hidden_layers"])
    experts = int(cfg["n_routed_experts"])
    top_k = int(cfg["num_experts_per_tok"])
    hidden = int(cfg["hidden_size"])
    inter = int(cfg["moe_intermediate_size"])
    report = []
    for li in range(layers):
        gu = data.get(KEY.format(li, "gate_up_proj"))
        dn = data.get(KEY.format(li, "down_proj"))
        rows = data.get(KEY.format(li, "gate_up_proj.rows"))
        if gu is None:
            report.append((li, None, None))
            continue
        assert gu.shape == (experts * hidden,), (li, gu.shape)
        assert dn.shape == (experts * inter,), (li, dn.shape)
        assert rows.shape == (experts,), (li, rows.shape)
        routed = int(rows.sum())
        assert routed % top_k == 0, (li, routed, top_k)
        report.append((li, routed // top_k, int((rows == 0).sum())))
    live = [r for r in report if r[1] is not None]
    if not live:
        raise SystemExit(f"{path}: no MoE layer captured")
    counts = {r[1] for r in live}
    print(f"{path}: {len(live)}/{layers} layers captured, "
          f"{sorted(counts)} tokens/layer, {os.path.getsize(path)} bytes")
    for li, toks, unrouted in report[:2] + report[-1:]:
        print(f"  layer {li}: " + ("dense, no expert entries" if toks is None
                                   else f"{toks} tokens, {unrouted}/{experts} experts never routed"))
    never = [r[2] for r in live]
    print(f"  experts never routed: min {min(never)}, max {max(never)}, "
          f"mean {sum(never)/len(never):.1f} of {experts}")
    if len(counts) != 1:
        print(f"  WARNING: MoE layers disagree on token count: {sorted(counts)}")
    if tokens is not None:
        expected = tokens + requests + WARMUP_TOKENS
        ok = "ok" if counts == {expected} else "MISMATCH"
        print(f"  reconcile: {tokens} prompt + {requests} decode ticks + {WARMUP_TOKENS} warmup "
              f"= {expected}, layers saw {sorted(counts)} [{ok}]")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "llm/models/MiMo-V2.6-Flash-RL"))
    ap.add_argument("--out", required=True, help="imatrix safetensors the server writes")
    ap.add_argument("--binary", default=str(REPO / "zig-out/bin/mlx-serve"))
    ap.add_argument("--port", type=int, default=8124)
    # Below the README's 100: the collector's accumulators are GPU-resident
    # (E x (hidden + inter) f32 per MoE layer, ~0.3 GB here) and come out of the
    # same headroom prefill admission reads.
    ap.add_argument("--ssd-budget-gb", type=int, default=94)
    ap.add_argument("--kv-quant", default="8")
    ap.add_argument("--ctx-size", type=int, default=4096)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=160_000, help="corpus token budget")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--load-timeout", type=int, default=2400)
    ap.add_argument("--request-timeout", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=0, help="stop after N windows (smoke)")
    ap.add_argument("--verify-only", action="store_true", help="only re-check an existing file")
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out)).resolve()
    if args.verify_only:
        verify(out, args.src)
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.src)
    windows, composition = corpus.build_windows(tok, args.seed, args.max_tokens, args.seq_len)
    # The corpus is built as token windows; the server re-tokenizes, so hand it
    # the decoded window rather than a second, divergent rendering of the text.
    prompts = [tok.decode(w) for w in windows]
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"corpus: {len(prompts)} windows, {sum(len(w) for w in windows)} tokens {composition}", flush=True)

    env = dict(os.environ)
    env["MLX_SERVE_IMATRIX_OUT"] = str(out)
    cmd = [args.binary, "--model", args.src, "--serve", "--host", "127.0.0.1",
           "--port", str(args.port), "--ssd-budget-gb", str(args.ssd_budget_gb),
           "--kv-quant", args.kv_quant, "--ctx-size", str(args.ctx_size),
           "--prefill-chunk", str(args.prefill_chunk),
           "--no-mtp", "--no-pld", "--no-vision", "--prefix-cache-entries", "0"]
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env)
    tokens, served, skipped = 0, 0, 0
    t0 = time.time()
    try:
        model = wait_ready(args.port, time.time() + args.load_timeout, proc)
        url = f"http://127.0.0.1:{args.port}/v1/completions"
        t0 = time.time()
        for i, prompt in enumerate(prompts):
            reply = post_window(url, {"model": model, "prompt": prompt, "max_tokens": 1,
                                      "temperature": 0, "stream": False}, args.request_timeout)
            if reply is None:
                skipped += 1
                continue
            tokens += int(reply.get("usage", {}).get("prompt_tokens", 0))
            served += 1
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(prompts)} windows  {tokens} tokens  "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=1800)
    wall = time.time() - t0

    if not out.is_file():
        raise SystemExit(f"server wrote no imatrix at {out}")
    print(f"served {served}/{len(prompts)} windows ({skipped} refused), "
          f"{tokens} prompt tokens in {wall/60:.1f} min", flush=True)
    verify(out, args.src, tokens, served)
    return 0


if __name__ == "__main__":
    sys.exit(main())
