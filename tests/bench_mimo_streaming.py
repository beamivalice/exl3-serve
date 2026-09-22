#!/usr/bin/env python3
"""Repeatable MiMo streaming benchmark using server-side prefill/decode timings."""
import argparse
import json
import statistics
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:11251")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--repetitions", type=int, default=3)
    args = p.parse_args()
    if args.repetitions < 1:
        p.error("--repetitions must be positive")
    tok = Tokenizer.from_file(args.tokenizer)

    def request(path, body=None):
        req = urllib.request.Request(args.url + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as response:
            return json.load(response)

    passage = (
        "A mixture-of-experts language model selects a small set of expert networks for each token. "
        "Expert streaming keeps frequently used weights in RAM and fetches other weights from SSD. "
        "The cache budget trades resident memory for storage traffic. Prefill processes the input prompt, "
        "while decode generates new tokens one at a time. Benchmarking must keep prompts, sampling, "
        "context settings, and cache policy unchanged when comparing weight formats. "
    )
    report = {"model": args.model, "props": request("/props"),
              "method": f"One warm-up then {args.repetitions} repeats per case; server timings; no KV reuse or speculation.",
              "cases": []}
    for target in (512, 1024):
        ids = tok.encode(passage * 32, add_special_tokens=False).ids[:target - 48]
        content = tok.decode(ids) + "\n\nUsing that context, write at least 250 words explaining expert streaming, its benefits, and its tradeoffs. Do not quote the passage."
        body = {"model": args.model, "messages": [{"role": "user", "content": content}],
                "temperature": 0, "seed": 1234, "max_tokens": 128, "stream": False,
                "enable_thinking": False, "enable_pld": False, "enable_mtp": False}
        case = {"target_prompt": target, "request": body, "runs": []}
        report["cases"].append(case)
        for run in range(args.repetitions + 1):
            response = request("/v1/chat/completions", body)
            timings = response["timings"]
            if timings["cached_n"] != 0:
                raise RuntimeError("KV prefix reuse invalidates this benchmark")
            case["runs"].append(response)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2))
            print(f"target={target} run={run} prompt={timings['prompt_n']} output={timings['predicted_n']} "
                  f"prefill={timings['prompt_per_second']:.2f} decode={timings['predicted_per_second']:.2f}", flush=True)
        measured = case["runs"][1:]
        case["median_prefill_tok_s"] = statistics.median(r["timings"]["prompt_per_second"] for r in measured)
        case["median_decode_tok_s"] = statistics.median(r["timings"]["predicted_per_second"] for r in measured)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"MEDIAN target={target}: prefill={case['median_prefill_tok_s']:.2f} "
              f"decode={case['median_decode_tok_s']:.2f}", flush=True)


if __name__ == "__main__":
    main()
