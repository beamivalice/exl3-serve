import collections
import json
import re
import statistics
import sys
from pathlib import Path


def summarize(raw):
    widths = collections.defaultdict(lambda: collections.defaultdict(list))
    greedy = collections.defaultdict(list)
    tokens = set()
    for rows, rnd, rep, arm, ns in re.findall(r'(?:\[perf-width\]|DECODE) rows=(\d+) round=(\d+) rep=(\d+) arm=(\d+) ns=(\d+)', raw):
        widths[rows][(int(arm), int(rnd))].append(int(ns) / 1e6)
    for rnd, arm, count, ns in re.findall(r'\[perf-greedy\] round=(\d+) arm=(\d+) tokens=(\d+) ns=(\d+)', raw):
        tokens.add(int(count))
        greedy[int(arm)].append(int(count) * 1e9 / int(ns))
    result = {'width_ms': {}, 'width_round_ms': {}, 'greedy_tok_s': [], 'greedy_samples_tok_s': {}, 'greedy_tokens_per_sample': next(iter(tokens)) if len(tokens) == 1 else None}
    for rows, groups in sorted(widths.items(), key=lambda kv: int(kv[0])):
        rounds = {arm: [statistics.median(v) for (a, r), v in sorted(groups.items()) if a == arm] for arm in [0, 1]}
        result['width_round_ms'][rows] = rounds
        result['width_ms'][rows] = [statistics.median(rounds[arm]) for arm in [0, 1]]
    if greedy:
        result['greedy_tok_s'] = [statistics.median(greedy[arm]) for arm in [0, 1]]
        result['greedy_samples_tok_s'] = dict(greedy)
    result['greedy_identical'] = ('greedy216 byte-identical across three pairs' in raw or raw.count('identical=true') == 3) and 'identical=false' not in raw if greedy else None
    result['verify_argmax_differences'] = raw.count('[perf-width-difference]')
    result['engagement'] = [line for line in raw.splitlines() if 'engaged' in line]
    affine = collections.defaultdict(list)
    for bits, gs, rest in re.findall(r'AFFINE bits=(\d+) gs=(\d+) (rows=\d+ round=\d+ rep=\d+ arm=\d+ ns=\d+)', raw):
        affine[bits + '/' + gs].append('DECODE ' + rest)
    result['affine_ms'] = {fmt: summarize('\n'.join(lines))['width_ms'] for fmt, lines in affine.items()}
    return result


if __name__ == '__main__':
    path = Path(sys.argv[1])
    result = summarize(path.read_text())
    meta = path.with_suffix('.meta.json')
    if meta.exists():
        result['measurement'] = json.loads(meta.read_text())
    out = path.with_name(path.stem + '-summary.json')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
