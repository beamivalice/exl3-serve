#!/bin/bash
# Shared helper for multi-model tests: enumerate model subdirectories under a
# root, filtering out any whose `config.json` declares a `model_type` this
# build does not serve. Must stay in sync with the arch-acceptance gate
# (`served_model_types` in src/model.zig): any other arch is refused at load.
#
# Usage:
#   source "$(dirname "$0")/_lib_supported_models.sh"
#   readarray -t MODELS < <(list_supported_models "$ROOT" [count])
# 'count' is optional; omitted = all.

list_supported_models() {
    local root="$1"
    local limit="${2:-}"
    python3 - "$root" "$limit" <<'PY'
import json, os, sys
root, limit = sys.argv[1], sys.argv[2]
supported = {"qwen4_exp", "mimo_v2"}
# The models root is TWO-LEVEL (`<org>/<repo>`); flat `<repo>` is the legacy shape.
def candidates(root):
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return
    for name in entries:
        if name.startswith("."):
            continue
        if os.path.isfile(os.path.join(root, name, "config.json")):
            yield name
            continue
        try:
            subs = sorted(os.listdir(os.path.join(root, name)))
        except OSError:
            continue
        for sub in subs:
            if sub.startswith("."):
                continue
            if os.path.isfile(os.path.join(root, name, sub, "config.json")):
                yield os.path.join(name, sub)

out = []
for name in candidates(root):
    cfg = os.path.join(root, name, "config.json")
    try:
        with open(cfg) as f:
            data = json.load(f)
        mt = data.get("model_type", "")
        q = data.get("quantization") or {}
        qmode = q.get("mode")
    except Exception:
        continue
    if mt not in supported:
        continue
    # MiMo's routed experts stay native MXFP4; everything else is affine.
    if qmode is not None and qmode not in ("affine", "mxfp4"):
        continue
    out.append(name)
n = int(limit) if limit else len(out)
for name in out[:n]:
    print(name)
PY
}
