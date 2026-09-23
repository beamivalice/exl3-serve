# EXL3-serve

A fork from ddalcu's MLX-serve, focus only to support selected EXL3 models in Apple Silicon.

## Model support list

* Qwen3.8-Flash-Next
* MiMo-V2.6-Flash-RL — experimental text-only native MXFP4 expert streaming

## MiMo-V2.6-Flash-RL streaming

The original HF checkpoint loads directly with expert streaming. Expert bytes
remain native MXFP4 and are read from the original shards. At load time, the
engine reconstructs rank-local FP8 QKV weights, splits Q/K/V, and converts FP8
trunk linears to affine 8-bit **in memory**. Source files are not modified;
no Python or converted copy is required. Vision/audio and MTP are omitted.

```sh
./.zig-toolchain/zig build -Doptimize=ReleaseFast
./zig-out/bin/mlx-serve run /path/to/MiMo-V2.6-Flash-RL \
  --host 127.0.0.1 --ssd-budget-gb 100 --kv-quant 8 \
  --ctx-size 4096 --prefill-chunk 512 --max-tokens 512 \
  --no-mtp --no-pld --no-vision --prefix-cache-entries 0
```

Existing converted MXFP4 packs remain supported. The optional pack converter
(PonyExl3, `python -m ponyexl3.serve_convert pack-mimo-v2`) performs the same
trunk preparation ahead of time. Both paths preload 80% of expert-cache slots
before readiness.

`--ssd-budget-gb` is a total resident target in GiB, including the trunk,
expert cache, and streaming workspaces. It must leave room for KV and serving
transients under the machine's wired limit. Reduce the budget if admission
refuses a longer conversation. MTP and multimodal input are not supported in
this bring-up; decode is serial.

`tests/test_mimo_streaming.sh` is the live smoke test.
`tests/bench_mimo_streaming.py` records server-side prefill/decode timings with
one warm-up and three measured repeats, without prefix reuse or speculation.

## Component-sharded EXL3 packs

The flat component layout keeps embeddings, the output head, two layer-aligned
trunk shards, one routed-expert shard per layer, MTP, and vision separate.
Each expert shard contains gate/up/down trellis and scale tensors.
`model.safetensors.index.json` maps the unchanged tensor names to these files;
`ngram_table.bin` stays outside safetensors. Legacy shard names still work.

Repack an existing K3 **or** K4 pack with the Python standard library:

```sh
python -m ponyexl3.serve_convert repack /path/to/old-pack /path/to/new-pack
```

This creates a new directory, never overwrites an existing one, and copies tensor
payloads without decoding or requantizing. Each written payload is verified with
SHA-256. The index records tensor bytes, not file-header overhead. Source packs
must remain unchanged during the operation.

For a development machine with both variants:

```sh
python -m ponyexl3.serve_convert repack /models/old-k3 /models/k3-components
python -m ponyexl3.serve_convert repack /models/old-k4 /models/k4-components \
  --share-with /models/k3-components
```

`--share-with` hard-links only byte-identical component files and n-gram tables.
Different experts remain separate. Both directories are self-contained: either
can be copied to another machine or retained after removing the other.
Hard-link sharing requires the same filesystem; the command refuses a
cross-filesystem sharing attempt rather than silently duplicating shared data.
The source n-gram table is also hard-linked when possible, even for a single pack.
Without a same-filesystem source table, it is copied and verified.

**Treat hard-linked files as immutable.** Replace a file to change it; do not
edit its contents in place, because all links refer to the same bytes.
Ordinary directory-size listings can count shared files more than once.
The repacker reports which files share storage.

The converter also accepts `--component-output NEW_DIR` after its normal
conversion, `--from-exl3` restack, or `--dense` compose stage, plus optional
`--share-with EXISTING_COMPONENT_PACK`. The original `--dst` directory remains
the resumable staging output; the component destination must be separate and new.

Verify a repacked model in your normal serving workflow before replacing the
old pack. Repacking does not delete originals, so keeping both temporarily uses
extra space for rewritten weights, but not for a hard-linked n-gram table.

The repacker and its hermetic tests live in PonyExl3
(`ponyexl3/serve_convert/`, `tests/test_repack_exl3.py`). What a pack owes this
engine is `docs/pack-format.md`.

### Local K3/K4 launchers

`scripts/exl3-qwen38flash-k3.sh` and `scripts/exl3-qwen38flash-k4.sh` use
`$HOME/llm/exl3-serve/zig-out/bin/mlx-serve` and the corresponding
`$HOME/llm/models/Qwen3.8-Flash-Next-EXL3-K3` or `Qwen3.8-Flash-Next-EXL3-K4`.
They can also be copied directly into `$HOME/llm/`.

Both mirror the local `mlx-serve-qwen38flash.sh` settings: loopback port 11234,
1M context, 8192 prefill chunk, one concurrent request, 8-bit KV, MTP, and
RAM/SSD prefix caching. Run one at a time; trailing arguments can override
defaults, for example:

```sh
./scripts/exl3-qwen38flash-k4.sh
./scripts/exl3-qwen38flash-k3.sh --port 11235 --no-mtp
```

Build the executable in ReleaseFast first. The launchers do not build or start
an SSD expert-streaming configuration automatically.
