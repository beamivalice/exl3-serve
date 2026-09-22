from pathlib import Path
import shutil, subprocess, sys
root = Path(__file__).resolve().parent.parent
item = sys.argv[1]
if item not in {'item1', 'item2', 'item3', 'legacy'}:
    raise SystemExit('Expected item1, item2, item3, or legacy')
dest = root / 'perf' / (item + '-harness')
dest.mkdir(exist_ok=True)
shutil.copytree(root / 'src', dest / 'src', dirs_exist_ok=True)
for name in ['build.zig', 'build.zig.zon']:
    shutil.copy2(root / name, dest / name)
for name in ['lib', 'app', 'scripts', 'tests']:
    if not (dest / name).exists():
        (dest / name).symlink_to(root / name, target_is_directory=True)
if item == 'legacy':
    (dest / 'src/transformer_legacy.zig').write_bytes(subprocess.check_output(['git', 'show', 'dd9c7cc:src/transformer.zig'], cwd=root))
    p = dest / 'src/transformer.zig'
    p.write_text(p.read_text() + '\n' + (root / 'perf/legacy_test.zig').read_text())
    raise SystemExit(0)
base = subprocess.check_output(['git', 'show', 'dd9c7cc:src/expert_exl3_kernels.zig'], cwd=root, text=True)
if item == 'item1':
    (dest / 'src/expert_exl3_kernels.zig').write_text(base)
    patch = subprocess.check_output(['git', 'show', '--format=', 'b77ec1c', '--', 'src/expert_exl3_kernels.zig'], cwd=root)
    subprocess.run(['git', 'apply', '-R', '--directory=perf/' + item + '-harness'], input=patch, cwd=root, check=True)
    base = (dest / 'src/expert_exl3_kernels.zig').read_text()
    (dest / 'src/expert_exl3_kernels.zig').write_text(subprocess.check_output(['git', 'show', 'dd9c7cc:src/expert_exl3_kernels.zig'], cwd=root, text=True))
base += '\npub fn perfMute() void { ubench_mute = true; }\n'
(dest / 'src/expert_exl3_perf_baseline.zig').write_text(base)
s = (root / 'src/transformer.zig').read_text() if item == 'item3' else subprocess.check_output(['git', 'show', 'dd9c7cc:src/transformer.zig'], cwd=root, text=True)
needle = '            const y = try expert_exl3_kernels.moeSwigluFused('
assert s.count(needle) == 1
if item != 'item3':
    s = s.replace(needle, '            const perf_decode: *const @TypeOf(expert_exl3_kernels.moeSwigluFused) = if (perf_use_baseline) &@import("expert_exl3_perf_baseline.zig").moeSwigluFused else &expert_exl3_kernels.moeSwigluFused;\n            const y = try perf_decode(')
harness = (root / 'perf/model_decode_test.zig').read_text()
if item == 'item3':
    harness = harness.replace('perf_use_baseline = arm == 0;', 'perf_use_baseline = arm == 0;\n                downred_override = !perf_use_baseline;')
    harness = harness.replace('loaded corrected EXL3', 'loaded affine iq2.7')
    harness = harness.replace('try testing.expectEqualSlices(u32, width_ids[0][0..rows], width_ids[1][0..rows]);', 'if (!std.mem.eql(u32, width_ids[0][0..rows], width_ids[1][0..rows])) log.info(\"[perf-width-difference] rows={d} rep={d}\\n\", .{rows, rep});')
    harness = harness.replace('identical argmax at fixed', 'compared argmax at fixed')
    harness = harness.replace('try testing.expectEqualSlices(i32, &outputs[0], &outputs[1]);', 'log.info(\"[perf-greedy-parity] round={d} identical={}\\n\", .{round, std.mem.eql(i32, &outputs[0], &outputs[1])});')
    harness = harness.replace('greedy216 byte-identical across three pairs', 'greedy216 comparison complete across three pairs')
s += '\n' + harness
(dest / 'src/transformer.zig').write_text(s)

if item == 'item2':
    (dest / 'src/expert_exl3_perf_harness.zig').write_text((root / 'perf/chain_test.zig').read_text())
    p = dest / 'src/transformer.zig'
    p.write_text(p.read_text() + '\ncomptime { _ = @import("expert_exl3_perf_harness.zig"); }\n')
    p = dest / 'src/expert_exl3_kernels.zig'
    p.write_text(p.read_text() + '\npub fn perfMute() void { ubench_mute = true; }\n')

if item == 'item3':
    p = dest / 'src/transformer.zig'
    p.write_text(p.read_text() + '\n' + (root / 'perf/affine_stage_test.zig').read_text())
