const std = @import("std");
const mlx = @import("mlx.zig");
const exl3 = @import("expert_exl3.zig");
const base = @import("expert_exl3_perf_baseline.zig");
const cand = @import("expert_exl3_kernels.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");

test "perf interleaved decode" {
    if (std.c.getenv("MLX_SERVE_EXL3_LAYER_UBENCH") == null) return error.SkipZigTest;
    log.enableStderr();
    const s = mlx.gpuStream();
    const dec = exl3.Decode{ .codebook = .tiny, .window = .w12 };
    base.setDecodeParams(dec);
    cand.setDecodeParams(dec);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E = 256;
    const H = 4096;
    const I = 2048;
    const K = 8;
    var prng = std.Random.DefaultPrng.init(7193);
    const rnd = prng.random();
    var banks: [3]mlx.mlx_array = undefined;
    for (&banks, 0..) |*bank, n| {
        const tr = try alloc.alloc(u16, E * (H / 16) * (I / 16) * 40);
        for (tr) |*v| v.* = rnd.int(u16);
        bank.* = mlx.mlx_array_new_data(tr.ptr, &.{ E, if (n == 2) I / 16 else H / 16, if (n == 2) H / 16 else I / 16, 40 }, 4, .uint16);
        try mlx.check(mlx.mlx_array_eval(bank.*));
        alloc.free(tr);
    }
    defer for (banks) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var scales: [4]mlx.mlx_array = undefined;
    for (&scales, 0..) |*a, n| {
        const dim: c_int = if (n == 0 or n == 3) H else I;
        const sh = try alloc.alloc(u16, @intCast(E * dim));
        @memset(sh, exl3.f32ToF16Bits(if (n == 0 or n == 2) 0.01 else 1));
        a.* = mlx.mlx_array_new_data(sh.ptr, &.{ E, dim }, 2, .float16);
    }
    defer for (scales) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for ([_]c_int{ 1, 2, 3, 4, 5, 6, 7, 8 }) |rows| {
        const xh = try alloc.alloc(u16, @intCast(rows * H));
        for (xh) |*v| v.* = @truncate(@as(u32, @bitCast((rnd.float(f32) * 2 - 1) * 0.1)) >> 16);
        const ids = try alloc.alloc(u32, @intCast(rows * K));
        for (ids, 0..) |*v, i| v.* = @intCast(i * 73 % E);
        const sc = try alloc.alloc(f32, @intCast(rows * K));
        @memset(sc, 0.125);
        const x = mlx.mlx_array_new_data(xh.ptr, &.{ rows, H }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        const slots = mlx.mlx_array_new_data(ids.ptr, &.{rows * K}, 1, .uint32);
        defer _ = mlx.mlx_array_free(slots);
        const scores = mlx.mlx_array_new_data(sc.ptr, &.{rows * K}, 1, .float32);
        defer _ = mlx.mlx_array_free(scores);
        for (0..3) |round| {
            for (0..109) |rep| {
                for (0..2) |offset| {
                    const arm = (offset + rep + round) % 2;
                    base.perfMute();
                    cand.perfMute();
                    var sw = io_util.Stopwatch.init(std.testing.io);
                    const y = if (arm == 0)
                        try base.moeSwigluFused(s, x, banks[0], scales[0], scales[1], banks[1], scales[0], scales[1], banks[2], scales[2], scales[3], slots, scores, .bfloat16)
                    else
                        try cand.moeSwigluFused(s, x, banks[0], scales[0], scales[1], banks[1], scales[0], scales[1], banks[2], scales[2], scales[3], slots, scores, .bfloat16);
                    try mlx.check(mlx.mlx_array_eval(y));
                    const ns = sw.read();
                    _ = mlx.mlx_array_free(y);
                    if (rep >= 100) std.debug.print("DECODE rows={d} round={d} rep={d} arm={d} ns={d}\n", .{ rows, round, rep, arm, ns });
                }
            }
        }
    }
}
