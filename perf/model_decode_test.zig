var perf_use_baseline: bool = false;

fn perfArgmax(logits: mlx.mlx_array, s: mlx.mlx_stream) !i32 {
    var am = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, logits, -1, false, s));
    try mlx.check(mlx.mlx_array_eval(am));
    const p = mlx.mlx_array_data_uint32(am) orelse return error.Unreadable;
    return @intCast(p[mlx.mlx_array_size(am) - 1]);
}

test "perf whole MiMo interleaved decode" {
    const path = std.c.getenv("PERF_MIMO_MODEL") orelse return error.SkipZigTest;
    if (!diagEnvOn("PERF_REPORT")) return error.SkipZigTest;
    log.enableStderr();
    const alloc = std.heap.page_allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    mlx.installErrorHandler();
    var cfg = try model_mod.parseConfig(io, alloc, std.mem.span(path));
    defer cfg.deinit(alloc);
    cfg.mtp_override = false;
    var weights = try model_mod.loadWeightsForConfig(io, alloc, std.mem.span(path), &cfg, false);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&cfg, &weights);
    var xfm = try Transformer.init(io, alloc, cfg, &weights);
    defer xfm.deinit();
    try xfm.cache.reinit(cfg.num_hidden_layers, KVQuantConfig.affine(8));
    xfm.compileMoeRouting();
    _ = mlx.applyWiredPolicy();
    @import("expert_exl3_perf_baseline.zig").setDecodeParams(.{ .codebook = .tiny, .window = .w12 });
    expert_exl3_kernels.setDecodeParams(.{ .codebook = .tiny, .window = .w12 });
    log.info("[perf-model] loaded corrected EXL3 ctx=8192 kv=8 mtp=false\n", .{});
    var prompt: [1024]i32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(1 + i % 997);
    const pre = mlx.mlx_array_new_data(&prompt, &.{ 1, 1024 }, 2, .int32);
    defer _ = mlx.mlx_array_free(pre);
    const fixed = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    for (1..9) |rows| {
        try xfm.resetCache();
        _ = mlx.mlx_clear_cache();
        var ctx = xfm.defaultCtx();
        const p = try xfm.forwardWith(&ctx, pre);
        try mlx.check(mlx.mlx_array_eval(p));
        _ = mlx.mlx_array_free(p);
        const ids = mlx.mlx_array_new_data(&fixed, &.{ 1, @intCast(rows) }, 2, .int32);
        defer _ = mlx.mlx_array_free(ids);
        var width_ids: [2][8]u32 = undefined;
        for (0..39) |rep| {
            for (0..2) |offset| {
                const arm = (offset + rep) % 2;
                perf_use_baseline = arm == 0;
                try perfRewind(&xfm.cache, &xfm.moe_seq_offset, 1024, xfm.s);
                var sw = @import("io_util.zig").Stopwatch.init(io);
                const y = try xfm.forwardWith(&ctx, ids);
                try mlx.check(mlx.mlx_array_eval(y));
                const ns = sw.read();
                var arg = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_argmax_axis(&arg, y, -1, false, xfm.s));
                try mlx.check(mlx.mlx_array_eval(arg));
                const chosen = mlx.mlx_array_data_uint32(arg) orelse return error.Unreadable;
                @memcpy(width_ids[arm][0..rows], chosen[0..rows]);
                _ = mlx.mlx_array_free(arg);
                _ = mlx.mlx_array_free(y);
                if (rep >= 12) log.info("[perf-width] rows={d} round={d} rep={d} arm={d} ns={d}\n", .{ rows, (rep - 12) / 9, rep, arm, ns });
            }
            try testing.expectEqualSlices(u32, width_ids[0][0..rows], width_ids[1][0..rows]);
        }
        log.info("[perf-width] rows={d} identical argmax at fixed kv=1024\n", .{rows});
    }
    for (0..3) |round| {
        var outputs: [2][216]i32 = undefined;
        for (0..2) |offset| {
            const arm = (offset + round) % 2;
            perf_use_baseline = arm == 0;
            try xfm.resetCache();
            _ = mlx.mlx_clear_cache();
            var ctx = xfm.defaultCtx();
            const p = try xfm.forwardWith(&ctx, pre);
            var next = try perfArgmax(p, xfm.s);
            _ = mlx.mlx_array_free(p);
            var total_ns: u64 = 0;
            for (0..216) |j| {
                outputs[arm][j] = next;
                var sw = @import("io_util.zig").Stopwatch.init(io);
                const one = mlx.mlx_array_new_data(&next, &.{ 1, 1 }, 2, .int32);
                const y = try xfm.forwardWith(&ctx, one);
                next = try perfArgmax(y, xfm.s);
                if (j >= 16) total_ns += sw.read();
                _ = mlx.mlx_array_free(y);
                _ = mlx.mlx_array_free(one);
            }
            log.info("[perf-greedy] round={d} arm={d} tokens=200 ns={d}\n", .{ round, arm, total_ns });
        }
        try testing.expectEqualSlices(i32, &outputs[0], &outputs[1]);
    }
    log.info("[perf-model] greedy216 byte-identical across three pairs\n", .{});
}

fn perfRewind(cache: *KVCache, offset: *usize, len: usize, s: mlx.mlx_stream) !void {
    try cache.truncate(len, s);
    offset.* = len;
}

test "perf rewind restores KV and MoE positions together" {
    var cache = try KVCache.init(testing.allocator, 0);
    defer cache.deinit();
    cache.step = 1025;
    var offset: usize = 1025;
    try perfRewind(&cache, &offset, 1024, mlx.gpuStream());
    try testing.expectEqual(@as(usize, 1024), cache.step);
    try testing.expectEqual(@as(usize, 1024), offset);
}
