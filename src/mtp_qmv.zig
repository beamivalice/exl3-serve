const std = @import("std");
const mlx = @import("mlx");
const dense_rows = @import("mtp_dense_rows.zig");

// Each row retains MLX qmv's lane assignment, affine dot, accumulation order and bf16 rounding.
const SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const uint vec0 = threadgroup_position_in_grid.x * NV;
    \\const uint out_row = threadgroup_position_in_grid.y * 8u + sg * 4u;
    \\const int K = int(K_size);
    \\const int N = int(N_size);
    \\constexpr int VPT = FAST ? 8 : 4;
    \\constexpr int BLOCK = VPT * 32;
    \\const int groups = K / GS;
    \\const device uchar* ws = (const device uchar*)w + size_t(out_row) * K + lane * VPT;
    \\const device T* sp = sc + size_t(out_row) * groups + lane / (GS / VPT);
    \\const device T* bp = bi + size_t(out_row) * groups + lane / (GS / VPT);
    \\const device T* xp[NV];
    \\for (int v = 0; v < NV; ++v) xp[v] = x + size_t(min(vec0 + uint(v), uint(M - 1))) * K + lane * VPT;
    \\float result[NV][4] = {};
    \\int k = 0;
    \\for (; k < (FAST ? K : K - BLOCK); k += BLOCK) {
    \\  float xt[NV][VPT];
    \\  float sums[NV] = {};
    \\  for (int v = 0; v < NV; ++v) {
    \\    for (int i = 0; i < VPT; ++i) { sums[v] += xp[v][i]; xt[v][i] = xp[v][i]; }
    \\  }
    \\  for (int row = 0; row < 4; ++row) {
    \\    uchar codes[VPT];
    \\    for (int i = 0; i < VPT; ++i) codes[i] = ws[row * K + i];
    \\    const float scale = sp[row * groups];
    \\    const float bias = bp[row * groups];
    \\    for (int v = 0; v < NV; ++v) {
    \\      float accum = 0.0f;
    \\      for (int i = 0; i < VPT; ++i) accum += xt[v][i] * codes[i];
    \\      result[v][row] += scale * accum + sums[v] * bias;
    \\    }
    \\  }
    \\  ws += BLOCK;
    \\  sp += BLOCK / GS;
    \\  bp += BLOCK / GS;
    \\  for (int v = 0; v < NV; ++v) xp[v] += BLOCK;
    \\}
    \\if (!FAST) {
    \\  const int remaining = clamp(K - k - int(lane) * VPT, 0, VPT);
    \\  if (remaining > 0) {
    \\    float xt[NV][VPT];
    \\    float sums[NV] = {};
    \\    for (int v = 0; v < NV; ++v) {
    \\      for (int i = 0; i < remaining; ++i) { sums[v] += xp[v][i]; xt[v][i] = xp[v][i]; }
    \\    }
    \\    for (int row = 0; row < 4; ++row) {
    \\      uchar codes[VPT];
    \\      for (int i = 0; i < remaining; ++i) codes[i] = ws[row * K + i];
    \\      const float scale = sp[row * groups];
    \\      const float bias = bp[row * groups];
    \\      for (int v = 0; v < NV; ++v) {
    \\        float accum = 0.0f;
    \\        for (int i = 0; i < remaining; ++i) accum += xt[v][i] * codes[i];
    \\        result[v][row] += scale * accum + sums[v] * bias;
    \\      }
    \\    }
    \\  }
    \\}
    \\for (int v = 0; v < NV; ++v) {
    \\  for (int row = 0; row < 4; ++row) {
    \\    const float value = simd_sum(result[v][row]);
    \\    if (lane == 0 && vec0 + uint(v) < uint(M)) y[size_t(vec0 + uint(v)) * N + out_row + row] = T(value);
    \\  }
    \\}
;

var kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;
const Key = struct { dims: [8]c_int = @splat(0), ndim: usize, n: c_int, gs: c_int };
const Entry = struct { key: Key, cfg: mlx.mlx_fast_metal_kernel_config, k_size: mlx.mlx_array, n_size: mlx.mlx_array, tick: u64 };
var entries: [128]Entry = undefined;
var count: usize = 0;
var tick: u64 = 0;

fn configuration(xs: []const c_int, n: c_int, k: c_int, m: c_int, gs: c_int) !*const Entry {
    var key = Key{ .ndim = xs.len, .n = n, .gs = gs };
    @memcpy(key.dims[0..xs.len], xs);
    tick +%= 1;
    for (entries[0..count]) |*entry| {
        if (std.meta.eql(key, entry.key)) {
            entry.tick = tick;
            return entry;
        }
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const k_size = mlx.mlx_array_new_int(k);
    errdefer _ = mlx.mlx_array_free(k_size);
    const n_size = mlx.mlx_array_new_int(n);
    errdefer _ = mlx.mlx_array_free(n_size);
    var shape = key.dims;
    shape[xs.len - 1] = n;
    const tiles = @divTrunc(m + 3, 4);
    const nv = @divTrunc(m + tiles - 1, tiles);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, xs.len, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, tiles * 32, @divExact(n, 8) * 2, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 2, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "M", m));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", gs));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NV", nv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "FAST", @intFromBool(@mod(k, 256) == 0)));
    var slot = count;
    if (count == entries.len) {
        slot = 0;
        for (entries[0..count], 0..) |entry, i| if (entry.tick < entries[slot].tick) {
            slot = i;
        };
        _ = mlx.mlx_fast_metal_kernel_config_free(entries[slot].cfg);
        _ = mlx.mlx_array_free(entries[slot].k_size);
        _ = mlx.mlx_array_free(entries[slot].n_size);
    } else count += 1;
    entries[slot] = .{ .key = key, .cfg = cfg, .k_size = k_size, .n_size = n_size, .tick = tick };
    return &entries[slot];
}

// Every admitted affine8 geometry uses M=1 arithmetic for each of at most 32 rows.
// Shapes outside the shared tile use separate M=1 calls, never a width-dependent qmm.
pub fn matmul(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array) !?mlx.mlx_array {
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    if (xs.len < 2 or xs.len > 8 or ws.len != 2 or ss.len != 2) return null;
    const k = xs[xs.len - 1];
    const n = ws[0];
    if (k <= 0 or n <= 0 or ws[1] <= 0 or ss[1] <= 0 or mlx.mlx_array_dtype(w) != .uint32) return null;
    const geom = @import("expert_quant.zig").affineGeomFromShapes(@intCast(ws[1]), @intCast(ss[1]), @intCast(k)) orelse return null;
    if (geom.bits != 8 or ss[0] != n or !std.mem.eql(c_int, ss, mlx.getShape(bi))) return null;
    const rows = mlx.mlx_array_size(x) / @as(usize, @intCast(k));
    if (rows < 2 or rows > 32) return null;
    const gs: c_int = @intCast(geom.group_size);
    if (!mlx.streamIsGpu(s) or k < 256 or @mod(n, 8) != 0 or
        mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16)
        return try serialRows(s, x, w, sc, bi, gs, @intCast(rows));
    if (kernel == null) {
        const names = [_][*:0]const u8{ "x", "w", "sc", "bi", "K_size", "N_size" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&names, names.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("mtp_serial_qmv8_rows", iv, ov, SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        kernel = value;
    }
    if (!engaged) {
        engaged = true;
        @import("log").info("[mtp-qmv] row-identical affine8 projections engaged\n", .{});
    }
    const cfg = try configuration(xs, n, k, @intCast(rows), gs);
    const inputs = [_]mlx.mlx_array{ x, w, sc, bi, cfg.k_size, cfg.n_size };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel.?, iv, cfg.cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

pub fn denseMatmul(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array) !?mlx.mlx_array {
    if (x.ctx == null or w.ctx == null) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len < 2 or xs.len > 8 or ws.len != 2 or ws[0] <= 0 or ws[1] <= 0 or xs[xs.len - 1] != ws[0]) return null;
    const rows = mlx.mlx_array_size(x) / @as(usize, @intCast(ws[0]));
    if (rows < 2 or rows > 32) return null;
    if (dense_rows.enabled()) {
        if (try dense_rows.matmul(s, x, w)) |out| return out;
    }
    return try serialRows(s, x, w, .{}, .{}, null, @intCast(rows));
}

fn serialRows(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, gs: ?c_int, rows: c_int) !mlx.mlx_array {
    const xs = mlx.getShape(x);
    const k = xs[xs.len - 1];
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &.{ rows, k }, 2, s));
    var parts: [32]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (parts) |part| {
        if (part.ctx != null) _ = mlx.mlx_array_free(part);
    };
    for (0..@intCast(rows)) |row| {
        var xr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xr);
        try mlx.check(mlx.mlx_slice(&xr, flat, &.{ @intCast(row), 0 }, 2, &.{ @intCast(row + 1), k }, 2, &.{ 1, 1 }, 2, s));
        parts[row] = mlx.mlx_array_new();
        if (gs) |group_size| {
            try mlx.check(mlx.mlx_quantized_matmul(&parts[row], xr, w, sc, bi, true, mlx.mlx_optional_int.some(group_size), mlx.mlx_optional_int.some(8), "affine", s));
        } else {
            try mlx.check(mlx.mlx_matmul(&parts[row], xr, w, s));
        }
    }
    const pv = mlx.mlx_vector_array_new_data(&parts, @intCast(rows));
    defer _ = mlx.mlx_vector_array_free(pv);
    var joined = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(joined);
    try mlx.check(mlx.mlx_concatenate_axis(&joined, pv, 0, s));
    var shape: [8]c_int = undefined;
    @memcpy(shape[0..xs.len], xs);
    shape[xs.len - 1] = mlx.getShape(w)[if (gs == null) @as(usize, 1) else 0];
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, joined, &shape, xs.len, s));
    return out;
}

test "qmv8 gs32 shared rows retain every serial qmv output bit" {
    try testSharedRows(32);
}

test "qmv8 gs64 shared rows retain every serial qmv output bit" {
    try testSharedRows(64);
}

test "qmv8 gs128 shared rows retain every serial qmv output bit" {
    try testSharedRows(128);
}

fn testSharedRows(gs: c_int) !void {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const a = std.testing.allocator;
    var random = std.Random.DefaultPrng.init(0x8a57a);
    const rng = random.random();
    for ([_][2]c_int{ .{ 12288, 2560 }, .{ 2560, 6144 }, .{ 512, 640 }, .{ 10240, 384 }, .{ 7, 256 }, .{ 8, 128 }, .{ 1, 128 }, .{ 13, 512 }, .{ 9, gs }, .{ 8, gs * 3 }, .{ 8, gs * 9 }, .{ 48, 2560 } }) |dims| {
        const n = dims[0];
        const k = dims[1];
        const wh = try a.alloc(u32, @intCast(n * @divExact(k, 4)));
        defer a.free(wh);
        const sh = try a.alloc(u16, @intCast(n * @divExact(k, gs)));
        defer a.free(sh);
        const bh = try a.alloc(u16, sh.len);
        defer a.free(bh);
        for (wh) |*v| v.* = rng.int(u32);
        for (sh, bh) |*sc, *bi| {
            sc.* = @truncate(@as(u32, @bitCast(0.001 + rng.float(f32) * 0.001)) >> 16);
            bi.* = @truncate(@as(u32, @bitCast(-0.12 + rng.float(f32) * 0.01)) >> 16);
        }
        const w = mlx.mlx_array_new_data(wh.ptr, &[_]c_int{ n, @divExact(k, 4) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(w);
        const sc = mlx.mlx_array_new_data(sh.ptr, &[_]c_int{ n, @divExact(k, gs) }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(sc);
        const bi = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{ n, @divExact(k, gs) }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(bi);
        for (2..10) |rows| {
            const xh = try a.alloc(u16, rows * @as(usize, @intCast(k)));
            defer a.free(xh);
            for (xh) |*v| v.* = @truncate(@as(u32, @bitCast((rng.float(f32) - 0.5) * 0.4)) >> 16);
            const x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), k }, 2, .bfloat16);
            defer _ = mlx.mlx_array_free(x);
            const together = (try matmul(s, x, w, sc, bi)) orelse {
                std.debug.print("qmv8 declined gs={d} N={d} K={d} M={d}\n", .{ gs, n, k, rows });
                return error.KernelDeclined;
            };
            defer _ = mlx.mlx_array_free(together);
            for (0..rows) |row| {
                var xr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(xr);
                var yr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(yr);
                try mlx.check(mlx.mlx_slice(&xr, x, &[_]c_int{ @intCast(row), 0 }, 2, &[_]c_int{ @intCast(row + 1), k }, 2, &[_]c_int{ 1, 1 }, 2, s));
                try mlx.check(mlx.mlx_slice(&yr, together, &[_]c_int{ @intCast(row), 0 }, 2, &[_]c_int{ @intCast(row + 1), n }, 2, &[_]c_int{ 1, 1 }, 2, s));
                var reference = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(reference);
                try mlx.check(mlx.mlx_quantized_matmul(&reference, xr, w, sc, bi, true, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(8), "affine", s));
                try mlx.check(mlx.mlx_array_eval(reference));
                try mlx.check(mlx.mlx_array_eval(yr));
                const rp = mlx.mlx_array_data_bfloat16(reference) orelse return error.Unreadable;
                const yp = mlx.mlx_array_data_bfloat16(yr) orelse return error.Unreadable;
                for (0..@intCast(n)) |i| {
                    if (rp[i] != yp[i]) {
                        std.debug.print("qmv8 gs={d} N={d} K={d} M={d} row={d} col={d}: serial={x} shared={x}\n", .{ gs, n, k, rows, row, i, rp[i], yp[i] });
                        return error.QmvRowNotIdentical;
                    }
                }
            }
        }
    }
}
