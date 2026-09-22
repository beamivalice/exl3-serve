//! Per-input-channel activation statistics ("imatrix") captured while the engine
//! serves the bf16 Qwen3.8-Flash-Next checkpoint with expert streaming, in the
//! exact contract `tests/qwen38_flash_next_imatrix_collect.py` writes and
//! `tests/convert_qwen38_flash_next_exl3.py` reads.
//!
//! Opt-in through `MLX_SERVE_IMATRIX_OUT=<abs>.safetensors`; absent or empty = off,
//! and nothing is allocated.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

pub const ENV_VAR = "MLX_SERVE_IMATRIX_OUT";

/// The SOURCE checkpoint's decoder-layer prefix. The converter looks the imatrix
/// up by HF weight name, never by the engine's internal module names.
pub const KEY_PREFIX = "model.language_model.layers.";

/// Absolute output path from the environment, or null when capture is off.
pub fn envPath() ?[]const u8 {
    const raw = std.c.getenv(ENV_VAR) orelse return null;
    const s = std.mem.sliceTo(raw, 0);
    return if (s.len == 0) null else s;
}

const Layer = struct {
    gu: mlx.mlx_array = .{ .ctx = null },
    down: mlx.mlx_array = .{ .ctx = null },
    rows: mlx.mlx_array = .{ .ctx = null },
    tokens: u64 = 0,

    fn deinit(self: *Layer) void {
        if (self.gu.ctx != null) _ = mlx.mlx_array_free(self.gu);
        if (self.down.ctx != null) _ = mlx.mlx_array_free(self.down);
        if (self.rows.ctx != null) _ = mlx.mlx_array_free(self.rows);
        self.* = .{};
    }
};

pub const Collector = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    path: []u8,
    experts: c_int,
    /// arange [1, E] int32 — the comparand every one-hot count broadcasts against.
    axis: mlx.mlx_array,
    layers: []Layer,

    pub fn init(allocator: std.mem.Allocator, s: mlx.mlx_stream, path: []const u8, num_layers: usize, experts: c_int) !*Collector {
        if (experts <= 0 or num_layers == 0) return error.ImatrixBadGeometry;
        const self = try allocator.create(Collector);
        errdefer allocator.destroy(self);
        const owned = try allocator.dupe(u8, path);
        errdefer allocator.free(owned);
        const layers = try allocator.alloc(Layer, num_layers);
        errdefer allocator.free(layers);
        for (layers) |*l| l.* = .{};
        var axis = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(axis);
        try mlx.check(mlx.mlx_arange(&axis, 0, @floatFromInt(experts), 1, .int32, s));
        var axis_2d = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(axis_2d);
        try mlx.check(mlx.mlx_reshape(&axis_2d, axis, &[_]c_int{ 1, experts }, 2, s));
        _ = mlx.mlx_array_free(axis);
        self.* = .{
            .allocator = allocator,
            .s = s,
            .path = owned,
            .experts = experts,
            .axis = axis_2d,
            .layers = layers,
        };
        return self;
    }

    /// `init` from the environment; null when capture is off.
    pub fn fromEnv(allocator: std.mem.Allocator, s: mlx.mlx_stream, num_layers: usize, experts: c_int) !?*Collector {
        const path = envPath() orelse return null;
        const self = try init(allocator, s, path, num_layers, experts);
        log.info("[imatrix] expert activation capture armed: {s} ({d} layers, {d} experts)\n", .{ path, num_layers, experts });
        return self;
    }

    pub fn deinit(self: *Collector) void {
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        _ = mlx.mlx_array_free(self.axis);
        self.allocator.free(self.path);
        self.allocator.destroy(self);
    }

    /// `counts[r, e]` = how many of row r's top-k slots picked expert e. One
    /// [rows, E] compare per slot rather than one [rows, k, E] tensor: k is small
    /// and the 3-D form costs k times the transient at prefill widths.
    fn counts(self: *Collector, ids: mlx.mlx_array, rows: c_int, k: c_int) !mlx.mlx_array {
        var acc = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(acc);
        try mlx.check(mlx.mlx_zeros(&acc, &[_]c_int{ rows, self.experts }, 2, .float32, self.s));
        var j: c_int = 0;
        while (j < k) : (j += 1) {
            var col = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(col);
            try mlx.check(mlx.mlx_slice(&col, ids, &[_]c_int{ 0, j }, 2, &[_]c_int{ rows, j + 1 }, 2, &[_]c_int{ 1, 1 }, 2, self.s));
            var eq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(eq);
            try mlx.check(mlx.mlx_equal(&eq, col, self.axis, self.s));
            var hot = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(hot);
            try mlx.check(mlx.mlx_astype(&hot, eq, .float32, self.s));
            var sum = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(sum);
            try mlx.check(mlx.mlx_add(&sum, acc, hot, self.s));
            _ = mlx.mlx_array_free(acc);
            acc = sum;
        }
        return acc;
    }

    /// `acc[e, c] += sum over rows routed to e of values[r, c]^2`, as one
    /// [E, rows] x [rows, C] matmul against the one-hot count matrix.
    fn addOuter(self: *Collector, acc: *mlx.mlx_array, cnt: mlx.mlx_array, values: mlx.mlx_array) !void {
        var v32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v32);
        try mlx.check(mlx.mlx_astype(&v32, values, .float32, self.s));
        var sq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sq);
        try mlx.check(mlx.mlx_square(&sq, v32, self.s));
        var cnt_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cnt_t);
        try mlx.check(mlx.mlx_transpose(&cnt_t, cnt, self.s));
        var prod = mlx.mlx_array_new();
        mlx.check(mlx.mlx_matmul(&prod, cnt_t, sq, self.s)) catch |e| {
            _ = mlx.mlx_array_free(prod);
            return e;
        };
        try self.fold(acc, prod);
    }

    /// Fold `term` (ownership taken) into `acc` and evaluate: the accumulator is
    /// outside the forward's own graph, so without the eval every chunk's inputs
    /// would stay pinned by an ever-growing add chain.
    fn fold(self: *Collector, acc: *mlx.mlx_array, term: mlx.mlx_array) !void {
        if (acc.ctx == null) {
            acc.* = term;
        } else {
            var sum = mlx.mlx_array_new();
            mlx.check(mlx.mlx_add(&sum, acc.*, term, self.s)) catch |e| {
                _ = mlx.mlx_array_free(sum);
                _ = mlx.mlx_array_free(term);
                return e;
            };
            _ = mlx.mlx_array_free(term);
            _ = mlx.mlx_array_free(acc.*);
            acc.* = sum;
        }
        try mlx.check(mlx.mlx_array_eval(acc.*));
    }

    /// The MLP input rows behind one MoE layer's chunk. `x_rows` is [rows, hidden],
    /// `ids` the [rows, top_k] GLOBAL expert ids (never remapped slab slots).
    pub fn observeGateUp(self: *Collector, layer: usize, x_rows: mlx.mlx_array, ids: mlx.mlx_array) !void {
        if (layer >= self.layers.len) return error.ImatrixLayerOutOfRange;
        const xs = mlx.getShape(x_rows);
        const is = mlx.getShape(ids);
        if (xs.len != 2 or is.len != 2 or is[0] != xs[0] or xs[0] <= 0 or is[1] <= 0) return error.ImatrixBadShape;
        const cnt = try self.counts(ids, is[0], is[1]);
        defer _ = mlx.mlx_array_free(cnt);
        const slot = &self.layers[layer];
        try self.addOuter(&slot.gu, cnt, x_rows);
        var routed = mlx.mlx_array_new();
        mlx.check(mlx.mlx_sum_axis(&routed, cnt, 0, false, self.s)) catch |e| {
            _ = mlx.mlx_array_free(routed);
            return e;
        };
        try self.fold(&slot.rows, routed);
        slot.tokens += @intCast(xs[0]);
    }

    /// The SwiGLU activation rows feeding down. `act_rows` is [n, inter] and `ids`
    /// the [n, 1] global expert id of each of those rows, in the SAME row order.
    pub fn observeDown(self: *Collector, layer: usize, act_rows: mlx.mlx_array, ids: mlx.mlx_array) !void {
        if (layer >= self.layers.len) return error.ImatrixLayerOutOfRange;
        const as = mlx.getShape(act_rows);
        const is = mlx.getShape(ids);
        if (as.len != 2 or is.len != 2 or is[0] != as[0] or is[1] != 1 or as[0] <= 0) return error.ImatrixBadShape;
        const cnt = try self.counts(ids, is[0], 1);
        defer _ = mlx.mlx_array_free(cnt);
        try self.addOuter(&self.layers[layer].down, cnt, act_rows);
    }

    fn scaledFlat(self: *Collector, acc: mlx.mlx_array, denom: mlx.mlx_array) !mlx.mlx_array {
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_divide(&scaled, acc, denom, self.s));
        const shape = mlx.getShape(scaled);
        var flat = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, scaled, &[_]c_int{shape[0] * shape[1]}, 1, self.s));
        try mlx.check(mlx.mlx_array_eval(flat));
        return flat;
    }

    /// Write the safetensors file. Returns its byte count.
    pub fn flush(self: *Collector) !u64 {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        var held: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (held.items) |a| _ = mlx.mlx_array_free(a);
            held.deinit(self.allocator);
        }
        var key_buf: [192]u8 = undefined;
        var entries: usize = 0;
        for (self.layers, 0..) |*slot, li| {
            if (slot.tokens == 0 or slot.gu.ctx == null or slot.down.ctx == null) continue;
            const denom = mlx.mlx_array_new_float(@floatFromInt(slot.tokens));
            defer _ = mlx.mlx_array_free(denom);
            const gu = try self.scaledFlat(slot.gu, denom);
            try held.append(self.allocator, gu);
            const down = try self.scaledFlat(slot.down, denom);
            try held.append(self.allocator, down);
            try mlx.check(mlx.mlx_array_eval(slot.rows));
            const gu_key = try std.fmt.bufPrintSentinel(&key_buf, KEY_PREFIX ++ "{d}.mlp.experts.gate_up_proj", .{li}, 0);
            try mlx.check(mlx.mlx_map_string_to_array_insert(map, gu_key.ptr, gu));
            const rows_key = try std.fmt.bufPrintSentinel(&key_buf, KEY_PREFIX ++ "{d}.mlp.experts.gate_up_proj.rows", .{li}, 0);
            try mlx.check(mlx.mlx_map_string_to_array_insert(map, rows_key.ptr, slot.rows));
            const down_key = try std.fmt.bufPrintSentinel(&key_buf, KEY_PREFIX ++ "{d}.mlp.experts.down_proj", .{li}, 0);
            try mlx.check(mlx.mlx_map_string_to_array_insert(map, down_key.ptr, down));
            entries += 3;
        }
        if (entries == 0) return error.ImatrixNothingCaptured;

        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        _ = mlx.mlx_map_string_to_string_insert(meta, "keys", "SOURCE checkpoint weight names");
        _ = mlx.mlx_map_string_to_string_insert(meta, "values", "experts: sum(x^2)/layer tokens, per expert concatenated");
        _ = mlx.mlx_map_string_to_string_insert(meta, "producer", "mlx-serve " ++ ENV_VAR);

        const path_z = try std.fmt.allocPrintSentinel(self.allocator, "{s}", .{self.path}, 0);
        defer self.allocator.free(path_z);
        try mlx.check(mlx.mlx_save_safetensors(path_z.ptr, map, meta));
        const io = std.Io.Threaded.global_single_threaded.io();
        const stat = std.Io.Dir.cwd().statFile(io, self.path, .{}) catch |e| {
            log.warn("[imatrix] wrote {s}: size unavailable ({s})\n", .{ self.path, @errorName(e) });
            return 0;
        };
        log.info("[imatrix] wrote {s}: {d} entries, {d} bytes\n", .{ self.path, entries, stat.size });
        return stat.size;
    }
};

// ── tests ──

const testing = std.testing;

fn f32At(arr: mlx.mlx_array, i: usize) !f32 {
    try mlx.check(mlx.mlx_array_eval(arr));
    const p = mlx.mlx_array_data_float32(arr) orelse return error.Unreadable;
    return p[i];
}

test "imatrix accumulates routed sums, rows and token counts by expert" {
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    const E: c_int = 3;
    const H: c_int = 2;
    const I: c_int = 2;

    const col = try Collector.init(alloc, s, "/dev/null", 2, E);
    defer col.deinit();

    // rows 0,1; top-2 routing: row0 -> {0,2}, row1 -> {2,2} (a duplicate slot counts twice).
    const x = [_]f32{ 1, 2, 3, 4 };
    const xa = mlx.mlx_array_new_data(&x, &[_]c_int{ 2, H }, 2, .float32);
    defer _ = mlx.mlx_array_free(xa);
    const ids = [_]i32{ 0, 2, 2, 2 };
    const ida = mlx.mlx_array_new_data(&ids, &[_]c_int{ 2, 2 }, 2, .int32);
    defer _ = mlx.mlx_array_free(ida);
    try col.observeGateUp(1, xa, ida);

    const gu = col.layers[1].gu;
    // expert 0: row0 only -> 1, 4;  expert 1: none;  expert 2: row0 + 2*row1 -> 1+18, 4+32
    try testing.expectEqual(@as(f32, 1), try f32At(gu, 0));
    try testing.expectEqual(@as(f32, 4), try f32At(gu, 1));
    try testing.expectEqual(@as(f32, 0), try f32At(gu, 2));
    try testing.expectEqual(@as(f32, 0), try f32At(gu, 3));
    try testing.expectEqual(@as(f32, 19), try f32At(gu, 4));
    try testing.expectEqual(@as(f32, 36), try f32At(gu, 5));

    const rows = col.layers[1].rows;
    try testing.expectEqual(@as(f32, 1), try f32At(rows, 0));
    try testing.expectEqual(@as(f32, 0), try f32At(rows, 1));
    try testing.expectEqual(@as(f32, 3), try f32At(rows, 2));
    try testing.expectEqual(@as(u64, 2), col.layers[1].tokens);

    // The activation rows carry one id each, in row order.
    const act = [_]f32{ 1, 1, 2, 2, 3, 3, 4, 4 };
    const acta = mlx.mlx_array_new_data(&act, &[_]c_int{ 4, I }, 2, .float32);
    defer _ = mlx.mlx_array_free(acta);
    const aids = [_]i32{ 0, 2, 2, 2 };
    const aida = mlx.mlx_array_new_data(&aids, &[_]c_int{ 4, 1 }, 2, .int32);
    defer _ = mlx.mlx_array_free(aida);
    try col.observeDown(1, acta, aida);

    const dn = col.layers[1].down;
    try testing.expectEqual(@as(f32, 1), try f32At(dn, 0));
    try testing.expectEqual(@as(f32, 0), try f32At(dn, 2));
    try testing.expectEqual(@as(f32, 4 + 9 + 16), try f32At(dn, 4));

    // A second chunk adds on top and keeps counting tokens.
    try col.observeGateUp(1, xa, ida);
    try testing.expectEqual(@as(f32, 2), try f32At(col.layers[1].gu, 0));
    try testing.expectEqual(@as(f32, 6), try f32At(col.layers[1].rows, 2));
    try testing.expectEqual(@as(u64, 4), col.layers[1].tokens);

    // An untouched layer stays empty.
    try testing.expect(col.layers[0].gu.ctx == null);
    try testing.expectEqual(@as(u64, 0), col.layers[0].tokens);
}

test "imatrix writes the converter's keys, shapes and layer-normalized values" {
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    const E: c_int = 3;

    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const out = try std.fs.path.join(alloc, &.{ dir, "imatrix.safetensors" });
    defer alloc.free(out);

    const col = try Collector.init(alloc, s, out, 2, E);
    defer col.deinit();

    const x = [_]f32{ 1, 2, 3, 4 };
    const xa = mlx.mlx_array_new_data(&x, &[_]c_int{ 2, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(xa);
    const ids = [_]i32{ 0, 2, 2, 2 };
    const ida = mlx.mlx_array_new_data(&ids, &[_]c_int{ 2, 2 }, 2, .int32);
    defer _ = mlx.mlx_array_free(ida);
    try col.observeGateUp(1, xa, ida);
    const act = [_]f32{ 1, 1, 2, 2, 3, 3, 4, 4 };
    const acta = mlx.mlx_array_new_data(&act, &[_]c_int{ 4, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(acta);
    const aids = [_]i32{ 0, 2, 2, 2 };
    const aida = mlx.mlx_array_new_data(&aids, &[_]c_int{ 4, 1 }, 2, .int32);
    defer _ = mlx.mlx_array_free(aida);
    try col.observeDown(1, acta, aida);

    const bytes = try col.flush();
    try testing.expect(bytes > 0);

    const path_z = try std.fmt.allocPrintSentinel(alloc, "{s}", .{out}, 0);
    defer alloc.free(path_z);
    var loaded = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(loaded);
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    // safetensors Load has no GPU eval — the reader runs on the CPU stream.
    const cpu = mlx.mlx_default_cpu_stream_new();
    try mlx.check(mlx.mlx_load_safetensors(&loaded, &meta, path_z, cpu));

    var gu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gu);
    try mlx.check(mlx.mlx_map_string_to_array_get(&gu, loaded, KEY_PREFIX ++ "1.mlp.experts.gate_up_proj"));
    try testing.expectEqualSlices(c_int, &[_]c_int{6}, mlx.getShape(gu));
    // 2 tokens in the layer: sum(x^2) per expert channel divided by that count.
    try testing.expectEqual(@as(f32, 0.5), try f32At(gu, 0));
    try testing.expectEqual(@as(f32, 9.5), try f32At(gu, 4));

    var dn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dn);
    try mlx.check(mlx.mlx_map_string_to_array_get(&dn, loaded, KEY_PREFIX ++ "1.mlp.experts.down_proj"));
    try testing.expectEqualSlices(c_int, &[_]c_int{6}, mlx.getShape(dn));
    try testing.expectEqual(@as(f32, 14.5), try f32At(dn, 4));

    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_map_string_to_array_get(&rows, loaded, KEY_PREFIX ++ "1.mlp.experts.gate_up_proj.rows"));
    try testing.expectEqualSlices(c_int, &[_]c_int{3}, mlx.getShape(rows));
    try testing.expectEqual(@as(f32, 1), try f32At(rows, 0));
    try testing.expectEqual(@as(f32, 3), try f32At(rows, 2));

    // A layer that never routed contributes no entries.
    var absent = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(absent);
    try testing.expect(mlx.mlx_map_string_to_array_get(&absent, loaded, KEY_PREFIX ++ "0.mlp.experts.gate_up_proj") != 0);
}

test "imatrix capture is off without the environment variable" {
    try testing.expect(envPath() == null);
    const alloc = testing.allocator;
    try testing.expect(try Collector.fromEnv(alloc, mlx.gpuStream(), 4, 8) == null);
}
