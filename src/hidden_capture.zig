//! The residual stream at every block boundary of a prompt forward, appended to
//! one directory while `sushi kld capture` runs the teacher:
//!   boundary-XX.bin   XX = 00..num_layers, raw bf16 row-major [tokens, hidden], no header;
//!                     boundary 0 is the input to layer 0, boundary b the output of layer b-1
//!   tokens.bin        u32 token ids [tokens]
//! Every file is opened for append, so prompts (and runs) concatenate in token
//! order. A prompt's ids land after all of its rows: `tokens.bin` counts only
//! prompts whose rows are complete, which is where a resumed run truncates to.
//!
//! Opt-in through `SUSHI_HIDDEN_OUT=<abs dir>`; absent or empty = off.

const std = @import("std");
const mlx = @import("mlx.zig");

pub const ENV_VAR = "SUSHI_HIDDEN_OUT";

/// The output directory from the environment, or null when capture is off.
pub fn envPath() ?[]const u8 {
    const raw = std.c.getenv(ENV_VAR) orelse return null;
    const s = std.mem.sliceTo(raw, 0);
    return if (s.len == 0) null else s;
}

pub const Writer = struct {
    allocator: std.mem.Allocator,
    hidden: usize,
    /// One append descriptor per boundary, then `tokens`.
    boundaries: []std.c.fd_t,
    tokens: std.c.fd_t = -1,

    pub fn open(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, num_layers: usize, hidden: usize) !*Writer {
        if (num_layers == 0 or hidden == 0) return error.HiddenCaptureBadGeometry;
        try std.Io.Dir.cwd().createDirPath(io, dir);
        const fds = try allocator.alloc(std.c.fd_t, num_layers + 1);
        @memset(fds, -1);
        const self = allocator.create(Writer) catch |e| {
            allocator.free(fds);
            return e;
        };
        self.* = .{ .allocator = allocator, .hidden = hidden, .boundaries = fds };
        errdefer self.close();
        var name: [32]u8 = undefined;
        for (fds, 0..) |*fd, b| fd.* = try openAppend(dir, try std.fmt.bufPrint(&name, "boundary-{d:0>2}.bin", .{b}));
        self.tokens = try openAppend(dir, "tokens.bin");
        return self;
    }

    pub fn close(self: *Writer) void {
        for (self.boundaries) |fd| if (fd >= 0) {
            _ = std.c.close(fd);
        };
        if (self.tokens >= 0) _ = std.c.close(self.tokens);
        self.allocator.free(self.boundaries);
        self.allocator.destroy(self);
    }

    /// One prompt forward: `rows[0]` the residual entering layer 0 and `rows[b]`
    /// layer b-1's output, each [1, n, hidden] or [n, hidden] bf16, for the n `ids`.
    /// Every row is checked before anything is written.
    pub fn append(self: *Writer, s: mlx.mlx_stream, ids: []const u32, rows: []const mlx.mlx_array) !void {
        if (rows.len != self.boundaries.len) return error.HiddenCaptureBoundaryCount;
        const n = ids.len;
        if (n == 0) return error.HiddenCaptureEmpty;
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (rows) |r| {
            if (r.ctx == null) return error.HiddenCaptureMissingBoundary;
            if (mlx.mlx_array_dtype(r) != .bfloat16) return error.HiddenCaptureNotBf16;
            const shape = mlx.getShape(r);
            const fits = switch (shape.len) {
                2 => shape[0] == n and shape[1] == self.hidden,
                3 => shape[0] == 1 and shape[1] == n and shape[2] == self.hidden,
                else => false,
            };
            if (!fits) return error.HiddenCaptureBadShape;
            try mlx.check(mlx.mlx_vector_array_append_value(vec, r));
        }
        try mlx.check(mlx.mlx_eval(vec));
        for (rows, self.boundaries) |r, fd| {
            var c = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(c);
            try mlx.check(mlx.mlx_contiguous(&c, r, false, s));
            try mlx.check(mlx.mlx_array_eval(c));
            const p = mlx.mlx_array_data_bfloat16(c) orelse return error.HiddenCaptureUnreadable;
            try writeAll(fd, std.mem.sliceAsBytes(p[0 .. n * self.hidden]));
        }
        try writeAll(self.tokens, std.mem.sliceAsBytes(ids));
    }
};

/// Output files are private: a new one is created exclusively without following
/// a link, and an existing one is appended to only when it is a regular file with
/// a single link, so a write can never reach another file's bytes.
fn openAppend(dir: []const u8, name: []const u8) !std.c.fd_t {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0);
    const fresh = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .APPEND = true }, @as(std.c.mode_t, 0o644));
    if (fresh >= 0) return fresh;
    if (std.c._errno().* != @intFromEnum(std.c.E.EXIST)) return error.HiddenCaptureOpenFailed;
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .NOFOLLOW = true, .APPEND = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return if (std.c._errno().* == @intFromEnum(std.c.E.LOOP)) error.HiddenCaptureNotPrivateFile else error.HiddenCaptureOpenFailed;
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0 or !std.c.S.ISREG(@intCast(st.mode)) or st.nlink != 1) {
        _ = std.c.close(fd);
        return error.HiddenCaptureNotPrivateFile;
    }
    return fd;
}

fn writeAll(fd: std.c.fd_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const got = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
        if (got < 0) {
            if (std.c._errno().* == @intFromEnum(std.c.E.INTR)) continue;
            return error.HiddenCaptureWriteFailed;
        }
        if (got == 0) return error.HiddenCaptureWriteFailed;
        done += @intCast(got);
    }
}

// ── tests ──

const testing = std.testing;

fn bf16Rows(values: []const f32, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const f = mlx.mlx_array_new_data(values.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

fn readAll(io: std.Io, dir: std.Io.Dir, name: []const u8) ![]u8 {
    return dir.readFileAlloc(io, name, testing.allocator, .limited(1 << 20));
}

test "hidden capture appends each boundary's rows and then the ids, prompt after prompt" {
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const dir = try std.fs.path.join(testing.allocator, &.{ root, "hidden" });
    defer testing.allocator.free(dir);

    // Two layers => three boundaries of hidden 2; bf16-exact values.
    const w = try Writer.open(testing.allocator, io, dir, 2, 2);
    const first = [_][4]f32{ .{ 1, 2, 3, 4 }, .{ 5, 6, 7, 8 }, .{ 9, 10, 11, 12 } };
    var rows: [3]mlx.mlx_array = undefined;
    for (&rows, first) |*r, v| r.* = try bf16Rows(&v, &.{ 1, 2, 2 }, s);
    try w.append(s, &.{ 7, 3 }, &rows);
    for (rows) |r| _ = mlx.mlx_array_free(r);
    const second = [_][2]f32{ .{ -1, -2 }, .{ -3, -4 }, .{ -5, -6 } };
    for (&rows, second) |*r, v| r.* = try bf16Rows(&v, &.{ 1, 2 }, s);
    try w.append(s, &.{9}, &rows);
    for (rows) |r| _ = mlx.mlx_array_free(r);
    w.close();

    var d = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer d.close(io);
    const toks = try readAll(io, d, "tokens.bin");
    defer testing.allocator.free(toks);
    try testing.expectEqual(@as(usize, 3 * 4), toks.len);
    for ([_]u32{ 7, 3, 9 }, 0..) |want, i| try testing.expectEqual(want, std.mem.readInt(u32, toks[i * 4 ..][0..4], .little));
    for (0..3) |b| {
        var name: [32]u8 = undefined;
        const bytes = try readAll(io, d, try std.fmt.bufPrint(&name, "boundary-{d:0>2}.bin", .{b}));
        defer testing.allocator.free(bytes);
        try testing.expectEqual(@as(usize, 3 * 2 * 2), bytes.len);
        for (0..6) |i| {
            const want: f32 = if (i < 4) first[b][i] else second[b][i - 4];
            const bits: u32 = @as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16;
            try testing.expectEqual(want, @as(f32, @bitCast(bits)));
        }
    }

    // A second writer on the same directory appends after what is there.
    const again = try Writer.open(testing.allocator, io, dir, 2, 2);
    for (&rows, second) |*r, v| r.* = try bf16Rows(&v, &.{ 1, 2 }, s);
    try again.append(s, &.{4}, &rows);
    for (rows) |r| _ = mlx.mlx_array_free(r);
    again.close();
    const more = try readAll(io, d, "tokens.bin");
    defer testing.allocator.free(more);
    try testing.expectEqual(@as(usize, 4 * 4), more.len);
}

test "hidden capture refuses a row it cannot store exactly, before writing anything" {
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];

    const w = try Writer.open(testing.allocator, io, dir, 1, 2);
    defer w.close();
    const v = [_]f32{ 1, 2 };
    const good = try bf16Rows(&v, &.{ 1, 2 }, s);
    defer _ = mlx.mlx_array_free(good);
    const wide = mlx.mlx_array_new_data(&v, &[_]c_int{ 1, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(wide);
    const unset = mlx.mlx_array_new();
    try testing.expectError(error.HiddenCaptureNotBf16, w.append(s, &.{1}, &.{ good, wide }));
    try testing.expectError(error.HiddenCaptureMissingBoundary, w.append(s, &.{1}, &.{ good, unset }));
    try testing.expectError(error.HiddenCaptureBadShape, w.append(s, &.{ 1, 2 }, &.{ good, good }));
    try testing.expectError(error.HiddenCaptureBoundaryCount, w.append(s, &.{1}, &.{good}));

    for ([_][]const u8{ "boundary-00.bin", "boundary-01.bin", "tokens.bin" }) |name| {
        const st = try tmp.dir.statFile(io, name, .{});
        try testing.expectEqual(@as(u64, 0), st.size);
    }
}

test "hidden capture never writes through a link: a hard-linked or symlinked output is refused, its target untouched" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    try tmp.dir.writeFile(io, .{ .sub_path = "precious.bin", .data = "checkpoint bytes" });
    const target = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/precious.bin", .{root}, 0);
    defer testing.allocator.free(target);

    for ([_][]const u8{ "hard", "soft" }) |kind| {
        const dir = try std.fmt.allocPrint(testing.allocator, "{s}/{s}", .{ root, kind });
        defer testing.allocator.free(dir);
        try tmp.dir.createDirPath(io, kind);
        const link = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/boundary-01.bin", .{dir}, 0);
        defer testing.allocator.free(link);
        if (std.mem.eql(u8, kind, "hard")) {
            try testing.expectEqual(@as(c_int, 0), std.c.link(target, link));
        } else {
            try testing.expectEqual(@as(c_int, 0), std.c.symlink(target, link));
        }
        try testing.expectError(error.HiddenCaptureNotPrivateFile, Writer.open(testing.allocator, io, dir, 1, 2));
        const kept = try readAll(io, tmp.dir, "precious.bin");
        defer testing.allocator.free(kept);
        try testing.expectEqualStrings("checkpoint bytes", kept);
    }
}

test "hidden capture is off without the environment variable" {
    try testing.expect(envPath() == null);
}
