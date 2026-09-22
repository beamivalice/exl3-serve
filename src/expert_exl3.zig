const std = @import("std");

pub const TILE: usize = 16;
pub const TILE_VALUES: usize = TILE * TILE;
pub const HAD_DIM: usize = 128;
pub const HAD_SCALE: f32 = 0.08838834764831845;
pub const K4: u32 = 4;
pub const K4_PACKED: usize = TILE_VALUES * K4 / 16;
pub const MCG_MULT: u32 = 0xCBAC1FED;
pub const MUL1_MULT: u32 = 0x83DCD12D;
/// TINY keeps MCG's mix and half-pair sum but ADDS the fixed bits: the mantissa
/// offsets (0x100 low half, 0x200 high half) carry into the exponent with
/// probability 1/4 and 1/2, which is what closes the gap to MUL1.
pub const TINY_MASK: u32 = 0x8FFF8FFF;
pub const TINY_ADD: u32 = 0x32003100;

pub const Codebook = enum(u8) {
    mul1 = 0,
    mcg = 1,
    tiny = 2,

    pub const count = 3;

    pub fn fromName(name: []const u8) ?Codebook {
        if (std.mem.eql(u8, name, "mcg")) return .mcg;
        if (std.mem.eql(u8, name, "mul1")) return .mul1;
        if (std.mem.eql(u8, name, "tiny")) return .tiny;
        return null;
    }
};

/// The codeword width a pack's Viterbi search hashed: a decoder takes the low
/// `w` bits of the 16-bit sliding window before it decodes. The bitstream and
/// the bits per weight are the same at every width — only the value each
/// window decodes to changes. A pack that names no width is w16, which masks
/// nothing. (Unrelated to the GEMM's row windows.)
pub const Window = enum(u8) {
    /// w16 is tag 0, so a zeroed config means "mask nothing" — the pack every
    /// converter wrote before the field existed. The narrowed widths follow in
    /// ascending order from tag 1, which is what `bits` and `fromBits` index.
    w16 = 0,
    w8,
    w9,
    w10,
    w11,
    w12,
    w13,
    w14,
    w15,

    pub const min_bits: u8 = 8;
    pub const count = 16 - min_bits + 1;

    pub fn bits(self: Window) u8 {
        if (self == .w16) return 16;
        return @intFromEnum(self) + min_bits - 1;
    }

    pub fn mask(self: Window) u16 {
        return @truncate((@as(u32, 1) << @intCast(self.bits())) - 1);
    }

    pub fn index(self: Window) usize {
        return @intFromEnum(self);
    }

    pub fn fromBits(w: i64) ?Window {
        if (w == 16) return .w16;
        if (w < min_bits or w > 15) return null;
        return @enumFromInt(w - min_bits + 1);
    }
};

/// How a pack's trellis decodes. The two travel together — a codeword read
/// under the wrong window is as wrong as one read under the wrong codebook.
pub const Decode = struct {
    codebook: Codebook,
    window: Window = .w16,

    pub const mul1: Decode = .{ .codebook = .mul1 };
    pub const mcg: Decode = .{ .codebook = .mcg };
    pub const tiny: Decode = .{ .codebook = .tiny };
};

/// A trellis rate K = n/16 bits per weight, `n` being the halfwords a packed
/// 256-weight tile carries. Weight t's codeword is the 16-bit window ending at
/// `bitEnd(t)`, so it takes `bitEnd(t) - bitEnd(t-1)` fresh bits; the pattern
/// follows from n and is never stored. An integer K is the uniform case.
pub const Rate = struct {
    n: u32,

    pub const min_n: u32 = 32;
    pub const max_n: u32 = 64;

    pub fn fromK(k: u32) Rate {
        return .{ .n = k * @as(u32, TILE) };
    }

    pub fn halfwords(self: Rate) usize {
        return self.n;
    }

    pub fn words(self: Rate) usize {
        return self.n / 2;
    }

    pub fn totalBits(self: Rate) usize {
        return TILE * @as(usize, self.n);
    }

    /// One past the last bit of weight t's codeword window: floor((t+1)*K).
    pub fn bitEnd(self: Rate, t: usize) usize {
        return ((t + 1) * @as(usize, self.n)) >> 4;
    }

    pub fn freshBits(self: Rate, t: usize) usize {
        return self.bitEnd(t) - if (t == 0) 0 else self.bitEnd(t - 1);
    }

    pub fn isInteger(self: Rate) bool {
        return self.n % TILE == 0;
    }

    /// The rate as a human reads it ("2.5"), never the packed halfword count.
    pub fn kText(self: Rate, buf: []u8) []const u8 {
        const k = @as(f64, @floatFromInt(self.n)) / 16.0;
        return std.fmt.bufPrint(buf, "{d}", .{k}) catch "?";
    }
};

pub fn packedWords(k: u32) usize {
    return TILE_VALUES * @as(usize, k) / 32;
}

pub fn packedHalfwords(k: u32) usize {
    return TILE_VALUES * @as(usize, k) / 16;
}

/// K2 to K4 in 1/8-bit steps. An odd n would leave a tile's bitstream short of
/// a whole uint32 word, which every reader indexes by.
pub fn kFromPackedDim(packed_hw: usize) ?Rate {
    if (packed_hw % 2 != 0) return null;
    if (packed_hw < Rate.min_n or packed_hw > Rate.max_n) return null;
    return .{ .n = @intCast(packed_hw) };
}

pub fn f16BitsToF32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

pub fn f32ToF16Bits(value: f32) u16 {
    return @bitCast(@as(f16, @floatCast(value)));
}

pub fn decodeMcg(codeword: u16) u16 {
    const mixed = @as(u32, codeword) *% MCG_MULT;
    const pair = 0x3B603B60 ^ (mixed & 0x8FFF8FFF);
    const lo = f16BitsToF32(@truncate(pair));
    const hi = f16BitsToF32(@truncate(pair >> 16));
    return f32ToF16Bits(lo + hi);
}

pub fn decodeMul1(codeword: u16) u16 {
    const mixed = @as(u32, codeword) *% MUL1_MULT;
    const bytes = std.mem.toBytes(mixed);
    var byte_sum: u32 = 0;
    for (bytes) |b| byte_sum += b;
    const h = f16BitsToF32(@truncate(0x6400 + byte_sum));
    const inverse = f16BitsToF32(0x1EEE);
    const bias = f16BitsToF32(0xC931);
    return f32ToF16Bits(@mulAdd(f32, h, inverse, bias));
}

pub fn decodeTiny(codeword: u16) u16 {
    const mixed = @as(u32, codeword) *% MCG_MULT;
    const pair = (mixed & TINY_MASK) +% TINY_ADD;
    const lo = f16BitsToF32(@truncate(pair));
    const hi = f16BitsToF32(@truncate(pair >> 16));
    return f32ToF16Bits(lo + hi);
}

pub fn decodeCodeword(codeword: u16, codebook: Codebook) u16 {
    return switch (codebook) {
        .mcg => decodeMcg(codeword),
        .mul1 => decodeMul1(codeword),
        .tiny => decodeTiny(codeword),
    };
}

fn wordU32(words: []const u16, index: usize) u32 {
    return @as(u32, words[index * 2]) | (@as(u32, words[index * 2 + 1]) << 16);
}

pub fn unpackTile(words: []const u16, rate: Rate, out: *[TILE_VALUES]u16) void {
    const word_count = rate.words();
    const total = rate.totalBits();
    var thread: usize = 0;
    while (thread < 128) : (thread += 1) {
        const e0 = rate.bitEnd(thread * 2);
        const e1 = rate.bitEnd(thread * 2 + 1);
        // Tail-biting: the pair's window starts 16 bits before e0, one lap up.
        const bit0 = e0 + total - 16;
        const bit2 = e1 + total;
        const index0 = bit0 / 32;
        const index1 = (bit2 - 1) / 32;
        const shift: u6 = @intCast((index1 + 1) * 32 - bit2);
        const merged = (@as(u64, wordU32(words, index0 % word_count)) << 32) | @as(u64, wordU32(words, index1 % word_count));
        const funnel: u32 = @truncate(merged >> shift);
        const fresh: u5 = @intCast(e1 - e0);
        out[thread * 2] = @truncate((funnel >> fresh) & 0xFFFF);
        out[thread * 2 + 1] = @truncate(funnel & 0xFFFF);
    }
}

pub fn decodeTile(words: []const u16, rate: Rate, dec: Decode, out: *[TILE_VALUES]u16) void {
    var codewords: [TILE_VALUES]u16 = undefined;
    unpackTile(words, rate, &codewords);
    var perm: [TILE_VALUES]usize = undefined;
    tensorCorePerm(&perm);
    const mask = dec.window.mask();
    for (codewords, 0..) |cw, i| {
        out[perm[i]] = decodeCodeword(cw & mask, dec.codebook);
    }
}

pub fn tensorCorePerm(out: *[TILE_VALUES]usize) void {
    var thread: usize = 0;
    while (thread < 32) : (thread += 1) {
        const row0 = (thread % 4) * 2;
        const row1 = row0 + 1;
        const row2 = row0 + 8;
        const row3 = row0 + 9;
        const col0 = thread / 4;
        const col1 = col0 + 8;
        const base = thread * 8;
        out[base + 0] = row0 * 16 + col0;
        out[base + 1] = row1 * 16 + col0;
        out[base + 2] = row2 * 16 + col0;
        out[base + 3] = row3 * 16 + col0;
        out[base + 4] = row0 * 16 + col1;
        out[base + 5] = row1 * 16 + col1;
        out[base + 6] = row2 * 16 + col1;
        out[base + 7] = row3 * 16 + col1;
    }
}

fn hadamardEntry(row: usize, col: usize) f32 {
    const bits = @popCount(row & col);
    const sign: f32 = if (bits % 2 == 0) 1.0 else -1.0;
    return sign * HAD_SCALE;
}

pub fn hadamard128(values: *[HAD_DIM]f32) void {
    var tmp: [HAD_DIM]f32 = undefined;
    for (0..HAD_DIM) |r| {
        var acc: f32 = 0;
        for (0..HAD_DIM) |k| acc += hadamardEntry(r, k) * values[k];
        tmp[r] = acc;
    }
    values.* = tmp;
}

pub fn reconstructInner(
    trellis: []const u16,
    in_features: usize,
    out_features: usize,
    rate: Rate,
    dec: Decode,
    out: []u16,
) void {
    const in_tiles = in_features / TILE;
    const out_tiles = out_features / TILE;
    const packed_n = rate.halfwords();
    var tile_out: [TILE_VALUES]u16 = undefined;
    for (0..in_tiles) |tk| {
        for (0..out_tiles) |tn| {
            const off = (tk * out_tiles + tn) * packed_n;
            decodeTile(trellis[off..][0..packed_n], rate, dec, &tile_out);
            for (0..TILE) |r| {
                const dst = (tk * TILE + r) * out_features + tn * TILE;
                @memcpy(out[dst .. dst + TILE], tile_out[r * TILE ..][0..TILE]);
            }
        }
    }
}

pub fn reconstructPublic(
    allocator: std.mem.Allocator,
    trellis: []const u16,
    suh: []const u16,
    svh: []const u16,
    in_features: usize,
    out_features: usize,
    rate: Rate,
    dec: Decode,
    out: []u16,
) !void {
    const inner = try allocator.alloc(u16, in_features * out_features);
    defer allocator.free(inner);
    reconstructInner(trellis, in_features, out_features, rate, dec, inner);
    const w = try allocator.alloc(f32, in_features * out_features);
    defer allocator.free(w);
    for (inner, 0..) |bits, i| w[i] = f16BitsToF32(bits);
    var row_block: usize = 0;
    while (row_block < in_features) : (row_block += HAD_DIM) {
        for (0..out_features) |col| {
            var vec: [HAD_DIM]f32 = undefined;
            for (0..HAD_DIM) |r| vec[r] = w[(row_block + r) * out_features + col];
            hadamard128(&vec);
            for (0..HAD_DIM) |r| w[(row_block + r) * out_features + col] = vec[r];
        }
    }
    for (0..in_features) |r| {
        const s = f16BitsToF32(suh[r]);
        const row = w[r * out_features ..][0..out_features];
        for (row) |*v| v.* *= s;
    }
    var col_block: usize = 0;
    while (col_block < out_features) : (col_block += HAD_DIM) {
        for (0..in_features) |r| {
            var vec: [HAD_DIM]f32 = undefined;
            for (0..HAD_DIM) |c| vec[c] = w[r * out_features + col_block + c];
            hadamard128(&vec);
            for (0..HAD_DIM) |c| w[r * out_features + col_block + c] = vec[c];
        }
    }
    for (0..out_features) |c| {
        const s = f16BitsToF32(svh[c]);
        var r: usize = 0;
        while (r < in_features) : (r += 1) {
            w[r * out_features + c] *= s;
        }
    }
    for (w, 0..) |v, i| out[i] = f32ToF16Bits(v);
}

pub fn prepareInput(x: []const f32, suh: []const u16, out: []f32) void {
    for (x, suh, out) |xv, s, *d| {
        d.* = f16BitsToF32(f32ToF16Bits(xv)) * f16BitsToF32(s);
    }
    var block: usize = 0;
    while (block < out.len) : (block += HAD_DIM) {
        var vec: [HAD_DIM]f32 = undefined;
        @memcpy(&vec, out[block..][0..HAD_DIM]);
        hadamard128(&vec);
        for (0..HAD_DIM) |i| out[block + i] = f16BitsToF32(f32ToF16Bits(vec[i]));
    }
}

pub fn innerGemv(
    trellis: []const u16,
    transformed: []const f32,
    in_features: usize,
    out_features: usize,
    rate: Rate,
    dec: Decode,
    out: []f32,
) void {
    const in_tiles = in_features / TILE;
    const out_tiles = out_features / TILE;
    const packed_n = rate.halfwords();
    @memset(out, 0);
    var tile_w: [TILE_VALUES]u16 = undefined;
    for (0..in_tiles) |tk| {
        for (0..out_tiles) |tn| {
            const off = (tk * out_tiles + tn) * packed_n;
            decodeTile(trellis[off..][0..packed_n], rate, dec, &tile_w);
            const xbase = tk * TILE;
            const ybase = tn * TILE;
            for (0..TILE) |r| {
                const xv = transformed[xbase + r];
                for (0..TILE) |c| {
                    out[ybase + c] += xv * f16BitsToF32(tile_w[r * TILE + c]);
                }
            }
        }
    }
    for (out) |*v| v.* = f16BitsToF32(f32ToF16Bits(v.*));
}

pub fn finishOutput(inner: []const f32, svh: []const u16, out: []f32) void {
    @memcpy(out, inner);
    var block: usize = 0;
    while (block < out.len) : (block += HAD_DIM) {
        var vec: [HAD_DIM]f32 = undefined;
        @memcpy(&vec, out[block..][0..HAD_DIM]);
        hadamard128(&vec);
        @memcpy(out[block..][0..HAD_DIM], &vec);
    }
    for (out, svh) |*v, s| v.* = f16BitsToF32(f32ToF16Bits(v.* * f16BitsToF32(s)));
}

pub fn project(
    x: []const f32,
    trellis: []const u16,
    suh: []const u16,
    svh: []const u16,
    in_features: usize,
    out_features: usize,
    rate: Rate,
    dec: Decode,
    transformed: []f32,
    inner: []f32,
    out: []f32,
) void {
    // Which scratch is which length is the whole contract here, and a wrong
    // one reaches `@memcpy` as silent UB under ReleaseFast.
    std.debug.assert(x.len == in_features and suh.len == in_features and transformed.len == in_features);
    std.debug.assert(svh.len == out_features and inner.len == out_features and out.len == out_features);
    prepareInput(x, suh, transformed);
    innerGemv(trellis, transformed, in_features, out_features, rate, dec, inner);
    finishOutput(inner, svh, out);
}

/// `@embedFile` is byte-aligned and the linker lands it wherever it likes, so a
/// fixture read as u16 through a `[]const u8` is a coin flip on the blob's
/// address. These copies carry the alignment every reader assumes.
pub const fixtures = struct {
    pub const k4 = aligned(@embedFile("fixtures/exl3_k4_linear.safetensors"));
    pub const k3 = aligned(@embedFile("fixtures/exl3_k3_linear.safetensors"));
    pub const k2 = aligned(@embedFile("fixtures/exl3_k2_linear.safetensors"));
    pub const k2p5_tiny = aligned(@embedFile("fixtures/exl3_k2p5_tiny_linear.safetensors"));
    pub const k3_tiny = aligned(@embedFile("fixtures/exl3_k3_tiny_linear.safetensors"));
    /// Searched AND decoded at window 12 by PonyExl3: the only fixture that
    /// certifies a narrowed window against the library rather than against our
    /// own masking of a w16 bitstream.
    pub const k2p5_tiny_w12 = aligned(@embedFile("fixtures/exl3_k2p5_tiny_w12_linear.safetensors"));

    fn aligned(comptime raw: []const u8) *align(8) const [raw.len]u8 {
        const holder = struct {
            const value: [raw.len]u8 align(8) = raw[0..raw.len].*;
        };
        return &holder.value;
    }
};

const fixture_bytes = fixtures.k4;

const TensorView = struct {
    dtype: []const u8,
    shape: []const usize,
    bytes: []const u8,
};

fn parseSafetensors(allocator: std.mem.Allocator, raw: []const u8) !std.StringHashMap(TensorView) {
    if (raw.len < 8) return error.TruncatedSafetensors;
    const header_len = std.mem.readInt(u64, raw[0..8], .little);
    if (8 + header_len > raw.len) return error.TruncatedSafetensors;
    const header = raw[8 .. 8 + header_len];
    const data = raw[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, header, .{});
    defer parsed.deinit();
    var map = std.StringHashMap(TensorView).init(allocator);
    errdefer map.deinit();
    if (parsed.value != .object) return error.BadSafetensorsHeader;
    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
        if (entry.value_ptr.* != .object) continue;
        const obj = entry.value_ptr.*.object;
        const dtype = obj.get("dtype") orelse continue;
        const shape_v = obj.get("shape") orelse continue;
        const offsets = obj.get("data_offsets") orelse continue;
        if (dtype != .string or shape_v != .array or offsets != .array) continue;
        if (offsets.array.items.len != 2) continue;
        const start: usize = @intCast(offsets.array.items[0].integer);
        const end: usize = @intCast(offsets.array.items[1].integer);
        var shape = try allocator.alloc(usize, shape_v.array.items.len);
        for (shape_v.array.items, 0..) |d, i| shape[i] = @intCast(d.integer);
        try map.put(try allocator.dupe(u8, entry.key_ptr.*), .{
            .dtype = try allocator.dupe(u8, dtype.string),
            .shape = shape,
            .bytes = data[start..end],
        });
    }
    return map;
}

fn asU16(view: TensorView) []const u16 {
    return @alignCast(std.mem.bytesAsSlice(u16, view.bytes));
}

test "exl3 packed dim is an even halfword count from K2 to K4 and prints as a rate" {
    const t = std.testing;
    var buf: [8]u8 = undefined;
    try t.expectEqual(@as(u32, 32), kFromPackedDim(32).?.n);
    try t.expectEqual(@as(u32, 40), kFromPackedDim(40).?.n);
    try t.expectEqual(@as(u32, 44), kFromPackedDim(44).?.n);
    try t.expectEqual(@as(u32, 64), kFromPackedDim(64).?.n);
    try t.expectEqual(@as(?Rate, null), kFromPackedDim(16));
    try t.expectEqual(@as(?Rate, null), kFromPackedDim(80));
    try t.expectEqual(@as(?Rate, null), kFromPackedDim(41));
    try t.expectEqualStrings("2", kFromPackedDim(32).?.kText(&buf));
    try t.expectEqualStrings("2.5", kFromPackedDim(40).?.kText(&buf));
    try t.expectEqualStrings("2.75", kFromPackedDim(44).?.kText(&buf));
    try t.expectEqualStrings("3", kFromPackedDim(48).?.kText(&buf));
    try t.expectEqualStrings("4", kFromPackedDim(64).?.kText(&buf));
    try t.expect(kFromPackedDim(48).?.isInteger());
    try t.expect(!kFromPackedDim(40).?.isInteger());
    try t.expectEqual(@as(usize, 32), packedHalfwords(2));
    try t.expectEqual(@as(usize, 48), packedHalfwords(3));
    try t.expectEqual(@as(usize, 64), packedHalfwords(4));
}

test "exl3 bit ends match a hand-computed table at n 40, 44, 48 and 64" {
    const t = std.testing;
    const Case = struct { n: u32, ends: [8]usize, fresh: [16]usize };
    const cases = [_]Case{
        .{
            .n = 40,
            .ends = .{ 2, 5, 7, 10, 12, 15, 17, 20 },
            .fresh = .{ 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3 },
        },
        .{
            .n = 44,
            .ends = .{ 2, 5, 8, 11, 13, 16, 19, 22 },
            .fresh = .{ 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3 },
        },
        .{
            .n = 48,
            .ends = .{ 3, 6, 9, 12, 15, 18, 21, 24 },
            .fresh = .{ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 },
        },
        .{
            .n = 64,
            .ends = .{ 4, 8, 12, 16, 20, 24, 28, 32 },
            .fresh = .{ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 },
        },
    };
    for (cases) |c| {
        const rate = Rate{ .n = c.n };
        for (c.ends, 0..) |want, i| try t.expectEqual(want, rate.bitEnd(i));
        for (c.fresh, 0..) |want, i| try t.expectEqual(want, rate.freshBits(i));
        try t.expectEqual(@as(usize, 16 * c.n), rate.bitEnd(255));
        try t.expectEqual(@as(usize, 16 * c.n), rate.totalBits());
        try t.expectEqual(@as(usize, c.n / 2), rate.words());
        var sum: usize = 0;
        for (0..256) |i| sum += rate.freshBits(i);
        try t.expectEqual(rate.totalBits(), sum);
    }
}

test "exl3 MUL1 codebook pins known codewords" {
    const t = std.testing;
    try t.expectEqual(@as(u16, 49896), decodeMul1(0));
    try t.expectEqual(@as(u16, 14625), decodeMul1(1));
    try t.expectEqual(@as(u16, 47511), decodeMul1(7));
}

test "exl3 TINY codebook pins the converter's codewords" {
    const t = std.testing;
    try t.expectEqual(@as(u16, 13696), decodeTiny(0));
    try t.expectEqual(@as(u16, 15406), decodeTiny(1));
    try t.expectEqual(@as(u16, 49398), decodeTiny(7));
    try t.expectEqual(@as(u16, 48865), decodeTiny(255));
    try t.expectEqual(@as(u16, 46719), decodeTiny(4096));
    try t.expectEqual(@as(u16, 13165), decodeTiny(65535));
}

test "exl3 codebook names resolve and an unknown name is null" {
    const t = std.testing;
    try t.expectEqual(Codebook.tiny, Codebook.fromName("tiny").?);
    try t.expectEqual(Codebook.mul1, Codebook.fromName("mul1").?);
    try t.expectEqual(Codebook.mcg, Codebook.fromName("mcg").?);
    try t.expectEqual(@as(?Codebook, null), Codebook.fromName("mul2"));
}

test "exl3 MUL1 codebook maps a zero codeword to the finite half" {
    const t = std.testing;
    const bits = decodeMul1(0);
    const mixed: u32 = 0;
    var byte_sum: u32 = 0x6400;
    const b = std.mem.toBytes(mixed);
    for (b) |x| byte_sum += x;
    const h = f16BitsToF32(@truncate(byte_sum));
    const inverse = f16BitsToF32(0x1EEE);
    const bias = f16BitsToF32(0xC931);
    const want = f32ToF16Bits(@mulAdd(f32, h, inverse, bias));
    try t.expectEqual(want, bits);
}

test "exl3 MCG codebook maps a zero codeword to the finite half pair" {
    const t = std.testing;
    const bits = decodeMcg(0);
    const mixed: u32 = 0;
    const pair: u32 = 0x3B603B60 ^ (mixed & 0x8FFF8FFF);
    const lo = f16BitsToF32(@truncate(pair));
    const hi = f16BitsToF32(@truncate(pair >> 16));
    const want = f32ToF16Bits(lo + hi);
    try t.expectEqual(want, bits);
}

test "exl3 K4 packed fixture decodes to the library inner and public f16" {
    const t = std.testing;
    try t.expect(fixture_bytes.len > 8);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tensors = try parseSafetensors(alloc, fixture_bytes);
    defer tensors.deinit();
    const trellis = tensors.get("trellis") orelse return error.MissingTrellis;
    const suh = tensors.get("suh") orelse return error.MissingSuh;
    const svh = tensors.get("svh") orelse return error.MissingSvh;
    const inner = tensors.get("inner") orelse return error.MissingInner;
    const public = tensors.get("public") orelse return error.MissingPublic;
    try t.expectEqual(@as(usize, 3), trellis.shape.len);
    try t.expectEqual(@as(usize, 8), trellis.shape[0]);
    try t.expectEqual(@as(usize, 8), trellis.shape[1]);
    try t.expectEqual(@as(usize, 64), trellis.shape[2]);
    const in_features: usize = 128;
    const out_features: usize = 128;
    const got_inner = try alloc.alloc(u16, in_features * out_features);
    reconstructInner(asU16(trellis), in_features, out_features, Rate.fromK(K4), .mul1, got_inner);
    try t.expectEqualSlices(u16, asU16(inner), got_inner);
    const got_public = try alloc.alloc(u16, in_features * out_features);
    try reconstructPublic(alloc, asU16(trellis), asU16(suh), asU16(svh), in_features, out_features, Rate.fromK(K4), .mul1, got_public);
    try t.expectEqualSlices(u16, asU16(public), got_public);

    var x: [128]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for (&x) |*v| v.* = rnd.float(f32) * 2 - 1;
    const transformed = try alloc.alloc(f32, 128);
    const inner_y = try alloc.alloc(f32, 128);
    const y = try alloc.alloc(f32, 128);
    project(&x, asU16(trellis), asU16(suh), asU16(svh), 128, 128, Rate.fromK(K4), .mul1, transformed, inner_y, y);
    const dense = try alloc.alloc(f32, 128);
    @memset(dense, 0);
    const pub_w = asU16(public);
    for (0..128) |o| {
        var acc: f32 = 0;
        for (0..128) |i| acc += x[i] * f16BitsToF32(pub_w[i * 128 + o]);
        dense[o] = acc;
    }
    var ss: f64 = 0;
    var ref: f64 = 0;
    for (y, dense) |g, d| {
        const diff = g - d;
        ss += @as(f64, diff) * @as(f64, diff);
        ref += @as(f64, d) * @as(f64, d);
    }
    const rel = @sqrt(ss / @max(ref, 1e-20));
    try t.expect(rel < 0.02);
}

const fixture_k3_bytes = fixtures.k3;
const fixture_k2_bytes = fixtures.k2;

fn decodePackedFixture(
    alloc: std.mem.Allocator,
    raw: []const u8,
    rate: Rate,
    dec: Decode,
) !void {
    const t = std.testing;
    var tensors = try parseSafetensors(alloc, raw);
    defer tensors.deinit();
    const trellis = tensors.get("trellis") orelse return error.MissingTrellis;
    const suh = tensors.get("suh") orelse return error.MissingSuh;
    const svh = tensors.get("svh") orelse return error.MissingSvh;
    const inner = tensors.get("inner") orelse return error.MissingInner;
    const public = tensors.get("public") orelse return error.MissingPublic;
    try t.expectEqual(@as(usize, 3), trellis.shape.len);
    try t.expectEqual(@as(usize, 8), trellis.shape[0]);
    try t.expectEqual(@as(usize, 8), trellis.shape[1]);
    try t.expectEqual(rate.halfwords(), trellis.shape[2]);
    const in_features: usize = 128;
    const out_features: usize = 128;
    const got_inner = try alloc.alloc(u16, in_features * out_features);
    reconstructInner(asU16(trellis), in_features, out_features, rate, dec, got_inner);
    try t.expectEqualSlices(u16, asU16(inner), got_inner);
    const got_public = try alloc.alloc(u16, in_features * out_features);
    try reconstructPublic(alloc, asU16(trellis), asU16(suh), asU16(svh), in_features, out_features, rate, dec, got_public);
    try t.expectEqualSlices(u16, asU16(public), got_public);
}

test "exl3 K3 packed fixture decodes to the library inner and public f16" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try decodePackedFixture(arena.allocator(), fixture_k3_bytes, Rate.fromK(3), .mul1);
}

test "exl3 K2 packed fixture decodes to the library inner and public f16" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try decodePackedFixture(arena.allocator(), fixture_k2_bytes, Rate.fromK(2), .mul1);
}

const fixture_k2p5_tiny_bytes = fixtures.k2p5_tiny;
const fixture_k3_tiny_bytes = fixtures.k3_tiny;

test "exl3 K2.5 TINY packed fixture decodes to the library inner and public f16" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try decodePackedFixture(arena.allocator(), fixture_k2p5_tiny_bytes, .{ .n = 40 }, .tiny);
}

test "exl3 K3 TINY packed fixture decodes to the library inner and public f16" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try decodePackedFixture(arena.allocator(), fixture_k3_tiny_bytes, .{ .n = 48 }, .tiny);
}

test "exl3 K2.5 TINY w12 packed fixture decodes to the library inner and public f16" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const dec: Decode = .{ .codebook = .tiny, .window = .w12 };
    try decodePackedFixture(alloc, fixtures.k2p5_tiny_w12, .{ .n = 40 }, dec);

    // The window is a decode parameter, not a preference: the same bitstream
    // read at w16 is a different weight matrix, so the fixture certifies w12
    // rather than the decoder's arithmetic alone.
    var tensors = try parseSafetensors(alloc, fixtures.k2p5_tiny_w12);
    defer tensors.deinit();
    const trellis = tensors.get("trellis") orelse return error.MissingTrellis;
    const inner = tensors.get("inner") orelse return error.MissingInner;
    const wide = try alloc.alloc(u16, 128 * 128);
    reconstructInner(asU16(trellis), 128, 128, .{ .n = 40 }, .tiny, wide);
    try t.expect(!std.mem.eql(u16, asU16(inner), wide));
}

test "exl3 every codeword window from 8 to 16 has its own kernel slot" {
    const t = std.testing;
    try t.expect(Window.fromBits(7) == null);
    try t.expect(Window.fromBits(17) == null);
    try t.expectEqual(@as(usize, 0), Window.w16.index());
    var seen: [Window.count]bool = @splat(false);
    var w: i64 = Window.min_bits;
    while (w <= 16) : (w += 1) {
        const win = Window.fromBits(w) orelse return error.TestUnexpectedResult;
        try t.expectEqual(@as(u8, @intCast(w)), win.bits());
        try t.expect(!seen[win.index()]);
        seen[win.index()] = true;
    }
    for (seen) |s| try t.expect(s);
}

test "exl3 the reference decode narrows the codeword window at w8 and w10" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var tensors = try parseSafetensors(arena.allocator(), fixtures.k2p5_tiny);
    defer tensors.deinit();
    const rate: Rate = .{ .n = 40 };
    const tile = asU16(tensors.get("trellis") orelse return error.MissingTrellis)[0..rate.halfwords()];
    var codewords: [TILE_VALUES]u16 = undefined;
    unpackTile(tile, rate, &codewords);
    var perm: [TILE_VALUES]usize = undefined;
    tensorCorePerm(&perm);
    var wide: [TILE_VALUES]u16 = undefined;
    decodeTile(tile, rate, .tiny, &wide);
    for ([_]Window{ .w8, .w10 }) |win| {
        var got: [TILE_VALUES]u16 = undefined;
        decodeTile(tile, rate, .{ .codebook = .tiny, .window = win }, &got);
        var want: [TILE_VALUES]u16 = undefined;
        for (codewords, 0..) |cw, i| want[perm[i]] = decodeCodeword(cw & win.mask(), .tiny);
        try t.expectEqualSlices(u16, &want, &got);
        // The narrowed window is a different weight matrix, not a rounding of
        // the wide one: a pack read at the wrong width decodes to noise.
        try t.expect(!std.mem.eql(u16, &wide, &got));
    }
}
