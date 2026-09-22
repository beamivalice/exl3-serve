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

pub fn packedWords(k: u32) usize {
    return TILE_VALUES * @as(usize, k) / 32;
}

pub fn packedHalfwords(k: u32) usize {
    return TILE_VALUES * @as(usize, k) / 16;
}

pub fn kFromPackedDim(packed_hw: usize) ?u32 {
    if (packed_hw % TILE != 0) return null;
    const k = packed_hw / TILE;
    return switch (k) {
        2, 3, 4 => @intCast(k),
        else => null,
    };
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

pub fn unpackTile(words: []const u16, k: u32, out: *[TILE_VALUES]u16) void {
    const bits: usize = k;
    const word_count = bits * TILE_VALUES / 32;
    var thread: usize = 0;
    while (thread < 128) : (thread += 1) {
        const bit0 = thread * 2 * bits + bits + TILE_VALUES * bits - 16;
        const bit2 = bit0 + bits + 16;
        const index0 = bit0 / 32;
        const index1 = (bit2 - 1) / 32;
        const shift: u6 = @intCast((index1 + 1) * 32 - bit2);
        const merged = (@as(u64, wordU32(words, index0 % word_count)) << 32) | @as(u64, wordU32(words, index1 % word_count));
        const funnel: u32 = @truncate(merged >> shift);
        out[thread * 2] = @truncate((funnel >> @intCast(bits)) & 0xFFFF);
        out[thread * 2 + 1] = @truncate(funnel & 0xFFFF);
    }
}

pub fn decodeTile(words: []const u16, k: u32, codebook: Codebook, out: *[TILE_VALUES]u16) void {
    var codewords: [TILE_VALUES]u16 = undefined;
    unpackTile(words, k, &codewords);
    var perm: [TILE_VALUES]usize = undefined;
    tensorCorePerm(&perm);
    for (codewords, 0..) |cw, i| {
        out[perm[i]] = decodeCodeword(cw, codebook);
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
    k: u32,
    codebook: Codebook,
    out: []u16,
) void {
    const in_tiles = in_features / TILE;
    const out_tiles = out_features / TILE;
    const packed_n = packedHalfwords(k);
    var tile_out: [TILE_VALUES]u16 = undefined;
    for (0..in_tiles) |tk| {
        for (0..out_tiles) |tn| {
            const off = (tk * out_tiles + tn) * packed_n;
            decodeTile(trellis[off..][0..packed_n], k, codebook, &tile_out);
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
    k: u32,
    codebook: Codebook,
    out: []u16,
) !void {
    const inner = try allocator.alloc(u16, in_features * out_features);
    defer allocator.free(inner);
    reconstructInner(trellis, in_features, out_features, k, codebook, inner);
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
    k: u32,
    codebook: Codebook,
    out: []f32,
) void {
    const in_tiles = in_features / TILE;
    const out_tiles = out_features / TILE;
    const packed_n = packedHalfwords(k);
    @memset(out, 0);
    var tile_w: [TILE_VALUES]u16 = undefined;
    for (0..in_tiles) |tk| {
        for (0..out_tiles) |tn| {
            const off = (tk * out_tiles + tn) * packed_n;
            decodeTile(trellis[off..][0..packed_n], k, codebook, &tile_w);
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

pub fn innerGemvF32(
    trellis: []const u16,
    transformed: []const f32,
    in_features: usize,
    out_features: usize,
    k: u32,
    codebook: Codebook,
    out: []f32,
) void {
    const in_tiles = in_features / TILE;
    const out_tiles = out_features / TILE;
    const packed_n = packedHalfwords(k);
    @memset(out, 0);
    var tile_w: [TILE_VALUES]u16 = undefined;
    for (0..in_tiles) |tk| {
        for (0..out_tiles) |tn| {
            const off = (tk * out_tiles + tn) * packed_n;
            decodeTile(trellis[off..][0..packed_n], k, codebook, &tile_w);
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
    k: u32,
    codebook: Codebook,
    transformed: []f32,
    inner: []f32,
    out: []f32,
) void {
    prepareInput(x, suh, transformed);
    innerGemv(trellis, transformed, in_features, out_features, k, codebook, inner);
    finishOutput(inner, svh, out);
}

const fixture_bytes = @embedFile("fixtures/exl3_k4_linear.safetensors");

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

test "exl3 packed dim 16 times K is K in 2,3,4" {
    const t = std.testing;
    try t.expectEqual(@as(?u32, 2), kFromPackedDim(32));
    try t.expectEqual(@as(?u32, 3), kFromPackedDim(48));
    try t.expectEqual(@as(?u32, 4), kFromPackedDim(64));
    try t.expectEqual(@as(?u32, null), kFromPackedDim(16));
    try t.expectEqual(@as(?u32, null), kFromPackedDim(80));
    try t.expectEqual(@as(usize, 32), packedHalfwords(2));
    try t.expectEqual(@as(usize, 48), packedHalfwords(3));
    try t.expectEqual(@as(usize, 64), packedHalfwords(4));
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
    reconstructInner(asU16(trellis), in_features, out_features, K4, .mul1, got_inner);
    try t.expectEqualSlices(u16, asU16(inner), got_inner);
    const got_public = try alloc.alloc(u16, in_features * out_features);
    try reconstructPublic(alloc, asU16(trellis), asU16(suh), asU16(svh), in_features, out_features, K4, .mul1, got_public);
    try t.expectEqualSlices(u16, asU16(public), got_public);

    var x: [128]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for (&x) |*v| v.* = rnd.float(f32) * 2 - 1;
    const transformed = try alloc.alloc(f32, 128);
    const inner_y = try alloc.alloc(f32, 128);
    const y = try alloc.alloc(f32, 128);
    project(&x, asU16(trellis), asU16(suh), asU16(svh), 128, 128, K4, .mul1, transformed, inner_y, y);
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

const fixture_k3_bytes = @embedFile("fixtures/exl3_k3_linear.safetensors");
const fixture_k2_bytes = @embedFile("fixtures/exl3_k2_linear.safetensors");

fn decodePackedFixture(
    alloc: std.mem.Allocator,
    raw: []const u8,
    k: u32,
    packed_hw: usize,
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
    try t.expectEqual(packed_hw, trellis.shape[2]);
    const in_features: usize = 128;
    const out_features: usize = 128;
    const got_inner = try alloc.alloc(u16, in_features * out_features);
    reconstructInner(asU16(trellis), in_features, out_features, k, .mul1, got_inner);
    try t.expectEqualSlices(u16, asU16(inner), got_inner);
    const got_public = try alloc.alloc(u16, in_features * out_features);
    try reconstructPublic(alloc, asU16(trellis), asU16(suh), asU16(svh), in_features, out_features, k, .mul1, got_public);
    try t.expectEqualSlices(u16, asU16(public), got_public);
}

test "exl3 K3 packed fixture decodes to the library inner and public f16" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try decodePackedFixture(arena.allocator(), fixture_k3_bytes, 3, 48);
}

test "exl3 K2 packed fixture decodes to the library inner and public f16" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try decodePackedFixture(arena.allocator(), fixture_k2_bytes, 2, 32);
}
