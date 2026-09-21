//! Characterization coverage for the flat EXL3 component layout. The fixture
//! uses real safetensors headers and data so discovery and resident byte
//! accounting exercise the same index/file paths as a downloaded pack.

const std = @import("std");
const expert_quant = @import("expert_quant.zig");
const model = @import("model.zig");
const model_discovery = @import("model_discovery.zig");

const TensorSpec = struct {
    key: []const u8,
    dtype: []const u8,
    shape: []const u64,
    bytes: usize,
};

const IndexSpec = struct {
    key: []const u8,
    file: []const u8,
};

const TRELLIS_SHAPE = [_]u64{ 2, 1, 1, 64 };
const SIDE_SHAPE = [_]u64{ 2, 16 };
const VECTOR_SHAPE = [_]u64{ 1, 16 };
const MATRIX_SHAPE = [_]u64{ 16, 16 };
const ONE_SHAPE = [_]u64{1};

const EXL3_SPECS = [_]TensorSpec{
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis", .dtype = "U16", .shape = TRELLIS_SHAPE[0..], .bytes = 256 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.suh", .dtype = "F16", .shape = SIDE_SHAPE[0..], .bytes = 64 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.svh", .dtype = "F16", .shape = SIDE_SHAPE[0..], .bytes = 64 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.up_proj.trellis", .dtype = "U16", .shape = TRELLIS_SHAPE[0..], .bytes = 256 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.up_proj.suh", .dtype = "F16", .shape = SIDE_SHAPE[0..], .bytes = 64 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.up_proj.svh", .dtype = "F16", .shape = SIDE_SHAPE[0..], .bytes = 64 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.down_proj.trellis", .dtype = "U16", .shape = TRELLIS_SHAPE[0..], .bytes = 256 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.down_proj.suh", .dtype = "F16", .shape = SIDE_SHAPE[0..], .bytes = 64 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.down_proj.svh", .dtype = "F16", .shape = SIDE_SHAPE[0..], .bytes = 64 },
};

const EMBED_SPEC = TensorSpec{
    .key = "language_model.model.embed_tokens.weight",
    .dtype = "F32",
    .shape = VECTOR_SHAPE[0..],
    .bytes = 64,
};

const LM_HEAD_SPEC = TensorSpec{
    .key = "language_model.lm_head.weight",
    .dtype = "F32",
    .shape = MATRIX_SHAPE[0..],
    .bytes = 1024,
};

const TRUNK_SPEC = TensorSpec{
    .key = "language_model.model.layers.0.self_attn.q_proj.weight",
    .dtype = "F32",
    .shape = MATRIX_SHAPE[0..],
    .bytes = 1024,
};

const TRUNK_AUX_SPEC = TensorSpec{
    .key = "language_model.model.layers.0.self_attn.k_proj.weight",
    .dtype = "F32",
    .shape = MATRIX_SHAPE[0..],
    .bytes = 1024,
};

const MTP_SPECS = [_]TensorSpec{
    .{
        .key = "language_model.mtp.fc_hidden.weight",
        .dtype = "F32",
        .shape = MATRIX_SHAPE[0..],
        .bytes = 1024,
    },
    .{
        .key = "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.trellis",
        .dtype = "U16",
        .shape = TRELLIS_SHAPE[0..],
        .bytes = 256,
    },
    .{
        .key = "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.suh",
        .dtype = "F16",
        .shape = SIDE_SHAPE[0..],
        .bytes = 64,
    },
    .{
        .key = "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.svh",
        .dtype = "F16",
        .shape = SIDE_SHAPE[0..],
        .bytes = 64,
    },
};

const VISION_SPEC = TensorSpec{
    .key = "model.visual.fake",
    .dtype = "F32",
    .shape = ONE_SHAPE[0..],
    .bytes = 4,
};

fn appendFormat(
    allocator: std.mem.Allocator,
    list: *std.ArrayList(u8),
    comptime format: []const u8,
    args: anytype,
) !void {
    const text = try std.fmt.allocPrint(allocator, format, args);
    defer allocator.free(text);
    try list.appendSlice(allocator, text);
}

fn writeSafetensors(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    filename: []const u8,
    specs: []const TensorSpec,
) !void {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    try header.append(allocator, '{');

    var offset: u64 = 0;
    for (specs, 0..) |spec, i| {
        if (i != 0) try header.append(allocator, ',');
        try appendFormat(allocator, &header, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ spec.key, spec.dtype });
        for (spec.shape, 0..) |dim, shape_i| {
            if (shape_i != 0) try header.append(allocator, ',');
            try appendFormat(allocator, &header, "{d}", .{dim});
        }
        const end = offset + @as(u64, @intCast(spec.bytes));
        try appendFormat(allocator, &header, "],\"data_offsets\":[{d},{d}]}}", .{ offset, end });
        offset = end;
    }
    try header.append(allocator, '}');

    const data_len: usize = @intCast(offset);
    const file_bytes = try allocator.alloc(u8, 8 + header.items.len + data_len);
    defer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.items.len, .little);
    @memcpy(file_bytes[8 .. 8 + header.items.len], header.items);
    for (file_bytes[8 + header.items.len ..], 0..) |*byte, i| {
        byte.* = @intCast((i + 17) % 251);
    }
    try dir.writeFile(io, .{ .sub_path = filename, .data = file_bytes });
}

fn writeIndex(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    entries: []const IndexSpec,
) !void {
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(allocator);
    try index.appendSlice(allocator, "{\"weight_map\":{");
    for (entries, 0..) |entry, i| {
        if (i != 0) try index.append(allocator, ',');
        try appendFormat(allocator, &index, "\"{s}\":\"{s}\"", .{ entry.key, entry.file });
    }
    try index.appendSlice(allocator, "}}");
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
}

fn writeFlatFixture(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    donor_dir: std.Io.Dir,
) !void {
    try dir.writeFile(io, .{
        .sub_path = "config.json",
        .data = "{\"model_type\":\"qwen4_exp\",\"hidden_size\":16,\"num_hidden_layers\":1,\"num_experts\":2,\"num_experts_per_tok\":1,\"moe_intermediate_size\":16,\"expert_quant\":{\"format\":\"exl3\",\"k\":4,\"codebook\":\"mul1\"}}",
    });
    try dir.writeFile(io, .{ .sub_path = "ngram_table.bin", .data = "tiny ngram placeholder" });

    try writeSafetensors(io, allocator, dir, "model-experts-L00.safetensors", EXL3_SPECS[0..]);
    try writeSafetensors(io, allocator, dir, "model-embed.safetensors", &.{EMBED_SPEC});
    try writeSafetensors(io, allocator, dir, "model-lm-head.safetensors", &.{LM_HEAD_SPEC});
    try writeSafetensors(io, allocator, dir, "model-trunk-00001-of-00002.safetensors", &.{TRUNK_SPEC});
    try writeSafetensors(io, allocator, dir, "model-mtp.safetensors", MTP_SPECS[0..]);
    try writeSafetensors(io, allocator, dir, "model-vision.safetensors", &.{VISION_SPEC});

    // This donor lives outside the model pack. The canonical flat shard is a
    // hardlink to it, so the indexed loader reads one tensor exactly once.
    try writeSafetensors(io, allocator, donor_dir, "trunk-donor.safetensors", &.{TRUNK_AUX_SPEC});
    try std.Io.Dir.hardLink(
        donor_dir,
        "trunk-donor.safetensors",
        dir,
        "model-trunk-00002-of-00002.safetensors",
        io,
        .{},
    );

    var entries: [EXL3_SPECS.len + MTP_SPECS.len + 5]IndexSpec = undefined;
    var at: usize = 0;
    for (EXL3_SPECS) |spec| {
        entries[at] = .{ .key = spec.key, .file = "model-experts-L00.safetensors" };
        at += 1;
    }
    for (MTP_SPECS) |spec| {
        entries[at] = .{ .key = spec.key, .file = "model-mtp.safetensors" };
        at += 1;
    }
    entries[at] = .{ .key = EMBED_SPEC.key, .file = "model-embed.safetensors" };
    at += 1;
    entries[at] = .{ .key = LM_HEAD_SPEC.key, .file = "model-lm-head.safetensors" };
    at += 1;
    entries[at] = .{ .key = TRUNK_SPEC.key, .file = "model-trunk-00001-of-00002.safetensors" };
    at += 1;
    entries[at] = .{ .key = TRUNK_AUX_SPEC.key, .file = "model-trunk-00002-of-00002.safetensors" };
    at += 1;
    entries[at] = .{ .key = VISION_SPEC.key, .file = "model-vision.safetensors" };
    try writeIndex(io, allocator, dir, entries[0 .. at + 1]);
}

fn createFixture(
    io: std.Io,
    allocator: std.mem.Allocator,
    tmp: *std.testing.TmpDir,
) ![]u8 {
    try tmp.dir.createDirPath(io, "flat-exl3");
    try tmp.dir.createDirPath(io, "external-donor");
    var donor_dir = try tmp.dir.openDir(io, "external-donor", .{});
    defer donor_dir.close(io);
    var dir = try tmp.dir.openDir(io, "flat-exl3", .{});
    defer dir.close(io);
    try writeFlatFixture(io, allocator, dir, donor_dir);

    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    return std.fmt.allocPrint(allocator, "{s}/flat-exl3", .{root_buf[0..root_len]});
}

test "flat EXL3 semantic shards remain a complete discovery candidate" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    const model_path = try createFixture(io, allocator, &tmp);
    defer allocator.free(model_path);

    try std.testing.expectEqual(
        expert_quant.Layout.exl3_k4,
        model_discovery.qwen4StreamingIndexComplete(io, allocator, model_path).?,
    );

    var model_dir = try std.Io.Dir.openDirAbsolute(io, model_path, .{});
    defer model_dir.close(io);
    var shards = model_discovery.indexShardSet(io, model_dir).?;
    defer model_discovery.freeShardSet(&shards);
    try std.testing.expect(shards.contains("model-experts-L00.safetensors"));
    try std.testing.expect(shards.contains("model-embed.safetensors"));
    try std.testing.expect(shards.contains("model-trunk-00001-of-00002.safetensors"));
    try std.testing.expect(shards.contains("model-trunk-00002-of-00002.safetensors"));
    try std.testing.expect(shards.contains("model-lm-head.safetensors"));
    try std.testing.expect(shards.contains("model-mtp.safetensors"));
    try std.testing.expect(shards.contains("model-vision.safetensors"));
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    const donor_path = try std.fmt.allocPrint(allocator, "{s}/external-donor", .{root_buf[0..root_len]});
    defer allocator.free(donor_path);
    const donor_stat = blk: {
        var donor_dir = try std.Io.Dir.openDirAbsolute(io, donor_path, .{});
        defer donor_dir.close(io);
        break :blk try donor_dir.statFile(io, "trunk-donor.safetensors", .{});
    };
    const canonical_stat = try model_dir.statFile(io, "model-trunk-00002-of-00002.safetensors", .{});
    try std.testing.expectEqual(donor_stat.inode, canonical_stat.inode);

    var result = try model_discovery.discoverModels(io, allocator, root_buf[0..root_len]);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 1), result.models.len);
    try std.testing.expectEqualStrings("flat-exl3", result.models[0].id);
    try std.testing.expectEqualStrings("qwen4_exp", result.models[0].model_type);
    try std.testing.expect(result.models[0].streaming_index_complete);
}

test "flat EXL3 resident split drops co-located routed banks and isolates MTP" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    const model_path = try createFixture(io, allocator, &tmp);
    defer allocator.free(model_path);

    const split = try model.streamingResidentSplit(io, allocator, model_path, .exl3_k4);
    // embed (64) + lm_head (1024) + two trunk tensors (2048); the nine routed EXL3
    // tensors, the MTP routed tensor, and model.visual are not trunk bytes.
    try std.testing.expectEqual(@as(u64, 3136), split.trunk);
    try std.testing.expectEqual(@as(u64, 1024), split.mtp);
}

test "flat EXL3 routed-key filtering is independent of shard filename" {
    var key_buf: [256]u8 = undefined;
    for (EXL3_SPECS) |spec| {
        try std.testing.expect(
            model.qwen4StreamingWeightKey(.exl3_k4, &key_buf, spec.key) == null,
        );
    }
    try std.testing.expectEqualStrings(
        EMBED_SPEC.key,
        model.qwen4StreamingWeightKey(.exl3_k4, &key_buf, EMBED_SPEC.key).?,
    );
    try std.testing.expectEqualStrings(
        "language_model.mtp.fc_hidden.weight",
        model.qwen4StreamingWeightKey(.exl3_k4, &key_buf, "language_model.mtp.fc_hidden.weight").?,
    );
}
