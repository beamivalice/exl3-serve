//! Direct loader for the raw MiMo-V2.6-Flash HF text trunk.
//!
//! The source checkpoint is not an MLX checkpoint: routed experts remain
//! individual native tensors for the expert-store adapter, while resident
//! trunk projections use the converter's FP8 -> bf16 -> affine-8 contract.

const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const expert_exl3 = @import("expert_exl3.zig");
const expert_quant = @import("expert_quant.zig");

const Allocator = std.mem.Allocator;

const MAX_HEADER_BYTES: u64 = 512 * 1024 * 1024;
const MAX_JSON_BYTES: usize = 512 * 1024 * 1024;
const FP8_BLOCK: u64 = 128;
const AFFINE_GROUP: u64 = 64;

const DType = enum {
    bf16,
    f16,
    f32,
    u16,
    u32,
    fp8_e4m3,
    unknown,
};

const TensorMeta = struct {
    dtype: DType,
    shape: []const u64,
    data_start: u64,
    data_end: u64,
    data_base: u64,
    file: []const u8,
};

const SourceIndex = struct {
    weight_map: std.StringHashMap([]const u8),
    files: std.StringHashMap(void),
    tensors: std.StringHashMap(TensorMeta),
    /// Per shard, what the converter stamped into `__metadata__`.
    stamps: std.StringHashMap(ShardStamp),
};

/// The decoder a shard was written for, as `tests/convert_mimo_v26_exl3.py`
/// stamps it. Every value is a string there. A shard naming none predates the
/// stamp and is admitted; one that names a decoder the config does not is
/// refused, because the same bytes decode to different weights under each.
const ShardStamp = struct {
    k: ?[]const u8 = null,
    codebook: ?[]const u8 = null,
    window: ?[]const u8 = null,
};

const TensorKind = enum {
    resident,
    /// A stacked routed-expert bank (EXL3 trellis or affine): resident bytes
    /// the kernels read as they are.
    routed_expert,
    fp8_weight,
    fp8_scale,
    skipped,
};

const QkvGeometry = struct {
    q_rows: u64,
    k_rows: u64,
    v_rows: u64,

    fn total(self: QkvGeometry) u64 {
        return self.q_rows + self.k_rows + self.v_rows;
    }
};

const QkvTp = struct {
    tp: usize,
    rows_per_rank: usize,
    blocks_per_rank: usize,
};

fn freeArray(arr: *mlx.mlx_array) void {
    if (arr.ctx != null) _ = mlx.mlx_array_free(arr.*);
    arr.* = .{};
}

const QuantTriple = struct {
    weight: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,

    fn deinit(self: *QuantTriple) void {
        freeArray(&self.weight);
        freeArray(&self.scales);
        freeArray(&self.biases);
    }
};

const DecodedQkv = struct {
    q: []f32,
    k: []f32,
    v: []f32,
    allocator: Allocator,

    fn deinit(self: *DecodedQkv) void {
        self.allocator.free(self.q);
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.* = undefined;
    }
};

pub fn loadWeights(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    config: *const model.ModelConfig,
) !model.Weights {
    try validateConfig(config);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const scratch = arena.allocator();
    var source = try loadSourceIndex(io, scratch, model_dir);
    try validatePlan(&source, scratch, config);

    var weights = model.Weights.init(allocator);
    errdefer weights.deinit();

    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);

    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const meta = entry.value_ptr.*;
        switch (try classifyKey(key, config)) {
            .skipped, .fp8_scale => {},
            .resident, .routed_expert => {
                const raw = try readTensor(allocator, model_dir, meta);
                defer allocator.free(raw);
                var arr = try uploadDense(raw, meta, stream);
                errdefer _ = mlx.mlx_array_free(arr);
                try putWeight(&weights, allocator, key, arr);
                arr = .{};
            },
            .fp8_weight => {
                try loadFp8Weight(
                    &weights,
                    allocator,
                    model_dir,
                    config,
                    key,
                    meta,
                    &source,
                    stream,
                );
            },
        }
    }

    if (weights.count() == 0) return error.NoResidentMimoWeights;
    return weights;
}

/// Exact post-affine-8 resident byte count for the raw source trunk.
pub fn residentBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !u64 {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir))
        return error.InvalidMimoModelPath;
    var config = try model.parseConfig(io, allocator, model_dir);
    defer config.deinit(allocator);
    return residentBytesWithConfig(io, allocator, model_dir, &config);
}

/// Variant for callers that already parsed the source config. The returned
/// value counts only arrays emitted into `model.Weights`; expert banks, MTP,
/// media, and the source FP8 scale grids are not resident trunk bytes.
pub fn residentBytesWithConfig(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    config: *const model.ModelConfig,
) !u64 {
    try validateConfig(config);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const scratch = arena.allocator();
    var source = try loadSourceIndex(io, scratch, model_dir);
    try validatePlan(&source, scratch, config);
    return countResidentBytes(&source, scratch, config);
}

/// E4M3FN as used by `convert_mimo_v2.py`: exponent 0 is subnormal,
/// exponent 15/mantissa 7 is NaN, and the other exponent-15 values are
/// finite (the maximum is 448).
pub fn fp8E4M3FnValue(code: u8) f32 {
    const sign: f32 = if ((code & 0x80) != 0) -1.0 else 1.0;
    const exponent: u8 = (code >> 3) & 0x0f;
    const mantissa: u8 = code & 0x07;
    if (exponent == 0x0f and mantissa == 0x07) return std.math.nan(f32);
    if (exponent == 0) {
        return sign * (@as(f32, @floatFromInt(mantissa)) / 8.0) * @exp2(-6.0);
    }
    return sign *
        (1.0 + @as(f32, @floatFromInt(mantissa)) / 8.0) *
        @exp2(@as(f32, @floatFromInt(exponent)) - 7.0);
}

fn validateConfig(config: *const model.ModelConfig) !void {
    if (config.num_hidden_layers == 0 or config.num_hidden_layers > 128 or
        config.hidden_size == 0 or config.vocab_size == 0 or
        config.num_attention_heads == 0 or config.num_key_value_heads == 0 or
        config.head_dim == 0 or config.v_head_dim == 0 or
        config.intermediate_size == 0)
        return error.InvalidMimoGeometry;
    if (config.num_hidden_layers > 1 and config.num_experts == 0)
        return error.InvalidMimoGeometry;

    for (0..config.num_hidden_layers) |i| {
        const layer: u32 = @intCast(i);
        const heads = config.layerNumHeads(layer);
        const kv = config.layerKVHeads(layer);
        const hd = config.layerHeadDim(layer);
        const vd = config.layerVHeadDim(layer);
        if (heads == 0 or kv == 0 or hd == 0 or vd == 0 or heads % kv != 0)
            return error.InvalidMimoGeometry;
        if (@as(u64, heads) * hd > std.math.maxInt(u32) or
            @as(u64, kv) * hd > std.math.maxInt(u32) or
            @as(u64, kv) * vd > std.math.maxInt(u32))
            return error.InvalidMimoGeometry;
    }
}

fn readFileAlloc(
    io: std.Io,
    allocator: Allocator,
    path: []const u8,
    limit: usize,
) ![]u8 {
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch
        return error.MimoSourceFileMissing;
    defer file.close(io);
    var read_buf: [8192]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    return reader.interface.allocRemaining(allocator, .limited(limit)) catch
        return error.MimoSourceRead;
}

fn preadExact(fd: std.c.fd_t, dst: []u8, offset: u64) !void {
    var done: usize = 0;
    while (done < dst.len) {
        const got = std.c.pread(
            fd,
            dst[done..].ptr,
            dst.len - done,
            @intCast(offset + done),
        );
        if (got < 0) {
            if (std.c._errno().* == @backingInt(std.c.E.INTR)) continue;
            return error.MimoSourceRead;
        }
        if (got == 0) return error.SafetensorsUnexpectedEof;
        done += @intCast(got);
    }
}

fn shardPath(allocator: Allocator, model_dir: []const u8, file: []const u8) ![:0]u8 {
    return std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, file }, 0);
}

const HeaderBytes = struct {
    bytes: []u8,
    data_base: u64,
};

fn readShardHeader(
    allocator: Allocator,
    model_dir: []const u8,
    file: []const u8,
) !HeaderBytes {
    const path = try shardPath(allocator, model_dir, file);
    defer allocator.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.MissingMimoShard;
    defer _ = std.c.close(fd);

    var header_len_bytes: [8]u8 = undefined;
    try preadExact(fd, &header_len_bytes, 0);
    const header_len = std.mem.readInt(u64, &header_len_bytes, .little);
    if (header_len > MAX_HEADER_BYTES) return error.SafetensorsHeaderTooLarge;
    const header_len_usize = std.math.cast(usize, header_len) orelse
        return error.SafetensorsHeaderTooLarge;
    const header = try allocator.alloc(u8, header_len_usize);
    errdefer allocator.free(header);
    try preadExact(fd, header, 8);
    const data_base = std.math.add(u64, 8, header_len) catch
        return error.InvalidSafetensorsHeader;
    return .{ .bytes = header, .data_base = data_base };
}

fn parseDType(name: []const u8) DType {
    if (std.mem.eql(u8, name, "BF16")) return .bf16;
    if (std.mem.eql(u8, name, "F16")) return .f16;
    if (std.mem.eql(u8, name, "F32")) return .f32;
    if (std.mem.eql(u8, name, "U16")) return .u16;
    if (std.mem.eql(u8, name, "U32")) return .u32;
    if (std.mem.eql(u8, name, "F8_E4M3") or std.mem.eql(u8, name, "F8_E4M3FN"))
        return .fp8_e4m3;
    return .unknown;
}

fn parseShape(allocator: Allocator, value: std.json.Value) ![]u64 {
    if (value != .array) return error.InvalidSafetensorsHeader;
    const shape = try allocator.alloc(u64, value.array.items.len);
    errdefer allocator.free(shape);
    for (value.array.items, 0..) |dim, i| {
        if (dim != .integer or dim.integer < 0) return error.InvalidSafetensorsHeader;
        shape[i] = std.math.cast(u64, dim.integer) orelse
            return error.InvalidSafetensorsHeader;
    }
    return shape;
}

fn parseOffsets(value: std.json.Value) ![2]u64 {
    if (value != .array or value.array.items.len != 2) return error.InvalidSafetensorsHeader;
    var offsets: [2]u64 = undefined;
    for (value.array.items, 0..) |v, i| {
        if (v != .integer or v.integer < 0) return error.InvalidSafetensorsHeader;
        offsets[i] = std.math.cast(u64, v.integer) orelse
            return error.InvalidSafetensorsHeader;
    }
    if (offsets[1] < offsets[0]) return error.InvalidSafetensorsHeader;
    return offsets;
}

fn parseTensorMeta(
    allocator: Allocator,
    value: std.json.Value,
    file: []const u8,
    data_base: u64,
) !TensorMeta {
    if (value != .object) return error.InvalidSafetensorsHeader;
    const dtype_value = value.object.get("dtype") orelse return error.InvalidSafetensorsHeader;
    if (dtype_value != .string) return error.InvalidSafetensorsHeader;
    const shape_value = value.object.get("shape") orelse return error.InvalidSafetensorsHeader;
    const offsets_value = value.object.get("data_offsets") orelse
        return error.InvalidSafetensorsHeader;
    const shape = try parseShape(allocator, shape_value);
    errdefer allocator.free(shape);
    const offsets = try parseOffsets(offsets_value);
    _ = std.math.add(u64, data_base, offsets[1]) catch
        return error.InvalidSafetensorsHeader;
    return .{
        .dtype = parseDType(dtype_value.string),
        .shape = shape,
        .data_start = offsets[0],
        .data_end = offsets[1],
        .data_base = data_base,
        .file = file,
    };
}

fn loadSourceIndex(io: std.Io, allocator: Allocator, model_dir: []const u8) !SourceIndex {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir))
        return error.InvalidMimoModelPath;

    const index_path = try std.fmt.allocPrint(
        allocator,
        "{s}/model.safetensors.index.json",
        .{model_dir},
    );
    const index_bytes = try readFileAlloc(io, allocator, index_path, MAX_JSON_BYTES);
    const parsed = std.json.parseFromSliceLeaky(
        std.json.Value,
        allocator,
        index_bytes,
        .{},
    ) catch return error.InvalidSafetensorsIndex;
    if (parsed != .object) return error.InvalidSafetensorsIndex;
    const weight_map_value = parsed.object.get("weight_map") orelse
        return error.InvalidSafetensorsIndex;
    if (weight_map_value != .object or weight_map_value.object.count() == 0)
        return error.InvalidSafetensorsIndex;

    var source = SourceIndex{
        .weight_map = std.StringHashMap([]const u8).init(allocator),
        .files = std.StringHashMap(void).init(allocator),
        .tensors = std.StringHashMap(TensorMeta).init(allocator),
        .stamps = std.StringHashMap(ShardStamp).init(allocator),
    };
    var it = weight_map_value.object.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.* != .string or entry.value_ptr.string.len == 0)
            return error.InvalidSafetensorsIndex;
        if (source.weight_map.contains(entry.key_ptr.*))
            return error.AmbiguousSafetensorsIndex;
        try source.weight_map.put(entry.key_ptr.*, entry.value_ptr.string);
        try source.files.put(entry.value_ptr.string, {});
    }

    var files = source.files.iterator();
    while (files.next()) |file_entry| {
        const filename = file_entry.key_ptr.*;
        var header_arena = std.heap.ArenaAllocator.init(allocator);
        defer header_arena.deinit();
        const header_scratch = header_arena.allocator();
        const header = try readShardHeader(header_scratch, model_dir, filename);
        const header_root = std.json.parseFromSliceLeaky(
            std.json.Value,
            header_scratch,
            header.bytes,
            .{},
        ) catch return error.InvalidSafetensorsHeader;
        if (header_root != .object) return error.InvalidSafetensorsHeader;
        try source.stamps.put(filename, try readShardStamp(allocator, header_root.object));

        // Index iteration is deliberate. It leaves all header-name slices in
        // the short-lived arena and copies only shapes for indexed tensors.
        var indexed = source.weight_map.iterator();
        while (indexed.next()) |indexed_entry| {
            if (!std.mem.eql(u8, indexed_entry.value_ptr.*, filename)) continue;
            const tensor_value = header_root.object.get(indexed_entry.key_ptr.*) orelse
                continue;
            if (source.tensors.contains(indexed_entry.key_ptr.*))
                return error.AmbiguousSafetensorsTensor;
            const meta = try parseTensorMeta(
                header_scratch,
                tensor_value,
                filename,
                header.data_base,
            );
            const shape = try allocator.dupe(u64, meta.shape);
            try source.tensors.put(indexed_entry.key_ptr.*, .{
                .dtype = meta.dtype,
                .shape = shape,
                .data_start = meta.data_start,
                .data_end = meta.data_end,
                .data_base = meta.data_base,
                .file = filename,
            });
        }
    }

    var indexed = source.weight_map.iterator();
    while (indexed.next()) |entry| {
        if (!source.tensors.contains(entry.key_ptr.*))
            return error.MissingIndexedSafetensorsTensor;
    }
    return source;
}

fn readShardStamp(allocator: Allocator, header: std.json.ObjectMap) !ShardStamp {
    const meta = header.get("__metadata__") orelse return .{};
    if (meta != .object) return .{};
    var out: ShardStamp = .{};
    inline for (.{ "k", "codebook", "window" }) |field| {
        if (meta.object.get(field)) |v| {
            if (v == .string) @field(out, field) = try allocator.dupe(u8, v.string);
        }
    }
    return out;
}

fn validateShardStamps(source: *const SourceIndex, config: *const model.ModelConfig) !void {
    if (config.expert_layout != .exl3_k4) return;
    var it = source.stamps.valueIterator();
    while (it.next()) |stamp| {
        if (stamp.codebook) |name| {
            const cb = expert_exl3.Codebook.fromName(name) orelse return error.Exl3ShardStampMismatch;
            if (cb != config.expert_quant_codebook) return error.Exl3ShardStampMismatch;
        }
        if (stamp.window) |bits| {
            const parsed = std.fmt.parseInt(i64, bits, 10) catch return error.Exl3ShardStampMismatch;
            const w = expert_exl3.Window.fromBits(parsed) orelse return error.Exl3ShardStampMismatch;
            if (w != config.expert_quant_window) return error.Exl3ShardStampMismatch;
        }
        if (stamp.k) |text| {
            // The stamp spells a rate ("2.5", "4"); the engine keys on the
            // halfwords per tile it implies.
            const k = std.fmt.parseFloat(f64, text) catch return error.Exl3ShardStampMismatch;
            const scaled = @round(k * 16.0);
            if (@abs(k * 16.0 - scaled) > 1e-6) return error.Exl3ShardStampMismatch;
            if (scaled < 0 or scaled > 1024) return error.Exl3ShardStampMismatch;
            if (@as(u32, @intFromFloat(scaled)) != config.expert_quant_rate.n)
                return error.Exl3ShardStampMismatch;
        }
    }
}

fn layerKey(key: []const u8) ?struct { layer: u32, rest: []const u8 } {
    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, key, prefix)) return null;
    const after = key[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, after, '.') orelse return null;
    if (dot == 0) return null;
    const layer = std.fmt.parseInt(u32, after[0..dot], 10) catch return null;
    return .{ .layer = layer, .rest = after[dot + 1 ..] };
}

fn classifyKey(key: []const u8, config: *const model.ModelConfig) !TensorKind {
    if (std.mem.startsWith(u8, key, "visual.") or
        std.mem.startsWith(u8, key, "audio_encoder.") or
        std.mem.startsWith(u8, key, "speech_embeddings.") or
        std.mem.startsWith(u8, key, "model.mtp.") or
        std.mem.startsWith(u8, key, "mtp."))
        return .skipped;
    if (std.mem.startsWith(u8, key, "model.layers.") and
        std.mem.indexOf(u8, key, ".mlp.experts.") != null)
        return .skipped;
    if (std.mem.startsWith(u8, key, "model.layers.") and
        std.mem.indexOf(u8, key, ".mlp.switch_mlp.") != null)
        return if (config.expert_layout == .exl3_k4 or config.expert_layout == .quantized_split)
            .routed_expert
        else
            error.UnclassifiedMimoTensor;

    if (std.mem.eql(u8, key, "model.embed_tokens.weight") or
        std.mem.eql(u8, key, "lm_head.weight") or
        std.mem.eql(u8, key, "model.norm.weight"))
        return .resident;

    const ref = layerKey(key) orelse return error.UnclassifiedMimoTensor;
    if (ref.layer >= config.num_hidden_layers) return error.MimoLayerOutOfRange;
    if (std.mem.eql(u8, ref.rest, "self_attn.qkv_proj.weight") or
        (ref.layer == 0 and
            (std.mem.eql(u8, ref.rest, "mlp.gate_proj.weight") or
                std.mem.eql(u8, ref.rest, "mlp.up_proj.weight") or
                std.mem.eql(u8, ref.rest, "mlp.down_proj.weight"))))
        return .fp8_weight;
    if (std.mem.eql(u8, ref.rest, "self_attn.qkv_proj.weight_scale_inv") or
        (ref.layer == 0 and
            (std.mem.eql(u8, ref.rest, "mlp.gate_proj.weight_scale_inv") or
                std.mem.eql(u8, ref.rest, "mlp.up_proj.weight_scale_inv") or
                std.mem.eql(u8, ref.rest, "mlp.down_proj.weight_scale_inv"))))
        return .fp8_scale;

    if (std.mem.eql(u8, ref.rest, "input_layernorm.weight") or
        std.mem.eql(u8, ref.rest, "post_attention_layernorm.weight") or
        std.mem.eql(u8, ref.rest, "self_attn.o_proj.weight") or
        std.mem.eql(u8, ref.rest, "self_attn.attention_sink_bias") or
        std.mem.eql(u8, ref.rest, "mlp.gate.weight") or
        std.mem.eql(u8, ref.rest, "mlp.gate.e_score_correction_bias"))
        return .resident;

    return error.UnclassifiedMimoTensor;
}

fn qkvGeometry(config: *const model.ModelConfig, layer: u32) QkvGeometry {
    const heads = config.layerNumHeads(layer);
    const kv = config.layerKVHeads(layer);
    const hd = config.layerHeadDim(layer);
    const vd = config.layerVHeadDim(layer);
    return .{
        .q_rows = @as(u64, heads) * hd,
        .k_rows = @as(u64, kv) * hd,
        .v_rows = @as(u64, kv) * vd,
    };
}

fn shapeProduct(shape: []const u64) !u64 {
    var product: u64 = 1;
    for (shape) |dim| {
        product = std.math.mul(u64, product, dim) catch
            return error.SafetensorsShapeOverflow;
    }
    return product;
}

fn payloadBytes(meta: TensorMeta, expected_dtype: ?DType) !u64 {
    if (expected_dtype) |want| if (meta.dtype != want) return error.MimoTensorDtypeMismatch;
    const elem_size: u64 = switch (meta.dtype) {
        .bf16, .f16, .u16 => 2,
        .f32, .u32 => 4,
        .fp8_e4m3 => 1,
        .unknown => return error.UnsupportedMimoTensorDtype,
    };
    const want = std.math.mul(u64, try shapeProduct(meta.shape), elem_size) catch
        return error.SafetensorsShapeOverflow;
    if (meta.data_end - meta.data_start != want) return error.SafetensorsPayloadMismatch;
    return want;
}

fn expectShape(meta: TensorMeta, expected: []const u64) !void {
    if (meta.shape.len != expected.len) return error.MimoTensorShapeMismatch;
    for (meta.shape, expected) |got, want| {
        if (got != want) return error.MimoTensorShapeMismatch;
    }
}

fn affine8Bytes(rows: u64, cols: u64) !u64 {
    if (rows == 0 or cols == 0 or cols % 4 != 0 or cols % AFFINE_GROUP != 0)
        return error.InvalidMimoAffineGeometry;
    const packed_bytes = std.math.mul(u64, rows, cols) catch
        return error.ResidentBytesOverflow;
    const groups = std.math.divExact(u64, cols, AFFINE_GROUP) catch
        return error.InvalidMimoAffineGeometry;
    const side = std.math.mul(
        u64,
        std.math.mul(u64, rows, groups) catch return error.ResidentBytesOverflow,
        4,
    ) catch return error.ResidentBytesOverflow;
    return std.math.add(u64, packed_bytes, side) catch error.ResidentBytesOverflow;
}

fn inferQkvTp(
    geometry: QkvGeometry,
    scale_rows: u64,
) !QkvTp {
    const total = geometry.total();
    var candidate: ?QkvTp = null;
    for ([_]usize{ 8, 4 }) |tp| {
        if (geometry.q_rows % tp != 0 or geometry.k_rows % tp != 0 or
            geometry.v_rows % tp != 0)
            continue;
        const rows_per_rank_u64 = total / tp;
        const blocks_u64 = (rows_per_rank_u64 + FP8_BLOCK - 1) / FP8_BLOCK;
        if (scale_rows != @as(u64, tp) * blocks_u64) continue;
        if (candidate != null) return error.AmbiguousQkvTensorParallelism;
        candidate = .{
            .tp = tp,
            .rows_per_rank = std.math.cast(usize, rows_per_rank_u64) orelse
                return error.InvalidQkvGeometry,
            .blocks_per_rank = std.math.cast(usize, blocks_u64) orelse
                return error.InvalidQkvGeometry,
        };
    }
    return candidate orelse error.InvalidQkvGeometry;
}

fn isQkvWeightKey(key: []const u8) bool {
    return std.mem.endsWith(u8, key, ".self_attn.qkv_proj.weight");
}

fn fp8Base(key: []const u8) []const u8 {
    return key[0 .. key.len - ".weight".len];
}

fn scaleKey(allocator: Allocator, key: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}.weight_scale_inv", .{fp8Base(key)});
}

fn validateFp8Pair(
    source: *const SourceIndex,
    allocator: Allocator,
    config: *const model.ModelConfig,
    key: []const u8,
    meta: TensorMeta,
) !void {
    if (meta.dtype != .fp8_e4m3) return error.MimoTensorDtypeMismatch;
    if (meta.shape.len != 2) return error.MimoTensorShapeMismatch;
    const ref = layerKey(key) orelse return error.UnclassifiedMimoTensor;
    const s_key = try scaleKey(allocator, key);
    const scale_meta = source.tensors.get(s_key) orelse return error.MissingFp8Scale;
    if (scale_meta.dtype != .f32 or scale_meta.shape.len != 2)
        return error.InvalidFp8ScaleShape;

    const rows = meta.shape[0];
    const cols = meta.shape[1];
    if (cols == 0 or cols % FP8_BLOCK != 0 or cols % AFFINE_GROUP != 0)
        return error.InvalidFp8Shape;
    if (isQkvWeightKey(key)) {
        if (cols != config.hidden_size) return error.InvalidFp8Shape;
        const geometry = qkvGeometry(config, ref.layer);
        if (rows != geometry.total()) return error.InvalidQkvGeometry;
        if (scale_meta.shape[1] != cols / FP8_BLOCK)
            return error.InvalidFp8ScaleShape;
        _ = try inferQkvTp(geometry, scale_meta.shape[0]);
    } else {
        var expected_rows = rows;
        var expected_cols = cols;
        if (ref.layer != 0) return error.UnclassifiedMimoTensor;
        if (std.mem.eql(u8, ref.rest, "mlp.gate_proj.weight") or
            std.mem.eql(u8, ref.rest, "mlp.up_proj.weight"))
        {
            expected_rows = config.intermediate_size;
            expected_cols = config.hidden_size;
        } else if (std.mem.eql(u8, ref.rest, "mlp.down_proj.weight")) {
            expected_rows = config.hidden_size;
            expected_cols = config.intermediate_size;
        } else {
            return error.UnclassifiedMimoTensor;
        }
        if (rows != expected_rows or cols != expected_cols)
            return error.InvalidFp8Shape;
        const required_scale_rows = (rows + FP8_BLOCK - 1) / FP8_BLOCK;
        if (scale_meta.shape[0] < required_scale_rows or
            scale_meta.shape[1] != cols / FP8_BLOCK)
            return error.InvalidFp8ScaleShape;
    }
    _ = try payloadBytes(meta, .fp8_e4m3);
    _ = try payloadBytes(scale_meta, .f32);
}

fn denseExpectedShape(
    key: []const u8,
    config: *const model.ModelConfig,
) !struct { shape: [2]u64, len: usize, dtype: DType } {
    if (std.mem.eql(u8, key, "model.embed_tokens.weight") or
        std.mem.eql(u8, key, "lm_head.weight"))
        return .{ .shape = .{ config.vocab_size, config.hidden_size }, .len = 2, .dtype = .bf16 };
    if (std.mem.eql(u8, key, "model.norm.weight"))
        return .{ .shape = .{ config.hidden_size, 0 }, .len = 1, .dtype = .bf16 };

    const ref = layerKey(key) orelse return error.UnclassifiedMimoTensor;
    const layer = ref.layer;
    if (std.mem.eql(u8, ref.rest, "input_layernorm.weight") or
        std.mem.eql(u8, ref.rest, "post_attention_layernorm.weight"))
        return .{ .shape = .{ config.hidden_size, 0 }, .len = 1, .dtype = .bf16 };
    if (std.mem.eql(u8, ref.rest, "self_attn.o_proj.weight")) {
        return .{
            .shape = .{
                config.hidden_size,
                @as(u64, config.layerNumHeads(layer)) * config.layerVHeadDim(layer),
            },
            .len = 2,
            .dtype = .bf16,
        };
    }
    if (std.mem.eql(u8, ref.rest, "self_attn.attention_sink_bias"))
        return .{ .shape = .{ config.layerNumHeads(layer), 0 }, .len = 1, .dtype = .bf16 };
    if (std.mem.eql(u8, ref.rest, "mlp.gate.weight"))
        return .{ .shape = .{ config.num_experts, config.hidden_size }, .len = 2, .dtype = .bf16 };
    if (std.mem.eql(u8, ref.rest, "mlp.gate.e_score_correction_bias"))
        return .{ .shape = .{ config.num_experts, 0 }, .len = 1, .dtype = .f32 };
    return error.UnclassifiedMimoTensor;
}

/// Stacked bank geometry: `[E, in/16, out/16, n]` trellis with `suh`/`svh` the
/// two axis scales. gate and up project hidden->inter, down the other way.
/// The routed bank of whichever layout this pack packs its experts in.
fn validateRoutedExpert(key: []const u8, meta: TensorMeta, config: *const model.ModelConfig) !void {
    return if (config.expert_layout == .quantized_split)
        validateAffineExpert(key, meta, config)
    else
        validateExl3Expert(key, meta, config);
}

/// `[E, out, packed]` U32 beside bf16 `[E, out, groups]` scales and biases, at
/// a (bits, group_size) the shapes themselves solve — this pack's widths are
/// per layer and per projection, so there is no config-wide width to check.
fn validateAffineExpert(key: []const u8, meta: TensorMeta, config: *const model.ModelConfig) !void {
    const parts = try routedBankKey(key, config);
    const experts: u64 = config.num_experts;
    if (meta.shape.len != 3) return error.MimoTensorShapeMismatch;
    if (std.mem.eql(u8, parts.part, "weight")) {
        if (meta.dtype != .u32) return error.MimoTensorDtypeMismatch;
        if (expert_quant.affineBitsFromPacked(meta.shape[2], parts.in_dim) == null)
            return error.MimoTensorShapeMismatch;
    } else if (std.mem.eql(u8, parts.part, "scales") or std.mem.eql(u8, parts.part, "biases")) {
        if (meta.dtype != .bf16) return error.MimoTensorDtypeMismatch;
        if (expert_quant.affineGroupFromScales(meta.shape[2], parts.in_dim) == null)
            return error.MimoTensorShapeMismatch;
    } else return error.UnclassifiedMimoTensor;
    try expectShape(meta, &[_]u64{ experts, parts.out_dim, meta.shape[2] });
    _ = try payloadBytes(meta, meta.dtype);
}

const RoutedBankKey = struct { part: []const u8, in_dim: u64, out_dim: u64 };

fn routedBankKey(key: []const u8, config: *const model.ModelConfig) !RoutedBankKey {
    const ref = layerKey(key) orelse return error.UnclassifiedMimoTensor;
    if (ref.layer >= config.num_hidden_layers or ref.layer < config.first_k_dense_replace)
        return error.MimoLayerOutOfRange;
    const bank_prefix = "mlp.switch_mlp.";
    if (!std.mem.startsWith(u8, ref.rest, bank_prefix)) return error.UnclassifiedMimoTensor;
    const rest = ref.rest[bank_prefix.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.UnclassifiedMimoTensor;
    const proj = rest[0..dot];
    const down = std.mem.eql(u8, proj, "down_proj");
    if (!down and !std.mem.eql(u8, proj, "gate_proj") and !std.mem.eql(u8, proj, "up_proj"))
        return error.UnclassifiedMimoTensor;
    return .{
        .part = rest[dot + 1 ..],
        .in_dim = if (down) config.moe_intermediate_size else config.hidden_size,
        .out_dim = if (down) config.hidden_size else config.moe_intermediate_size,
    };
}

fn validateExl3Expert(key: []const u8, meta: TensorMeta, config: *const model.ModelConfig) !void {
    const ref = layerKey(key) orelse return error.UnclassifiedMimoTensor;
    if (ref.layer >= config.num_hidden_layers or ref.layer < config.first_k_dense_replace)
        return error.MimoLayerOutOfRange;
    const bank_prefix = "mlp.switch_mlp.";
    if (!std.mem.startsWith(u8, ref.rest, bank_prefix)) return error.UnclassifiedMimoTensor;
    const rest = ref.rest[bank_prefix.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.UnclassifiedMimoTensor;
    const proj = rest[0..dot];
    const part = rest[dot + 1 ..];
    const down = std.mem.eql(u8, proj, "down_proj");
    if (!down and !std.mem.eql(u8, proj, "gate_proj") and !std.mem.eql(u8, proj, "up_proj"))
        return error.UnclassifiedMimoTensor;
    const hidden: u64 = config.hidden_size;
    const inter: u64 = config.moe_intermediate_size;
    const in_dim: u64 = if (down) inter else hidden;
    const out_dim: u64 = if (down) hidden else inter;
    const experts: u64 = config.num_experts;
    if (std.mem.eql(u8, part, "trellis")) {
        if (meta.dtype != .u16) return error.MimoTensorDtypeMismatch;
        if (meta.shape.len != 4) return error.MimoTensorShapeMismatch;
        if (in_dim % 16 != 0 or out_dim % 16 != 0) return error.MimoTensorShapeMismatch;
        if (expert_exl3.kFromPackedDim(meta.shape[3]) == null) return error.Exl3TrellisGeometry;
        try expectShape(meta, &[_]u64{ experts, in_dim / 16, out_dim / 16, meta.shape[3] });
    } else if (std.mem.eql(u8, part, "suh")) {
        if (meta.dtype != .f16) return error.MimoTensorDtypeMismatch;
        try expectShape(meta, &[_]u64{ experts, in_dim });
    } else if (std.mem.eql(u8, part, "svh")) {
        if (meta.dtype != .f16) return error.MimoTensorDtypeMismatch;
        try expectShape(meta, &[_]u64{ experts, out_dim });
    } else return error.UnclassifiedMimoTensor;
    _ = try payloadBytes(meta, meta.dtype);
}

fn validateDense(key: []const u8, meta: TensorMeta, config: *const model.ModelConfig) !void {
    const expected = try denseExpectedShape(key, config);
    if (meta.dtype != expected.dtype) return error.MimoTensorDtypeMismatch;
    const expected_slice = expected.shape[0..expected.len];
    try expectShape(meta, expected_slice);
    _ = try payloadBytes(meta, expected.dtype);
}

fn requireKind(
    source: *const SourceIndex,
    allocator: Allocator,
    config: *const model.ModelConfig,
    key: []const u8,
    expected: TensorKind,
) !void {
    const meta = source.tensors.get(key) orelse return error.MissingMimoRequiredTensor;
    if (try classifyKey(key, config) != expected)
        return error.MimoRequiredTensorKindMismatch;
    switch (expected) {
        .resident => try validateDense(key, meta, config),
        .routed_expert => try validateRoutedExpert(key, meta, config),
        .fp8_weight => try validateFp8Pair(source, allocator, config, key, meta),
        .fp8_scale, .skipped => {},
    }
}

fn validateRequired(
    source: *const SourceIndex,
    allocator: Allocator,
    config: *const model.ModelConfig,
) !void {
    try requireKind(source, allocator, config, "model.embed_tokens.weight", .resident);
    try requireKind(source, allocator, config, "lm_head.weight", .resident);
    try requireKind(source, allocator, config, "model.norm.weight", .resident);

    for (0..config.num_hidden_layers) |i| {
        const layer: u32 = @intCast(i);
        const layer_prefix = try std.fmt.allocPrint(allocator, "model.layers.{d}", .{layer});
        try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.input_layernorm.weight", .{layer_prefix}), .resident);
        try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.post_attention_layernorm.weight", .{layer_prefix}), .resident);
        try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.self_attn.qkv_proj.weight", .{layer_prefix}), .fp8_weight);
        try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.self_attn.o_proj.weight", .{layer_prefix}), .resident);
        if (config.layerHasAttnSinks(layer)) {
            try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.self_attn.attention_sink_bias", .{layer_prefix}), .resident);
        }
        if (layer == 0) {
            for ([_][]const u8{ "gate", "up", "down" }) |projection| {
                const k = try std.fmt.allocPrint(
                    allocator,
                    "{s}.mlp.{s}_proj.weight",
                    .{ layer_prefix, projection },
                );
                try requireKind(source, allocator, config, k, .fp8_weight);
            }
        } else {
            try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.mlp.gate.weight", .{layer_prefix}), .resident);
            try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.mlp.gate.e_score_correction_bias", .{layer_prefix}), .resident);
            const bank: ?[3][]const u8 = switch (config.expert_layout) {
                .exl3_k4 => .{ "trellis", "suh", "svh" },
                .quantized_split => .{ "weight", "scales", "biases" },
                else => null,
            };
            if (bank) |parts| {
                for ([_][]const u8{ "gate", "up", "down" }) |projection| {
                    for (parts) |part| {
                        const k = try std.fmt.allocPrint(allocator, "{s}.mlp.switch_mlp.{s}_proj.{s}", .{ layer_prefix, projection, part });
                        try requireKind(source, allocator, config, k, .routed_expert);
                    }
                }
            }
        }
    }
}

fn validatePlan(source: *const SourceIndex, allocator: Allocator, config: *const model.ModelConfig) !void {
    try validateShardStamps(source, config);
    try validateRequired(source, allocator, config);
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const meta = entry.value_ptr.*;
        switch (try classifyKey(key, config)) {
            .skipped => {},
            .resident => try validateDense(key, meta, config),
            .routed_expert => try validateRoutedExpert(key, meta, config),
            .fp8_weight => try validateFp8Pair(source, allocator, config, key, meta),
            .fp8_scale => {
                const suffix = ".weight_scale_inv";
                if (!std.mem.endsWith(u8, key, suffix))
                    return error.UnclassifiedMimoTensor;
                const base = key[0 .. key.len - suffix.len];
                const weight_key = try std.fmt.allocPrint(allocator, "{s}.weight", .{base});
                if (!source.tensors.contains(weight_key))
                    return error.MissingFp8Weight;
            },
        }
    }
}

fn countResidentBytes(
    source: *const SourceIndex,
    allocator: Allocator,
    config: *const model.ModelConfig,
) !u64 {
    var total: u64 = 0;
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const meta = entry.value_ptr.*;
        switch (try classifyKey(key, config)) {
            .skipped, .fp8_scale => {},
            .resident, .routed_expert => {
                total = std.math.add(u64, total, try payloadBytes(meta, null)) catch
                    return error.ResidentBytesOverflow;
            },
            .fp8_weight => {
                const ref = layerKey(key) orelse return error.UnclassifiedMimoTensor;
                const scale_key = try scaleKey(allocator, key);
                const scale_meta = source.tensors.get(scale_key) orelse
                    return error.MissingFp8Scale;
                const bytes = if (isQkvWeightKey(key)) blk: {
                    const geometry = qkvGeometry(config, ref.layer);
                    _ = try inferQkvTp(geometry, scale_meta.shape[0]);
                    break :blk try affine8Bytes(geometry.q_rows, meta.shape[1]) +
                        try affine8Bytes(geometry.k_rows, meta.shape[1]) +
                        try affine8Bytes(geometry.v_rows, meta.shape[1]);
                } else try affine8Bytes(meta.shape[0], meta.shape[1]);
                total = std.math.add(u64, total, bytes) catch
                    return error.ResidentBytesOverflow;
            },
        }
    }
    return total;
}

fn readTensor(allocator: Allocator, model_dir: []const u8, meta: TensorMeta) ![]u8 {
    const len_u64 = meta.data_end - meta.data_start;
    const len = std.math.cast(usize, len_u64) orelse return error.SafetensorsShapeOverflow;
    const out = try allocator.alloc(u8, len);
    errdefer allocator.free(out);
    const path = try shardPath(allocator, model_dir, meta.file);
    defer allocator.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.MissingMimoShard;
    defer _ = std.c.close(fd);
    const absolute = std.math.add(u64, meta.data_base, meta.data_start) catch
        return error.InvalidSafetensorsHeader;
    try preadExact(fd, out, absolute);
    return out;
}

fn shapeForUpload(meta: TensorMeta, shape: *[4]c_int) ![]const c_int {
    if (meta.shape.len == 0 or meta.shape.len > shape.len)
        return error.MimoTensorShapeMismatch;
    for (meta.shape, 0..) |dim, i| {
        shape[i] = std.math.cast(c_int, dim) orelse return error.MimoTensorShapeMismatch;
    }
    return shape[0..meta.shape.len];
}

fn uploadDense(raw: []const u8, meta: TensorMeta, stream: mlx.mlx_stream) !mlx.mlx_array {
    var shape: [4]c_int = undefined;
    const shape_slice = try shapeForUpload(meta, &shape);
    const dtype: mlx.mlx_dtype = switch (meta.dtype) {
        .bf16 => .bfloat16,
        .f16 => .float16,
        .f32 => .float32,
        .u16 => .uint16,
        .u32 => .uint32,
        else => return error.MimoTensorDtypeMismatch,
    };
    _ = stream;
    return mlx.mlx_array_new_data(
        @ptrCast(raw.ptr),
        shape_slice.ptr,
        @intCast(shape_slice.len),
        dtype,
    );
}

fn f32At(bytes: []const u8, index: usize) !f32 {
    const begin = std.math.mul(usize, index, 4) catch return error.SafetensorsShapeOverflow;
    if (begin + 4 > bytes.len) return error.SafetensorsPayloadMismatch;
    const bits = std.mem.readInt(u32, bytes[begin..][0..4], .little);
    const value: f32 = @bitCast(bits);
    if (!std.math.isFinite(value)) return error.InvalidFp8Scale;
    return value;
}

const fp8_values: [256]f32 = blk: {
    @setEvalBranchQuota(10000);
    var values: [256]f32 = undefined;
    for (&values, 0..) |*value, code| value.* = fp8E4M3FnValue(@intCast(code));
    break :blk values;
};

fn decodedFp8Value(code: u8, scale: f32) !f32 {
    const value = fp8_values[code] * scale;
    const bf16_max: f32 = @bitCast(@as(u32, 0x7f7f0000));
    if (!std.math.isFinite(value) or @abs(value) > bf16_max) return error.InvalidFp8Value;
    return value;
}

fn decodeBlocks(
    allocator: Allocator,
    raw: []const u8,
    rows: usize,
    cols: usize,
    scales: []const u8,
    scale_rows: usize,
    scale_cols: usize,
) ![]f32 {
    if (cols == 0 or cols % @as(usize, @intCast(FP8_BLOCK)) != 0 or
        scale_cols != cols / @as(usize, @intCast(FP8_BLOCK)) or
        scale_rows * @as(usize, @intCast(FP8_BLOCK)) < rows)
        return error.InvalidFp8ScaleShape;
    const elements = std.math.mul(usize, rows, cols) catch return error.SafetensorsShapeOverflow;
    if (raw.len != elements or scales.len != scale_rows * scale_cols * 4)
        return error.SafetensorsPayloadMismatch;
    const out = try allocator.alloc(f32, elements);
    errdefer allocator.free(out);
    for (0..rows) |row| {
        for (0..cols) |col| {
            const scale_index = (row / 128) * scale_cols + col / 128;
            const scale = try f32At(scales, scale_index);
            out[row * cols + col] = try decodedFp8Value(raw[row * cols + col], scale);
        }
    }
    return out;
}

fn decodeQkv(
    allocator: Allocator,
    raw: []const u8,
    rows: usize,
    cols: usize,
    scales: []const u8,
    scale_rows: usize,
    scale_cols: usize,
    geometry: QkvGeometry,
) !DecodedQkv {
    if (geometry.total() != rows or cols == 0 or cols % 128 != 0 or
        scale_cols != cols / 128)
        return error.InvalidQkvGeometry;
    const tp_info = try inferQkvTp(geometry, scale_rows);
    const q_rows = std.math.cast(usize, geometry.q_rows) orelse return error.InvalidQkvGeometry;
    const k_rows = std.math.cast(usize, geometry.k_rows) orelse return error.InvalidQkvGeometry;
    const v_rows = std.math.cast(usize, geometry.v_rows) orelse return error.InvalidQkvGeometry;
    const elements = std.math.mul(usize, rows, cols) catch return error.SafetensorsShapeOverflow;
    if (raw.len != elements or scales.len != scale_rows * scale_cols * 4)
        return error.SafetensorsPayloadMismatch;
    const q = try allocator.alloc(f32, q_rows * cols);
    errdefer allocator.free(q);
    const k = try allocator.alloc(f32, k_rows * cols);
    errdefer allocator.free(k);
    const v = try allocator.alloc(f32, v_rows * cols);
    errdefer allocator.free(v);

    const q_per = q_rows / tp_info.tp;
    const k_per = k_rows / tp_info.tp;
    const v_per = v_rows / tp_info.tp;
    for (0..tp_info.tp) |rank| {
        const source_rank_row = rank * tp_info.rows_per_rank;
        const scale_rank_row = rank * tp_info.blocks_per_rank;
        for (0..tp_info.rows_per_rank) |local_row| {
            const source_row = source_rank_row + local_row;
            const scale_row = scale_rank_row + local_row / 128;
            const source = raw[source_row * cols ..][0..cols];
            const destination: []f32 = if (local_row < q_per) blk: {
                break :blk q[(rank * q_per + local_row) * cols ..][0..cols];
            } else if (local_row < q_per + k_per) blk: {
                break :blk k[(rank * k_per + local_row - q_per) * cols ..][0..cols];
            } else blk: {
                break :blk v[(rank * v_per + local_row - q_per - k_per) * cols ..][0..cols];
            };
            for (0..cols) |col| {
                const scale_index = scale_row * scale_cols + col / 128;
                destination[col] = try decodedFp8Value(source[col], try f32At(scales, scale_index));
            }
        }
    }
    return .{ .q = q, .k = k, .v = v, .allocator = allocator };
}

fn quantizeAffine8(
    allocator: Allocator,
    values: []const f32,
    rows: usize,
    cols: usize,
    stream: mlx.mlx_stream,
) !QuantTriple {
    if (values.len != rows * cols) return error.InvalidMimoAffineGeometry;
    var shape = [_]c_int{
        std.math.cast(c_int, rows) orelse return error.InvalidMimoAffineGeometry,
        std.math.cast(c_int, cols) orelse return error.InvalidMimoAffineGeometry,
    };
    const dense = mlx.mlx_array_new_data(@ptrCast(values.ptr), &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(dense);
    var bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bf16);
    try mlx.check(mlx.mlx_astype(&bf16, dense, .bfloat16, stream));

    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_quantize(
        &outputs,
        bf16,
        mlx.mlx_optional_int.some(64),
        mlx.mlx_optional_int.some(8),
        "affine",
        .{},
        stream,
    ));
    if (mlx.mlx_vector_array_size(outputs) != 3)
        return error.UnexpectedQuantizeOutput;
    var triple = QuantTriple{
        .weight = mlx.mlx_array_new(),
        .scales = mlx.mlx_array_new(),
        .biases = mlx.mlx_array_new(),
    };
    errdefer triple.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&triple.weight, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&triple.scales, outputs, 1));
    try mlx.check(mlx.mlx_vector_array_get(&triple.biases, outputs, 2));
    try mlx.check(mlx.mlx_eval(outputs));
    _ = allocator;
    return triple;
}

fn putWeight(weights: *model.Weights, allocator: Allocator, key: []const u8, arr: mlx.mlx_array) !void {
    if (weights.map.contains(key)) return error.DuplicateOutputWeight;
    const owned_key = try allocator.dupe(u8, key);
    errdefer allocator.free(owned_key);
    try weights.map.put(owned_key, arr);
}

fn putTriple(
    weights: *model.Weights,
    allocator: Allocator,
    base: []const u8,
    triple: *QuantTriple,
) !void {
    const names = [_][]const u8{ ".weight", ".scales", ".biases" };
    const arrays = [_]*mlx.mlx_array{ &triple.weight, &triple.scales, &triple.biases };
    var inserted = [_]bool{ false, false, false };
    errdefer {
        for (inserted, arrays) |done, arr| {
            if (!done) {
                freeArray(arr);
            }
        }
    }
    for (names, arrays, 0..) |suffix, arr, i| {
        const key = try std.fmt.allocPrint(allocator, "{s}{s}", .{ base, suffix });
        defer allocator.free(key);
        try putWeight(weights, allocator, key, arr.*);
        arr.* = .{};
        inserted[i] = true;
    }
}

fn outputQkvBase(allocator: Allocator, source_key: []const u8, projection: []const u8) ![]u8 {
    const source_base = fp8Base(source_key);
    const suffix = ".self_attn.qkv_proj";
    if (!std.mem.endsWith(u8, source_base, suffix)) return error.InvalidQkvGeometry;
    return std.fmt.allocPrint(
        allocator,
        "{s}.{s}_proj",
        .{ source_base[0 .. source_base.len - ".qkv_proj".len], projection },
    );
}

fn loadFp8Weight(
    weights: *model.Weights,
    allocator: Allocator,
    model_dir: []const u8,
    config: *const model.ModelConfig,
    key: []const u8,
    meta: TensorMeta,
    source: *const SourceIndex,
    stream: mlx.mlx_stream,
) !void {
    const scale_name = try scaleKey(allocator, key);
    defer allocator.free(scale_name);
    const scale_meta = source.tensors.get(scale_name) orelse return error.MissingFp8Scale;
    const raw = try readTensor(allocator, model_dir, meta);
    defer allocator.free(raw);
    const scale_raw = try readTensor(allocator, model_dir, scale_meta);
    defer allocator.free(scale_raw);

    const ref = layerKey(key) orelse return error.UnclassifiedMimoTensor;
    if (isQkvWeightKey(key)) {
        const geometry = qkvGeometry(config, ref.layer);
        const rows = std.math.cast(usize, meta.shape[0]) orelse return error.InvalidQkvGeometry;
        const cols = std.math.cast(usize, meta.shape[1]) orelse return error.InvalidQkvGeometry;
        const scale_rows = std.math.cast(usize, scale_meta.shape[0]) orelse return error.InvalidQkvGeometry;
        const scale_cols = std.math.cast(usize, scale_meta.shape[1]) orelse return error.InvalidQkvGeometry;
        var decoded = try decodeQkv(
            allocator,
            raw,
            rows,
            cols,
            scale_raw,
            scale_rows,
            scale_cols,
            geometry,
        );
        defer decoded.deinit();
        for ([_][]const u8{ "q", "k", "v" }, [_][]const f32{ decoded.q, decoded.k, decoded.v }) |projection, values| {
            const base = try outputQkvBase(allocator, key, projection);
            defer allocator.free(base);
            const part_rows: usize = switch (projection[0]) {
                'q' => std.math.cast(usize, geometry.q_rows) orelse return error.InvalidQkvGeometry,
                'k' => std.math.cast(usize, geometry.k_rows) orelse return error.InvalidQkvGeometry,
                else => std.math.cast(usize, geometry.v_rows) orelse return error.InvalidQkvGeometry,
            };
            var triple = try quantizeAffine8(allocator, values, part_rows, cols, stream);
            errdefer triple.deinit();
            try putTriple(weights, allocator, base, &triple);
        }
    } else {
        const rows = std.math.cast(usize, meta.shape[0]) orelse return error.InvalidFp8Shape;
        const cols = std.math.cast(usize, meta.shape[1]) orelse return error.InvalidFp8Shape;
        const scale_rows = std.math.cast(usize, scale_meta.shape[0]) orelse return error.InvalidFp8Shape;
        const scale_cols = std.math.cast(usize, scale_meta.shape[1]) orelse return error.InvalidFp8Shape;
        const values = try decodeBlocks(allocator, raw, rows, cols, scale_raw, scale_rows, scale_cols);
        defer allocator.free(values);
        var triple = try quantizeAffine8(allocator, values, rows, cols, stream);
        errdefer triple.deinit();
        try putTriple(weights, allocator, fp8Base(key), &triple);
    }
}

const TestTensor = struct {
    key: []const u8,
    dtype: []const u8,
    shape: []const u64,
    bytes: []const u8,
};

const TestIndexEntry = struct {
    key: []const u8,
    file: []const u8,
};

fn appendTestFormat(
    allocator: Allocator,
    list: *std.ArrayList(u8),
    comptime format: []const u8,
    args: anytype,
) !void {
    const text = try std.fmt.allocPrint(allocator, format, args);
    defer allocator.free(text);
    try list.appendSlice(allocator, text);
}

fn writeTestShard(
    io: std.Io,
    allocator: Allocator,
    dir: std.Io.Dir,
    filename: []const u8,
    tensors: []const TestTensor,
) !void {
    return writeTestShardStamped(io, allocator, dir, filename, tensors, null);
}

fn writeTestShardStamped(
    io: std.Io,
    allocator: Allocator,
    dir: std.Io.Dir,
    filename: []const u8,
    tensors: []const TestTensor,
    stamp: ?[]const u8,
) !void {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    try header.append(allocator, '{');
    if (stamp) |body| try appendTestFormat(allocator, &header, "\"__metadata__\":{{{s}}},", .{body});
    var data_size: usize = 0;
    for (tensors, 0..) |tensor, i| {
        if (i != 0) try header.append(allocator, ',');
        try appendTestFormat(
            allocator,
            &header,
            "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[",
            .{ tensor.key, tensor.dtype },
        );
        for (tensor.shape, 0..) |dim, j| {
            if (j != 0) try header.append(allocator, ',');
            try appendTestFormat(allocator, &header, "{d}", .{dim});
        }
        const end = std.math.add(usize, data_size, tensor.bytes.len) catch
            return error.TestFixtureTooLarge;
        try appendTestFormat(
            allocator,
            &header,
            "],\"data_offsets\":[{d},{d}]}}",
            .{ data_size, end },
        );
        data_size = end;
    }
    try header.append(allocator, '}');

    const total = std.math.add(usize, 8 + header.items.len, data_size) catch
        return error.TestFixtureTooLarge;
    const file_bytes = try allocator.alloc(u8, total);
    defer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.items.len, .little);
    @memcpy(file_bytes[8 .. 8 + header.items.len], header.items);
    var at = 8 + header.items.len;
    for (tensors) |tensor| {
        @memcpy(file_bytes[at..][0..tensor.bytes.len], tensor.bytes);
        at += tensor.bytes.len;
    }
    try dir.writeFile(io, .{ .sub_path = filename, .data = file_bytes });
}

fn writeTestIndex(
    io: std.Io,
    allocator: Allocator,
    dir: std.Io.Dir,
    entries: []const TestIndexEntry,
) !void {
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(allocator);
    try json.appendSlice(allocator, "{\"weight_map\":{");
    for (entries, 0..) |entry, i| {
        if (i != 0) try json.append(allocator, ',');
        try appendTestFormat(allocator, &json, "\"{s}\":\"{s}\"", .{ entry.key, entry.file });
    }
    try json.appendSlice(allocator, "}}");
    try dir.writeFile(io, .{
        .sub_path = "model.safetensors.index.json",
        .data = json.items,
    });
}

fn testBf16Bytes(allocator: Allocator, count: usize, bits: u16) ![]u8 {
    const bytes = try allocator.alloc(u8, count * 2);
    for (0..count) |i| std.mem.writeInt(u16, bytes[i * 2 ..][0..2], bits, .little);
    return bytes;
}

fn testFp8Bytes(allocator: Allocator, count: usize, code: u8) ![]u8 {
    const bytes = try allocator.alloc(u8, count);
    @memset(bytes, code);
    return bytes;
}

fn testF32Bytes(allocator: Allocator, values: []const f32) ![]u8 {
    const bytes = try allocator.alloc(u8, values.len * 4);
    for (values, 0..) |value, i| {
        std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
    }
    return bytes;
}

const TinySourceFixture = struct {
    allocator: Allocator,
    path: []u8,
    config: model.ModelConfig,
    qkv_weight: []u8,
    qkv_scales: []u8,
    mlp_weight: []u8,
    mlp_scales: []u8,
    embed: []u8,

    fn deinit(self: *TinySourceFixture) void {
        self.allocator.free(self.path);
        self.allocator.free(self.qkv_weight);
        self.allocator.free(self.qkv_scales);
        self.allocator.free(self.mlp_weight);
        self.allocator.free(self.mlp_scales);
        self.allocator.free(self.embed);
    }
};

fn makeTinySourceFixture(
    io: std.Io,
    allocator: Allocator,
    tmp: *std.testing.TmpDir,
) !TinySourceFixture {
    try tmp.dir.createDirPath(io, "mimo-source");
    var dir = try tmp.dir.openDir(io, "mimo-source", .{});
    defer dir.close(io);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try dir.realPath(io, &path_buf);
    const path = try allocator.dupe(u8, path_buf[0..path_len]);
    errdefer allocator.free(path);

    try dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"mimo_v2","vocab_size":2,"hidden_size":128,
        \\ "num_hidden_layers":1,"intermediate_size":128,
        \\ "moe_intermediate_size":128,"n_routed_experts":1,
        \\ "num_experts_per_tok":1,"n_group":1,"topk_group":1,
        \\ "num_attention_heads":8,"num_key_value_heads":8,"head_dim":16,
        \\ "v_head_dim":16,"swa_num_attention_heads":8,
        \\ "swa_num_key_value_heads":8,"swa_head_dim":16,"swa_v_head_dim":16,
        \\ "hybrid_layer_pattern":[1],"moe_layer_freq":[0],
        \\ "attention_projection_layout":"fused_qkv",
        \\ "add_swa_attention_sink_bias":false,
        \\ "add_full_attention_sink_bias":false}
    });

    const embed_shape = [_]u64{ 2, 128 };
    const vector_shape = [_]u64{128};
    const matrix_shape = [_]u64{ 128, 128 };
    const qkv_shape = [_]u64{ 384, 128 };
    const qkv_scale_shape = [_]u64{ 4, 1 };
    const mlp_scale_shape = [_]u64{ 1, 1 };
    const embed = try testBf16Bytes(allocator, 2 * 128, 0x3f80);
    errdefer allocator.free(embed);
    const qkv_weight = try testFp8Bytes(allocator, 384 * 128, 0x38);
    errdefer allocator.free(qkv_weight);
    const qkv_scales = try testF32Bytes(allocator, &.{ 1.0, 2.0, 3.0, 4.0 });
    errdefer allocator.free(qkv_scales);
    const mlp_weight = try testFp8Bytes(allocator, 128 * 128, 0x38);
    errdefer allocator.free(mlp_weight);
    const mlp_scales = try testF32Bytes(allocator, &.{1.0});
    errdefer allocator.free(mlp_scales);
    const norm = try testBf16Bytes(allocator, 128, 0x3f80);
    defer allocator.free(norm);
    const o_proj = try testBf16Bytes(allocator, 128 * 128, 0x3f80);
    defer allocator.free(o_proj);
    const ignored_expert = [_]u8{0};
    const ignored_mtp = [_]u8{ 0, 0 };
    const ignored_media = [_]u8{ 0, 0, 0, 0 };

    const shard_a = [_]TestTensor{
        .{ .key = "model.embed_tokens.weight", .dtype = "BF16", .shape = &embed_shape, .bytes = embed },
        .{ .key = "model.layers.0.self_attn.qkv_proj.weight", .dtype = "F8_E4M3", .shape = &qkv_shape, .bytes = qkv_weight },
    };
    const shard_b = [_]TestTensor{
        .{ .key = "lm_head.weight", .dtype = "BF16", .shape = &embed_shape, .bytes = embed },
        .{ .key = "model.norm.weight", .dtype = "BF16", .shape = &vector_shape, .bytes = norm },
        .{ .key = "model.layers.0.input_layernorm.weight", .dtype = "BF16", .shape = &vector_shape, .bytes = norm },
        .{ .key = "model.layers.0.post_attention_layernorm.weight", .dtype = "BF16", .shape = &vector_shape, .bytes = norm },
        .{ .key = "model.layers.0.self_attn.o_proj.weight", .dtype = "BF16", .shape = &matrix_shape, .bytes = o_proj },
        .{ .key = "model.layers.0.mlp.gate_proj.weight", .dtype = "F8_E4M3", .shape = &matrix_shape, .bytes = mlp_weight },
        .{ .key = "model.layers.0.mlp.gate_proj.weight_scale_inv", .dtype = "F32", .shape = &mlp_scale_shape, .bytes = mlp_scales },
        .{ .key = "model.layers.0.mlp.up_proj.weight", .dtype = "F8_E4M3", .shape = &matrix_shape, .bytes = mlp_weight },
        .{ .key = "model.layers.0.mlp.up_proj.weight_scale_inv", .dtype = "F32", .shape = &mlp_scale_shape, .bytes = mlp_scales },
        .{ .key = "model.layers.0.mlp.down_proj.weight", .dtype = "F8_E4M3", .shape = &matrix_shape, .bytes = mlp_weight },
        .{ .key = "model.layers.0.mlp.down_proj.weight_scale_inv", .dtype = "F32", .shape = &mlp_scale_shape, .bytes = mlp_scales },
        .{ .key = "model.layers.0.mlp.experts.0.gate_proj.weight", .dtype = "U8", .shape = &[_]u64{1}, .bytes = &ignored_expert },
        .{ .key = "model.mtp.layers.0.fake.weight", .dtype = "BF16", .shape = &[_]u64{1}, .bytes = &ignored_mtp },
        .{ .key = "visual.fake", .dtype = "F32", .shape = &[_]u64{1}, .bytes = &ignored_media },
    };
    const shard_c = [_]TestTensor{
        .{ .key = "model.layers.0.self_attn.qkv_proj.weight_scale_inv", .dtype = "F32", .shape = &qkv_scale_shape, .bytes = qkv_scales },
    };
    try writeTestShard(io, allocator, dir, "model-00001.safetensors", &shard_a);
    try writeTestShard(io, allocator, dir, "model-00002.safetensors", &shard_b);
    try writeTestShard(io, allocator, dir, "model-00003.safetensors", &shard_c);

    const entries = [_]TestIndexEntry{
        .{ .key = "model.embed_tokens.weight", .file = "model-00001.safetensors" },
        .{ .key = "model.layers.0.self_attn.qkv_proj.weight", .file = "model-00001.safetensors" },
        .{ .key = "lm_head.weight", .file = "model-00002.safetensors" },
        .{ .key = "model.norm.weight", .file = "model-00002.safetensors" },
        .{ .key = "model.layers.0.input_layernorm.weight", .file = "model-00002.safetensors" },
        .{ .key = "model.layers.0.post_attention_layernorm.weight", .file = "model-00002.safetensors" },
        .{ .key = "model.layers.0.self_attn.o_proj.weight", .file = "model-00002.safetensors" },
        .{ .key = "model.layers.0.mlp.gate_proj.weight", .file = "model-00002.safetensors" },
        .{ .key = "model.layers.0.mlp.gate_proj.weight_scale_inv", .file = "model-00002.safetensors" },
        .{ .key = "model.layers.0.mlp.up_proj.weight", .file = "model-00002.safetensors" },
        .{ .key = "model.layers.0.mlp.up_proj.weight_scale_inv", .file = "model-00002.safetensors" },
        .{ .key = "model.layers.0.mlp.down_proj.weight", .file = "model-00002.safetensors" },
        .{ .key = "model.layers.0.mlp.down_proj.weight_scale_inv", .file = "model-00002.safetensors" },
        .{ .key = "model.layers.0.mlp.experts.0.gate_proj.weight", .file = "model-00002.safetensors" },
        .{ .key = "model.mtp.layers.0.fake.weight", .file = "model-00002.safetensors" },
        .{ .key = "visual.fake", .file = "model-00002.safetensors" },
        .{ .key = "model.layers.0.self_attn.qkv_proj.weight_scale_inv", .file = "model-00003.safetensors" },
    };
    try writeTestIndex(io, allocator, dir, &entries);

    return .{
        .allocator = allocator,
        .path = path,
        .config = .{
            .model_type = "mimo_v2",
            .vocab_size = 2,
            .hidden_size = 128,
            .intermediate_size = 128,
            .num_hidden_layers = 1,
            .num_attention_heads = 8,
            .num_key_value_heads = 8,
            .head_dim = 16,
            .v_head_dim = 16,
            .num_experts = 1,
            .has_sliding_window = false,
        },
        .qkv_weight = qkv_weight,
        .qkv_scales = qkv_scales,
        .mlp_weight = mlp_weight,
        .mlp_scales = mlp_scales,
        .embed = embed,
    };
}

/// A two-layer MiMo checkpoint whose layer 1 carries stacked EXL3 banks at the
/// given packed halfword count. `n = 0` writes no banks at all.
/// The routed bank a synthetic MiMo source carries: none, an EXL3 trellis at
/// `n` packed halfwords, or an affine bank at (bits, group_size).
const TinyAffine = struct { bits: u64, group_size: u64 };

/// `stamp` is the shard's `__metadata__` body, as the converter writes it.
const TinyExl3 = struct { n: u64, stamp: ?[]const u8 = null };

const TinyBank = union(enum) {
    none,
    exl3: TinyExl3,
    affine: TinyAffine,
};

fn writeTinyExl3Source(io: std.Io, allocator: Allocator, dir: std.Io.Dir, n: u64) !void {
    return writeTinySource(io, allocator, dir, if (n == 0) .none else .{ .exl3 = .{ .n = n } });
}

fn writeTinySource(io: std.Io, allocator: Allocator, dir: std.Io.Dir, bank: TinyBank) !void {
    const dim: u64 = 128;
    const tiles = dim / 16;
    const n: u64 = switch (bank) {
        .exl3 => |v| v.n,
        else => 0,
    };
    const stamp: ?[]const u8 = switch (bank) {
        .exl3 => |v| v.stamp,
        else => null,
    };
    try dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"mimo_v2","vocab_size":2,"hidden_size":128,
        \\ "num_hidden_layers":2,"intermediate_size":128,
        \\ "moe_intermediate_size":128,"n_routed_experts":2,
        \\ "num_experts_per_tok":1,"n_group":1,"topk_group":1,
        \\ "num_attention_heads":8,"num_key_value_heads":8,"head_dim":16,
        \\ "v_head_dim":16,"swa_num_attention_heads":8,
        \\ "swa_num_key_value_heads":8,"swa_head_dim":16,"swa_v_head_dim":16,
        \\ "hybrid_layer_pattern":[1,1],"moe_layer_freq":[0,1],
        \\ "attention_projection_layout":"fused_qkv",
        \\ "add_swa_attention_sink_bias":false,
        \\ "add_full_attention_sink_bias":false,
        \\ "expert_quant":{"format":"exl3","k":2.5,"codebook":"tiny"}}
    });
    const affine: TinyAffine = switch (bank) {
        .affine => |a| a,
        else => .{ .bits = 0, .group_size = 0 },
    };
    const packed_cols: u64 = if (affine.bits == 0) 0 else dim * affine.bits / 32;
    const group_cols: u64 = if (affine.group_size == 0) 0 else dim / affine.group_size;
    const affine_w = try allocator.alloc(u8, @intCast(2 * dim * @max(packed_cols, 1) * 4));
    defer allocator.free(affine_w);
    @memset(affine_w, 0x11);
    const affine_s = try testBf16Bytes(allocator, @intCast(2 * dim * @max(group_cols, 1)), 0x3c00);
    defer allocator.free(affine_s);
    const embed = try testBf16Bytes(allocator, 2 * 128, 0x3f80);
    defer allocator.free(embed);
    const norm = try testBf16Bytes(allocator, 128, 0x3f80);
    defer allocator.free(norm);
    const matrix = try testBf16Bytes(allocator, 128 * 128, 0x3f80);
    defer allocator.free(matrix);
    const router = try testBf16Bytes(allocator, 2 * 128, 0x3f80);
    defer allocator.free(router);
    const corr = try testF32Bytes(allocator, &.{ 0.0, 0.0 });
    defer allocator.free(corr);
    const qkv_weight = try testFp8Bytes(allocator, 384 * 128, 0x38);
    defer allocator.free(qkv_weight);
    const qkv_scales = try testF32Bytes(allocator, &.{ 1.0, 2.0, 3.0, 4.0 });
    defer allocator.free(qkv_scales);
    const mlp_weight = try testFp8Bytes(allocator, 128 * 128, 0x38);
    defer allocator.free(mlp_weight);
    const mlp_scales = try testF32Bytes(allocator, &.{1.0});
    defer allocator.free(mlp_scales);
    const trellis = try allocator.alloc(u8, @intCast(2 * tiles * tiles * @max(n, 1) * 2));
    defer allocator.free(trellis);
    @memset(trellis, 0x5a);
    const axis = try testBf16Bytes(allocator, 2 * 128, 0x3c00);
    defer allocator.free(axis);

    var tensors: std.ArrayList(TestTensor) = .empty;
    defer tensors.deinit(allocator);
    var entries: std.ArrayList(TestIndexEntry) = .empty;
    defer entries.deinit(allocator);
    const embed_shape = [_]u64{ 2, dim };
    const vector_shape = [_]u64{dim};
    const matrix_shape = [_]u64{ dim, dim };
    const qkv_shape = [_]u64{ 384, dim };
    const qkv_scale_shape = [_]u64{ 4, 1 };
    const mlp_scale_shape = [_]u64{ 1, 1 };
    const corr_shape = [_]u64{2};
    const trellis_shape = [_]u64{ 2, tiles, tiles, n };
    const axis_shape = [_]u64{ 2, dim };
    try tensors.append(allocator, .{ .key = "model.embed_tokens.weight", .dtype = "BF16", .shape = &embed_shape, .bytes = embed });
    try tensors.append(allocator, .{ .key = "lm_head.weight", .dtype = "BF16", .shape = &embed_shape, .bytes = embed });
    try tensors.append(allocator, .{ .key = "model.norm.weight", .dtype = "BF16", .shape = &vector_shape, .bytes = norm });
    for (0..2) |layer| {
        inline for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight" }) |leaf| {
            try tensors.append(allocator, .{
                .key = try std.fmt.allocPrint(allocator, "model.layers.{d}.{s}", .{ layer, leaf }),
                .dtype = "BF16",
                .shape = &vector_shape,
                .bytes = norm,
            });
        }
        try tensors.append(allocator, .{
            .key = try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn.o_proj.weight", .{layer}),
            .dtype = "BF16",
            .shape = &matrix_shape,
            .bytes = matrix,
        });
        try tensors.append(allocator, .{
            .key = try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn.qkv_proj.weight", .{layer}),
            .dtype = "F8_E4M3",
            .shape = &qkv_shape,
            .bytes = qkv_weight,
        });
        try tensors.append(allocator, .{
            .key = try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn.qkv_proj.weight_scale_inv", .{layer}),
            .dtype = "F32",
            .shape = &qkv_scale_shape,
            .bytes = qkv_scales,
        });
        if (layer == 0) {
            inline for ([_][]const u8{ "gate", "up", "down" }) |proj| {
                try tensors.append(allocator, .{
                    .key = try std.fmt.allocPrint(allocator, "model.layers.0.mlp.{s}_proj.weight", .{proj}),
                    .dtype = "F8_E4M3",
                    .shape = &matrix_shape,
                    .bytes = mlp_weight,
                });
                try tensors.append(allocator, .{
                    .key = try std.fmt.allocPrint(allocator, "model.layers.0.mlp.{s}_proj.weight_scale_inv", .{proj}),
                    .dtype = "F32",
                    .shape = &mlp_scale_shape,
                    .bytes = mlp_scales,
                });
            }
        } else {
            try tensors.append(allocator, .{ .key = "model.layers.1.mlp.gate.weight", .dtype = "BF16", .shape = &embed_shape, .bytes = router });
            try tensors.append(allocator, .{ .key = "model.layers.1.mlp.gate.e_score_correction_bias", .dtype = "F32", .shape = &corr_shape, .bytes = corr });
            if (affine.bits > 0) {
                const w_shape = [_]u64{ 2, dim, packed_cols };
                const s_shape = [_]u64{ 2, dim, group_cols };
                inline for ([_][]const u8{ "gate", "up", "down" }) |proj| {
                    try tensors.append(allocator, .{
                        .key = try std.fmt.allocPrint(allocator, "model.layers.1.mlp.switch_mlp.{s}_proj.weight", .{proj}),
                        .dtype = "U32",
                        .shape = &w_shape,
                        .bytes = affine_w[0..@intCast(2 * dim * packed_cols * 4)],
                    });
                    inline for ([_][]const u8{ "scales", "biases" }) |part| {
                        try tensors.append(allocator, .{
                            .key = try std.fmt.allocPrint(allocator, "model.layers.1.mlp.switch_mlp.{s}_proj.{s}", .{ proj, part }),
                            .dtype = "BF16",
                            .shape = &s_shape,
                            .bytes = affine_s[0..@intCast(2 * dim * group_cols * 2)],
                        });
                    }
                }
            }
            if (n > 0) {
                inline for ([_][]const u8{ "gate", "up", "down" }) |proj| {
                    try tensors.append(allocator, .{
                        .key = try std.fmt.allocPrint(allocator, "model.layers.1.mlp.switch_mlp.{s}_proj.trellis", .{proj}),
                        .dtype = "U16",
                        .shape = &trellis_shape,
                        .bytes = trellis[0..@intCast(2 * tiles * tiles * n * 2)],
                    });
                    try tensors.append(allocator, .{
                        .key = try std.fmt.allocPrint(allocator, "model.layers.1.mlp.switch_mlp.{s}_proj.suh", .{proj}),
                        .dtype = "F16",
                        .shape = &axis_shape,
                        .bytes = axis,
                    });
                    try tensors.append(allocator, .{
                        .key = try std.fmt.allocPrint(allocator, "model.layers.1.mlp.switch_mlp.{s}_proj.svh", .{proj}),
                        .dtype = "F16",
                        .shape = &axis_shape,
                        .bytes = axis,
                    });
                }
            }
        }
    }
    for (tensors.items) |tensor| try entries.append(allocator, .{ .key = tensor.key, .file = "model-00001.safetensors" });
    try writeTestShardStamped(io, allocator, dir, "model-00001.safetensors", tensors.items, stamp);
    try writeTestIndex(io, allocator, dir, entries.items);
}

test "mimo source loads and bills EXL3 routed banks beside the prepared trunk" {
    const t = std.testing;
    const io = t.io;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "mimo-exl3");
    var dir = try tmp.dir.openDir(io, "mimo-exl3", .{});
    defer dir.close(io);
    try writeTinyExl3Source(io, alloc, dir, 40);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try dir.realPath(io, &path_buf);
    const path = path_buf[0..path_len];

    var config = try model.parseConfig(io, t.allocator, path);
    defer config.deinit(t.allocator);
    try t.expectEqual(expert_quant.Layout.exl3_k4, config.expert_layout);
    try t.expectEqual(@as(u32, 40), config.expert_quant_rate.n);
    try t.expect(!config.expertStreamingRequired());
    try t.expect(config.usesMimoSourceTrunk());

    // Nine banks: three projections of [2,8,8,40] u16 plus two [2,128] f16 axes.
    const bank_bytes: u64 = 3 * (2 * 8 * 8 * 40 * 2 + 2 * 2 * 128 * 2);
    const with_banks = try residentBytesWithConfig(io, t.allocator, path, &config);
    try tmp.dir.createDirPath(io, "mimo-trunk-only");
    var bare = try tmp.dir.openDir(io, "mimo-trunk-only", .{});
    defer bare.close(io);
    try writeTinyExl3Source(io, alloc, bare, 0);
    var bare_buf: [std.fs.max_path_bytes]u8 = undefined;
    const bare_len = try bare.realPath(io, &bare_buf);
    var bare_config = try model.parseConfig(io, t.allocator, bare_buf[0..bare_len]);
    defer bare_config.deinit(t.allocator);
    const without = try residentBytesWithConfig(io, t.allocator, bare_buf[0..bare_len], &bare_config);
    try t.expectEqual(bank_bytes, with_banks - without);

    var weights = try loadWeights(io, t.allocator, path, &config);
    defer weights.deinit();
    const trellis = weights.get("model.layers.1.mlp.switch_mlp.gate_proj.trellis") orelse
        return error.MissingWeight;
    try t.expectEqualSlices(c_int, &[_]c_int{ 2, 8, 8, 40 }, mlx.getShape(trellis));
    try t.expectEqual(mlx.mlx_dtype.uint16, mlx.mlx_array_dtype(trellis));
    const suh = weights.get("model.layers.1.mlp.switch_mlp.down_proj.suh") orelse
        return error.MissingWeight;
    try t.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(suh));
}

test "mimo source loads and bills affine routed banks beside the prepared trunk" {
    const t = std.testing;
    const io = t.io;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "mimo-affine");
    var dir = try tmp.dir.openDir(io, "mimo-affine", .{});
    defer dir.close(io);
    try writeTinySource(io, alloc, dir, .{ .affine = .{ .bits = 2, .group_size = 64 } });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try dir.realPath(io, &path_buf);
    const path = path_buf[0..path_len];

    var config = try model.parseConfig(io, t.allocator, path);
    defer config.deinit(t.allocator);
    try t.expectEqual(expert_quant.Layout.quantized_split, config.expert_layout);
    try t.expect(!config.expertStreamingRequired());
    try t.expect(config.usesMimoSourceTrunk());
    // The source trunk hands back split q/k/v, so the fused layout the config
    // names must not survive the layout resolve.
    try t.expect(!config.attn_fused_qkv);

    // Nine banks: three projections of [2,128,8] u32 plus two [2,128,2] bf16.
    const bank_bytes: u64 = 3 * (2 * 128 * 8 * 4 + 2 * (2 * 128 * 2 * 2));
    const with_banks = try residentBytesWithConfig(io, t.allocator, path, &config);
    try tmp.dir.createDirPath(io, "mimo-affine-trunk-only");
    var bare = try tmp.dir.openDir(io, "mimo-affine-trunk-only", .{});
    defer bare.close(io);
    try writeTinySource(io, alloc, bare, .none);
    var bare_buf: [std.fs.max_path_bytes]u8 = undefined;
    const bare_len = try bare.realPath(io, &bare_buf);
    var bare_config = try model.parseConfig(io, t.allocator, bare_buf[0..bare_len]);
    defer bare_config.deinit(t.allocator);
    const without = try residentBytesWithConfig(io, t.allocator, bare_buf[0..bare_len], &bare_config);
    try t.expectEqual(bank_bytes, with_banks - without);

    var weights = try loadWeights(io, t.allocator, path, &config);
    defer weights.deinit();
    const w = weights.get("model.layers.1.mlp.switch_mlp.gate_proj.weight") orelse return error.MissingWeight;
    try t.expectEqualSlices(c_int, &[_]c_int{ 2, 128, 8 }, mlx.getShape(w));
    try t.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(w));
    const sc = weights.get("model.layers.1.mlp.switch_mlp.down_proj.scales") orelse return error.MissingWeight;
    try t.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(sc));
    const bi = weights.get("model.layers.1.mlp.switch_mlp.down_proj.biases") orelse return error.MissingWeight;
    try t.expectEqualSlices(c_int, &[_]c_int{ 2, 128, 2 }, mlx.getShape(bi));
}

test "mimo source refuses an affine bank whose packed width does not solve" {
    const t = std.testing;
    const io = t.io;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "mimo-affine-bad");
    var dir = try tmp.dir.openDir(io, "mimo-affine-bad", .{});
    defer dir.close(io);
    // 7 bits per weight is not a width this build reads.
    try writeTinySource(io, alloc, dir, .{ .affine = .{ .bits = 7, .group_size = 64 } });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try dir.realPath(io, &path_buf);
    const path = path_buf[0..path_len];
    var config = try model.parseConfig(io, t.allocator, path);
    defer config.deinit(t.allocator);
    config.expert_layout = .quantized_split;
    try t.expectError(error.MimoTensorShapeMismatch, residentBytesWithConfig(io, t.allocator, path, &config));
}

test "mimo source refuses an EXL3 trellis the kernels cannot decode" {
    const t = std.testing;
    const io = t.io;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "mimo-exl3-bad");
    var dir = try tmp.dir.openDir(io, "mimo-exl3-bad", .{});
    defer dir.close(io);
    try writeTinyExl3Source(io, alloc, dir, 41);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try dir.realPath(io, &path_buf);
    const path = path_buf[0..path_len];
    var config = try model.parseConfig(io, t.allocator, path);
    defer config.deinit(t.allocator);
    config.expert_layout = .exl3_k4;
    try t.expectError(error.Exl3TrellisGeometry, residentBytesWithConfig(io, t.allocator, path, &config));
}

test "mimo source refuses an EXL3 shard whose stamp disagrees with the config" {
    const t = std.testing;
    const io = t.io;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();

    // The config every arm parses names k 2.5 / tiny / w16.
    const Case = struct { dir: []const u8, stamp: ?[]const u8, window: expert_exl3.Window, want: ?anyerror };
    const cases = [_]Case{
        .{ .dir = "stamp-matches", .stamp = "\"k\":\"2.5\",\"codebook\":\"tiny\",\"window\":\"16\"", .window = .w16, .want = null },
        .{ .dir = "stamp-window", .stamp = "\"k\":\"2.5\",\"codebook\":\"tiny\",\"window\":\"16\"", .window = .w12, .want = error.Exl3ShardStampMismatch },
        .{ .dir = "stamp-codebook", .stamp = "\"k\":\"2.5\",\"codebook\":\"mul1\",\"window\":\"16\"", .window = .w16, .want = error.Exl3ShardStampMismatch },
        .{ .dir = "stamp-k", .stamp = "\"k\":\"3\",\"codebook\":\"tiny\",\"window\":\"16\"", .window = .w16, .want = error.Exl3ShardStampMismatch },
        // A pack written before the stamp existed is admitted as legacy.
        .{ .dir = "stamp-absent", .stamp = null, .window = .w12, .want = null },
    };
    for (cases) |case| {
        try tmp.dir.createDirPath(io, case.dir);
        var dir = try tmp.dir.openDir(io, case.dir, .{});
        defer dir.close(io);
        try writeTinySource(io, alloc, dir, .{ .exl3 = .{ .n = 40, .stamp = case.stamp } });
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try dir.realPath(io, &path_buf);
        const path = path_buf[0..path_len];
        var config = try model.parseConfig(io, t.allocator, path);
        defer config.deinit(t.allocator);
        config.expert_quant_window = case.window;
        const got = residentBytesWithConfig(io, t.allocator, path, &config);
        if (case.want) |want| try t.expectError(want, got) else _ = try got;
    }
}

test "mimo source FP8 E4M3FN decodes all 256 codes" {
    for (0..256) |i| {
        const code: u8 = @intCast(i);
        const value = fp8E4M3FnValue(code);
        if ((code & 0x7f) == 0x7f) {
            try std.testing.expect(std.math.isNan(value));
            try std.testing.expect(std.math.isNan(fp8_values[code]));
        } else {
            const exponent: u32 = (code >> 3) & 15;
            const mantissa: u32 = code & 7;
            const magnitude_bits: u32 = if (exponent == 0)
                @bitCast(@as(f32, @floatFromInt(mantissa)) / 512)
            else
                ((exponent + 120) << 23) | (mantissa << 20);
            const expected_bits = magnitude_bits | (@as(u32, code & 0x80) << 24);
            try std.testing.expectEqual(expected_bits, @as(u32, @bitCast(value)));
            try std.testing.expectEqual(expected_bits, @as(u32, @bitCast(fp8_values[code])));
            try std.testing.expect(std.math.isFinite(value));
            const negative = fp8E4M3FnValue(code ^ 0x80);
            if (value == 0) {
                if ((code & 0x80) == 0) {
                    try std.testing.expect(std.math.isNegativeZero(negative));
                } else {
                    try std.testing.expect(!std.math.isNegativeZero(negative));
                }
            } else {
                try std.testing.expectEqual(-value, negative);
            }
        }
    }
    try std.testing.expectEqual(@as(f32, 0.0), fp8E4M3FnValue(0));
    try std.testing.expectEqual(@as(f32, 1.0), fp8E4M3FnValue(0x38));
    try std.testing.expectEqual(@as(f32, 448.0), fp8E4M3FnValue(0x7e));
    try std.testing.expectEqual(@as(f32, @exp2(-9.0)), fp8E4M3FnValue(1));
}

fn qkvBoundaryFixture(
    allocator: Allocator,
    tp: usize,
) !DecodedQkv {
    const q_per = 128;
    const k_per = 96;
    const v_per = 64;
    const geometry = QkvGeometry{
        .q_rows = q_per * tp,
        .k_rows = k_per * tp,
        .v_rows = v_per * tp,
    };
    const rows = std.math.cast(usize, geometry.total()) orelse return error.InvalidQkvGeometry;
    const cols = 128;
    const blocks = (rows / tp + 127) / 128;
    const raw = try allocator.alloc(u8, rows * cols);
    defer allocator.free(raw);
    @memset(raw, 0x38);
    const scales = try allocator.alloc(u8, tp * blocks * 4);
    defer allocator.free(scales);
    for (0..tp * blocks) |i| {
        const value: f32 = @floatFromInt(i + 1);
        std.mem.writeInt(u32, scales[i * 4 ..][0..4], @bitCast(value), .little);
    }
    return decodeQkv(
        allocator,
        raw,
        rows,
        cols,
        scales,
        tp * blocks,
        1,
        geometry,
    );
}

test "mimo source TP4 and TP8 QKV regroup keeps rank-local K/V tiles" {
    for ([_]usize{ 4, 8 }) |tp| {
        var decoded = try qkvBoundaryFixture(std.testing.allocator, tp);
        defer decoded.deinit();
        const q_per = 128;
        const k_per = 96;
        const v_per = 64;
        const cols = 128;
        for (0..tp) |rank| {
            for (0..q_per) |row| {
                try std.testing.expectEqual(
                    @as(f32, @floatFromInt(rank * 3 + 1)),
                    decoded.q[(rank * q_per + row) * cols],
                );
            }
            for (0..k_per) |row| {
                try std.testing.expectEqual(
                    @as(f32, @floatFromInt(rank * 3 + 2)),
                    decoded.k[(rank * k_per + row) * cols],
                );
            }
            for (0..32) |row| {
                try std.testing.expectEqual(
                    @as(f32, @floatFromInt(rank * 3 + 2)),
                    decoded.v[(rank * v_per + row) * cols],
                );
            }
            for (32..v_per) |row| {
                try std.testing.expectEqual(
                    @as(f32, @floatFromInt(rank * 3 + 3)),
                    decoded.v[(rank * v_per + row) * cols],
                );
            }
        }
    }
}

test "mimo source rejects QKV input width that differs from hidden size" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var fixture = try makeTinySourceFixture(t.io, t.allocator, &tmp);
    defer fixture.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var source = try loadSourceIndex(t.io, arena.allocator(), fixture.path);
    const key = "model.layers.0.self_attn.qkv_proj.weight";
    var meta = source.tensors.get(key).?;
    meta.shape = &.{ 384, 256 };
    meta.data_start = 0;
    meta.data_end = 384 * 256;
    const scales = source.tensors.getPtr("model.layers.0.self_attn.qkv_proj.weight_scale_inv").?;
    scales.shape = &.{ 4, 2 };
    scales.data_start = 0;
    scales.data_end = 4 * 2 * 4;
    try t.expectError(error.InvalidFp8Shape, validateFp8Pair(&source, arena.allocator(), &fixture.config, key, meta));
}

test "mimo source rejects malformed FP8 scale grids" {
    var raw: [128]u8 = undefined;
    @memset(&raw, 0x38);
    const bad_scales = [_]u8{0};
    try std.testing.expectError(
        error.InvalidFp8ScaleShape,
        decodeBlocks(std.testing.allocator, &raw, 1, 128, &bad_scales, 1, 0),
    );
}

test "mimo source rejects nonfinite and BF16-overflowing decoded weights" {
    const t = std.testing;
    const cases = [_]struct { code: u8, scale: f32 }{
        .{ .code = 0x7f, .scale = 1 },
        .{ .code = 0xff, .scale = 1 },
        .{ .code = 0x7e, .scale = std.math.floatMax(f32) },
        .{ .code = 0x38, .scale = std.math.floatMax(f32) },
    };
    for (cases) |case| {
        var raw: [384 * 128]u8 = @splat(case.code);
        var scales: [16]u8 = undefined;
        for (0..4) |i| std.mem.writeInt(u32, scales[i * 4 ..][0..4], @bitCast(case.scale), .little);
        if (decodeBlocks(t.allocator, raw[0..128], 1, 128, scales[0..4], 1, 1)) |values| {
            t.allocator.free(values);
            return error.TestExpectedError;
        } else |err| try t.expectEqual(error.InvalidFp8Value, err);
        if (decodeQkv(t.allocator, &raw, 384, 128, &scales, 4, 1, .{ .q_rows = 128, .k_rows = 128, .v_rows = 128 })) |result| {
            var values = result;
            values.deinit();
            return error.TestExpectedError;
        } else |err| try t.expectEqual(error.InvalidFp8Value, err);
    }
}

test "mimo source loads a complete split map and preserves dense bytes" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var fixture = try makeTinySourceFixture(io, std.testing.allocator, &tmp);
    defer fixture.deinit();

    const expected_bytes: u64 = 139008;
    try std.testing.expectEqual(
        expected_bytes,
        try residentBytesWithConfig(io, std.testing.allocator, fixture.path, &fixture.config),
    );
    try std.testing.expectEqual(
        expected_bytes,
        try residentBytes(io, std.testing.allocator, fixture.path),
    );

    var weights = try loadWeights(io, std.testing.allocator, fixture.path, &fixture.config);
    defer weights.deinit();
    try std.testing.expectEqual(@as(u32, 24), weights.count());
    try std.testing.expect(weights.get("model.layers.0.self_attn.qkv_proj.weight") == null);
    try std.testing.expect(weights.get("model.layers.0.self_attn.q_proj.weight") != null);
    try std.testing.expect(weights.get("model.layers.0.self_attn.k_proj.scales") != null);
    try std.testing.expect(weights.get("model.layers.0.mlp.gate_proj.biases") != null);
    try std.testing.expect(weights.get("model.layers.0.mlp.experts.0.gate_proj.weight") == null);
    try std.testing.expect(weights.get("model.mtp.layers.0.fake.weight") == null);
    try std.testing.expect(weights.get("visual.fake") == null);

    const embed = weights.get("model.embed_tokens.weight").?;
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(embed));
    const embed_data = mlx.mlx_array_data_bfloat16(embed) orelse return error.TestUnexpectedNullData;
    for (0..fixture.embed.len / 2) |i| {
        const want = std.mem.readInt(u16, fixture.embed[i * 2 ..][0..2], .little);
        try std.testing.expectEqual(want, embed_data[i]);
    }

    const geometry = QkvGeometry{
        .q_rows = 128,
        .k_rows = 128,
        .v_rows = 128,
    };
    var decoded = try decodeQkv(
        std.testing.allocator,
        fixture.qkv_weight,
        384,
        128,
        fixture.qkv_scales,
        4,
        1,
        geometry,
    );
    defer decoded.deinit();
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var expected = try quantizeAffine8(std.testing.allocator, decoded.q, 128, 128, stream);
    defer expected.deinit();
    const loaded_weight = weights.get("model.layers.0.self_attn.q_proj.weight").?;
    const loaded_scales = weights.get("model.layers.0.self_attn.q_proj.scales").?;
    const loaded_biases = weights.get("model.layers.0.self_attn.q_proj.biases").?;
    try std.testing.expectEqual(mlx.mlx_array_size(expected.weight), mlx.mlx_array_size(loaded_weight));
    try std.testing.expectEqual(mlx.mlx_array_size(expected.scales), mlx.mlx_array_size(loaded_scales));
    try std.testing.expectEqual(mlx.mlx_array_size(expected.biases), mlx.mlx_array_size(loaded_biases));
    const want_weight = mlx.mlx_array_data_uint32(expected.weight) orelse return error.TestUnexpectedNullData;
    const got_weight = mlx.mlx_array_data_uint32(loaded_weight) orelse return error.TestUnexpectedNullData;
    for (0..mlx.mlx_array_size(expected.weight)) |i| try std.testing.expectEqual(want_weight[i], got_weight[i]);
    const want_scales = mlx.mlx_array_data_bfloat16(expected.scales) orelse return error.TestUnexpectedNullData;
    const got_scales = mlx.mlx_array_data_bfloat16(loaded_scales) orelse return error.TestUnexpectedNullData;
    for (0..mlx.mlx_array_size(expected.scales)) |i| try std.testing.expectEqual(want_scales[i], got_scales[i]);
    const want_biases = mlx.mlx_array_data_bfloat16(expected.biases) orelse return error.TestUnexpectedNullData;
    const got_biases = mlx.mlx_array_data_bfloat16(loaded_biases) orelse return error.TestUnexpectedNullData;
    for (0..mlx.mlx_array_size(expected.biases)) |i| try std.testing.expectEqual(want_biases[i], got_biases[i]);
}
