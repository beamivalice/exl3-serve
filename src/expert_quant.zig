const std = @import("std");
const io_mod = @import("expert_io.zig");
pub const expert_exl3 = @import("expert_exl3.zig");

pub const Geometry = struct {
    layers: u16,
    experts: u16,
    hidden: u32,
    intermediate: u32,
    /// Absolute layer at which routed expert banks begin. Qwen4 starts at zero;
    /// sparse MoE packs such as MiMo keep a dense layer zero.
    first_moe_layer: u16 = 0,
};

pub const SourceSpan = struct {
    file: u16,
    offset: u64,
    len: u64,
};

pub const QuantGeom = struct {
    bits: u32,
    group_size: u32,
};

/// Affine (bits, group_size) from PACKED shapes alone: a row holds `w_cols * 32`
/// packed bits over `in_dim` values and `s_cols` scale groups cover them.
/// The bit width `w_cols` U32 columns pack `in_dim` weights at, when it is one
/// this build reads.
pub fn affineBitsFromPacked(w_cols: u64, in_dim: u64) ?u32 {
    if (w_cols == 0 or in_dim == 0) return null;
    const packed_bits = std.math.mul(u64, w_cols, 32) catch return null;
    if (packed_bits % in_dim != 0) return null;
    return switch (packed_bits / in_dim) {
        2, 3, 4, 5, 6, 8 => |bits| @intCast(bits),
        else => null,
    };
}

/// The group size `s_cols` scale columns cover `in_dim` weights at.
pub fn affineGroupFromScales(s_cols: u64, in_dim: u64) ?u32 {
    if (s_cols == 0 or in_dim == 0 or in_dim % s_cols != 0) return null;
    return switch (in_dim / s_cols) {
        32, 64, 128 => |gs| @intCast(gs),
        else => null,
    };
}

pub fn affineGeomFromShapes(w_cols: u64, s_cols: u64, in_dim: u64) ?QuantGeom {
    const bits = affineBitsFromPacked(w_cols, in_dim) orelse return null;
    const gs = affineGroupFromScales(s_cols, in_dim) orelse return null;
    return .{ .bits = bits, .group_size = gs };
}

/// Native MiMo MXFP4 is a biasless packed pair, not affine quantization:
/// U32 weights carry four-bit values and U8 e8m0 scales cover groups of 32.
/// Keep this proof at the store boundary so a same-shaped affine tensor cannot
/// silently enter the native byte-preserving path.
pub fn mxfp4GeomFromShapes(
    w_cols: u64,
    s_cols: u64,
    in_dim: u64,
    weight_dtype: io_mod.Dtype,
    scale_dtype: io_mod.Dtype,
    bias_present: bool,
) ?QuantGeom {
    if (bias_present or weight_dtype != .u32 or scale_dtype != .u8) return null;
    const geom = affineGeomFromShapes(w_cols, s_cols, in_dim) orelse return null;
    if (geom.bits != 4 or geom.group_size != 32) return null;
    return geom;
}

pub fn isExpertStreamingArch(model_type: []const u8) bool {
    return std.mem.eql(u8, model_type, "qwen4_exp") or
        std.mem.eql(u8, model_type, "mimo_v2");
}

pub const Exl3Spec = struct {
    rate: expert_exl3.Rate,
    codebook: expert_exl3.Codebook,
};

/// `k` is a rate, integer or fractional: admitted only when 16k is an even
/// whole number of halfwords per tile, which is what every reader indexes by.
fn rateFromConfigK(k_v: std.json.Value) ?expert_exl3.Rate {
    const scaled: f64 = switch (k_v) {
        .integer => |i| @as(f64, @floatFromInt(i)) * 16.0,
        .float => |f| f * 16.0,
        else => return null,
    };
    const rounded = @round(scaled);
    if (@abs(scaled - rounded) > 1e-6) return null;
    if (rounded < 0 or rounded > 1024) return null;
    return expert_exl3.kFromPackedDim(@intFromFloat(rounded));
}

pub fn parseExpertQuant(obj: std.json.ObjectMap) !Exl3Spec {
    const block = obj.get("expert_quant") orelse return error.ExpertLayoutUnsupported;
    if (block != .object) return error.ExpertLayoutUnsupported;
    const format = block.object.get("format") orelse return error.ExpertLayoutUnsupported;
    if (format != .string or !std.mem.eql(u8, format.string, "exl3")) return error.ExpertLayoutUnsupported;
    const k_v = block.object.get("k") orelse return error.ExpertLayoutUnsupported;
    const rate = rateFromConfigK(k_v) orelse return error.ExpertLayoutUnsupported;
    const cb_v = block.object.get("codebook") orelse return error.ExpertLayoutUnsupported;
    if (cb_v != .string) return error.ExpertLayoutUnsupported;
    const codebook = expert_exl3.Codebook.fromName(cb_v.string) orelse return error.ExpertLayoutUnsupported;
    return .{ .rate = rate, .codebook = codebook };
}

pub fn kFromPackedDim(last: u64) ?expert_exl3.Rate {
    if (last > std.math.maxInt(u32)) return null;
    return expert_exl3.kFromPackedDim(@intCast(last));
}

/// Routed experts are leading-index banks or individual source tensors.
/// Both MXFP4 layouts use nine component ids with three absent bias slots.
pub const Layout = enum { bf16_fused, quantized_split, exl3_k4, mxfp4_split, mxfp4_individual };

pub const REDUCE_BANK_TOPK: u32 = 32;

pub fn admitExl3TopK(topk: u32) !void {
    if (topk > REDUCE_BANK_TOPK) return error.Exl3TopKExceedsReduceBank;
}

pub const Component = enum(u4) {
    gate_w,
    gate_s,
    gate_b,
    up_w,
    up_s,
    up_b,
    down_w,
    down_s,
    down_b,
};

pub const component_count: usize = 9;

pub const Projection = enum { gate, up, down };
pub const Part = enum { weight, scales, biases };

pub fn projectionOf(c: Component) Projection {
    return switch (c) {
        .gate_w, .gate_s, .gate_b => .gate,
        .up_w, .up_s, .up_b => .up,
        .down_w, .down_s, .down_b => .down,
    };
}

pub fn partOf(c: Component) Part {
    return switch (c) {
        .gate_w, .up_w, .down_w => .weight,
        .gate_s, .up_s, .down_s => .scales,
        .gate_b, .up_b, .down_b => .biases,
    };
}

pub fn weightOf(p: Projection) Component {
    return switch (p) {
        .gate => .gate_w,
        .up => .up_w,
        .down => .down_w,
    };
}

pub fn scalesOf(p: Projection) Component {
    return switch (p) {
        .gate => .gate_s,
        .up => .up_s,
        .down => .down_s,
    };
}

/// The affine routed-bank nesting, per arch: qwen4 packs sit under
/// `language_model.`, MiMo packs do not.
const AFFINE_PREFIXES = [_][]const u8{ "language_model.model.layers.", "model.layers." };

fn affineTensorKeyAt(buf: []u8, prefix: []const u8, layer: u16, c: Component) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{d}.mlp.switch_mlp.{s}_proj.{s}", .{
        prefix,
        layer,
        @tagName(projectionOf(c)),
        @tagName(partOf(c)),
    });
}

pub fn tensorKey(buf: []u8, layer: u16, c: Component) ![]const u8 {
    return affineTensorKeyAt(buf, AFFINE_PREFIXES[0], layer, c);
}

pub fn fusedTensorKey(buf: []u8, layer: u16, down: bool) ![]const u8 {
    return std.fmt.bufPrint(buf, "model.language_model.layers.{d}.mlp.experts.{s}", .{
        layer,
        if (down) "down_proj" else "gate_up_proj",
    });
}

pub fn mxfp4TensorKey(buf: []u8, layer: u16, projection: Projection, part: Part) ![]const u8 {
    if (part == .biases) return error.Mxfp4BiasUnsupported;
    return std.fmt.bufPrint(buf, "model.layers.{d}.mlp.switch_mlp.{s}_proj.{s}", .{
        layer,
        @tagName(projection),
        @tagName(part),
    });
}

/// Native MiMo source tensors are one expert per tensor. Their packed U8
/// weight bytes are relabelled as U32 by the loader; no payload conversion is
/// needed. The source uses `weight_scale`, not the FP8 trunk's `weight_scale_inv`.
pub fn mxfp4IndividualTensorKey(
    buf: []u8,
    layer: u16,
    expert: u16,
    projection: Projection,
    part: Part,
) ![]const u8 {
    const suffix = switch (part) {
        .weight => "weight",
        .scales => "weight_scale",
        .biases => return error.Mxfp4BiasUnsupported,
    };
    return std.fmt.bufPrint(buf, "model.layers.{d}.mlp.experts.{d}.{s}_proj.{s}", .{
        layer,
        expert,
        @tagName(projection),
        suffix,
    });
}

fn isMxfp4Layout(layout: Layout) bool {
    return layout == .mxfp4_split or layout == .mxfp4_individual;
}

const ParsedMxfp4IndividualKey = struct {
    layer: u16,
    expert: u16,
    projection: Projection,
    part: Part,
};

const Mxfp4IndividualKeyParse = union(enum) {
    not_family,
    malformed,
    valid: ParsedMxfp4IndividualKey,
};

fn decimalU16(text: []const u8) ?u16 {
    if (text.len == 0) return null;
    for (text) |byte| {
        if (!std.ascii.isDigit(byte)) return null;
    }
    return std.fmt.parseInt(u16, text, 10) catch null;
}

fn parseMxfp4IndividualKey(key: []const u8) Mxfp4IndividualKeyParse {
    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, key, prefix)) return .not_family;
    const after_prefix = key[prefix.len..];
    const layer_dot = std.mem.indexOfScalar(u8, after_prefix, '.') orelse return .not_family;
    const layer_text = after_prefix[0..layer_dot];
    if (layer_text.len != 0 and (layer_text[0] == '-' or layer_text[0] == '+')) return .malformed;
    if (layer_text.len == 0 or !std.ascii.isDigit(layer_text[0])) return .not_family;
    const layer = decimalU16(layer_text) orelse return .malformed;

    const experts_prefix = "mlp.experts.";
    const after_layer = after_prefix[layer_dot + 1 ..];
    if (!std.mem.startsWith(u8, after_layer, experts_prefix)) return .not_family;
    const after_experts = after_layer[experts_prefix.len..];
    // The dense fused expert format has no numeric expert component. It is a
    // different family, not malformed individual source metadata.
    if (after_experts.len != 0 and (after_experts[0] == '-' or after_experts[0] == '+')) return .malformed;
    if (after_experts.len == 0 or !std.ascii.isDigit(after_experts[0])) return .not_family;
    const expert_dot = std.mem.indexOfScalar(u8, after_experts, '.') orelse return .malformed;
    const expert = decimalU16(after_experts[0..expert_dot]) orelse return .malformed;
    const tail = after_experts[expert_dot + 1 ..];

    const projection: Projection = if (std.mem.startsWith(u8, tail, "gate_proj."))
        .gate
    else if (std.mem.startsWith(u8, tail, "up_proj."))
        .up
    else if (std.mem.startsWith(u8, tail, "down_proj."))
        .down
    else
        return .malformed;
    const suffix = tail[(switch (projection) {
        .gate => "gate_proj.".len,
        .up => "up_proj.".len,
        .down => "down_proj.".len,
    })..];
    const part: Part = if (std.mem.eql(u8, suffix, "weight"))
        .weight
    else if (std.mem.eql(u8, suffix, "weight_scale"))
        .scales
    else
        return .malformed;
    return .{ .valid = .{ .layer = layer, .expert = expert, .projection = projection, .part = part } };
}

/// True when `key` names a routed-expert bank of `layout` — the tensors the
/// streamed loader must NOT fault into RAM.
pub fn isRoutedExpertKey(layout: Layout, key: []const u8) bool {
    return switch (layout) {
        .bf16_fused => std.mem.startsWith(u8, key, "model.language_model.layers.") and
            (std.mem.endsWith(u8, key, ".mlp.experts.gate_up_proj") or
                std.mem.endsWith(u8, key, ".mlp.experts.down_proj")),
        .quantized_split => (std.mem.startsWith(u8, key, AFFINE_PREFIXES[0]) or
            std.mem.startsWith(u8, key, AFFINE_PREFIXES[1])) and
            std.mem.indexOf(u8, key, ".mlp.switch_mlp.") != null,
        .exl3_k4 => (std.mem.startsWith(u8, key, "language_model.model.layers.") or
            std.mem.startsWith(u8, key, "language_model.mtp.") or
            std.mem.startsWith(u8, key, "model.layers.")) and
            std.mem.indexOf(u8, key, ".mlp.switch_mlp.") != null,
        .mxfp4_split => std.mem.startsWith(u8, key, "model.layers.") and
            std.mem.indexOf(u8, key, ".mlp.switch_mlp.") != null,
        .mxfp4_individual => parseMxfp4IndividualKey(key) == .valid,
    };
}

fn stringAt(map: std.json.ObjectMap, key: []const u8) bool {
    const v = map.get(key) orelse return false;
    return v == .string;
}

fn mxfp4BankCompleteFromFirst(map: std.json.ObjectMap, layers: u16, first_moe_layer: u16) bool {
    if (layers == 0 or first_moe_layer == 0 or first_moe_layer >= layers) return false;
    var buf: [192]u8 = undefined;
    for (first_moe_layer..layers) |layer_usize| {
        const layer: u16 = @intCast(layer_usize);
        for ([_]Projection{ .gate, .up, .down }) |projection| {
            const weight = mxfp4TensorKey(&buf, layer, projection, .weight) catch return false;
            if (!stringAt(map, weight)) return false;
            const scales = mxfp4TensorKey(&buf, layer, projection, .scales) catch return false;
            if (!stringAt(map, scales)) return false;
            const bias = std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.switch_mlp.{s}_proj.biases", .{
                layer,
                @tagName(projection),
            }) catch return false;
            if (map.get(bias) != null) return false;
        }
    }
    // A converter-produced pack has a dense layer zero. Seeing a switch bank
    // there would make the absolute layer map ambiguous, so decline it.
    for (0..first_moe_layer) |layer_usize| {
        const layer: u16 = @intCast(layer_usize);
        for ([_]Projection{ .gate, .up, .down }) |projection| {
            for ([_]Part{ .weight, .scales }) |part| {
                const key = mxfp4TensorKey(&buf, layer, projection, part) catch return false;
                if (map.get(key) != null) return false;
            }
            const bias = std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.switch_mlp.{s}_proj.biases", .{
                layer,
                @tagName(projection),
            }) catch return false;
            if (map.get(bias) != null) return false;
        }
    }
    return true;
}

const IndividualMapState = struct {
    seen: bool = false,
    malformed: bool = false,
    out_of_range: bool = false,
    dense_prefix: bool = false,
    max_expert: u16 = 0,
};

fn inspectMxfp4IndividualMap(
    map: std.json.ObjectMap,
    layers: u16,
    experts: ?u16,
    first_moe_layer: u16,
) IndividualMapState {
    var state: IndividualMapState = .{};
    var it = map.iterator();
    while (it.next()) |entry| {
        switch (parseMxfp4IndividualKey(entry.key_ptr.*)) {
            .not_family => {},
            .malformed => state.malformed = true,
            .valid => |parsed| {
                state.seen = true;
                state.max_expert = @max(state.max_expert, parsed.expert);
                if (entry.value_ptr.* != .string) state.malformed = true;
                if (parsed.layer < first_moe_layer) state.dense_prefix = true;
                if (parsed.layer >= layers or (experts != null and parsed.expert >= experts.?))
                    state.out_of_range = true;
            },
        }
    }
    return state;
}

fn mxfp4IndividualBankCompleteFromFirst(
    map: std.json.ObjectMap,
    layers: u16,
    experts: ?u16,
    first_moe_layer: u16,
) bool {
    if (layers == 0 or first_moe_layer == 0 or first_moe_layer >= layers) return false;
    const state = inspectMxfp4IndividualMap(map, layers, experts, first_moe_layer);
    if (!state.seen or state.malformed or state.out_of_range or state.dense_prefix) return false;
    const expert_count: u32 = if (experts) |count|
        count
    else
        @as(u32, state.max_expert) + 1;
    if (expert_count == 0 or expert_count > std.math.maxInt(u16)) return false;

    var buf: [192]u8 = undefined;
    for (first_moe_layer..layers) |layer_usize| {
        const layer: u16 = @intCast(layer_usize);
        for (0..expert_count) |expert_usize| {
            const expert: u16 = @intCast(expert_usize);
            for ([_]Projection{ .gate, .up, .down }) |projection| {
                for ([_]Part{ .weight, .scales }) |part| {
                    const key = mxfp4IndividualTensorKey(&buf, layer, expert, projection, part) catch return false;
                    if (!stringAt(map, key)) return false;
                }
            }
        }
    }
    return true;
}

/// `language_model.model.layers.` is the qwen4 pack's nesting, `model.layers.`
/// MiMo's; a pack uses one of them for every routed layer it owns.
const EXL3_PREFIXES = [_][]const u8{ "language_model.model.layers.", "model.layers." };

fn exl3BankComplete(map: std.json.ObjectMap, prefix: []const u8, first_moe_layer: u16, layers: u16) bool {
    if (first_moe_layer >= layers) return false;
    var buf: [192]u8 = undefined;
    var layer: u16 = first_moe_layer;
    while (layer < layers) : (layer += 1) {
        for ([_][]const u8{ "gate", "up", "down" }) |proj| {
            for ([_][]const u8{ "trellis", "suh", "svh" }) |part| {
                const key = std.fmt.bufPrint(&buf, "{s}{d}.mlp.switch_mlp.{s}_proj.{s}", .{
                    prefix, layer, proj, part,
                }) catch return false;
                if (!stringAt(map, key)) return false;
            }
        }
    }
    return true;
}

fn affineBankComplete(map: std.json.ObjectMap, prefix: []const u8, first_moe_layer: u16, layers: u16) bool {
    if (first_moe_layer >= layers) return false;
    var buf: [192]u8 = undefined;
    var layer: u16 = first_moe_layer;
    while (layer < layers) : (layer += 1) {
        for (0..component_count) |ci| {
            const key = affineTensorKeyAt(&buf, prefix, layer, @fromBackingInt(@intCast(ci))) catch return false;
            if (!stringAt(map, key)) return false;
        }
    }
    return true;
}

fn hasAnyAffineKey(map: std.json.ObjectMap, prefix: []const u8, layers: u16) bool {
    var buf: [192]u8 = undefined;
    for (0..layers) |layer| {
        for (0..component_count) |ci| {
            const key = affineTensorKeyAt(&buf, prefix, @intCast(layer), @fromBackingInt(@intCast(ci))) catch return true;
            if (map.get(key) != null) return true;
        }
    }
    return false;
}

fn hasAnyExl3Key(map: std.json.ObjectMap, layers: u16) bool {
    var buf: [192]u8 = undefined;
    for (EXL3_PREFIXES) |prefix| {
        for (0..layers) |layer| {
            for ([_][]const u8{ "gate", "up", "down" }) |proj| {
                for ([_][]const u8{ "trellis", "suh", "svh" }) |part| {
                    const key = std.fmt.bufPrint(&buf, "{s}{d}.mlp.switch_mlp.{s}_proj.{s}", .{
                        prefix, layer, proj, part,
                    }) catch return true;
                    if (map.get(key) != null) return true;
                }
            }
        }
    }
    return false;
}

/// Affine or MXFP4 routed keys beside an EXL3 bank: the pack is mixed.
fn hasAnyAffineOrMxfp4ExpertKey(map: std.json.ObjectMap, layers: u16) bool {
    if (hasAnyMxfp4SplitKey(map, layers)) return true;
    return hasAnyAffineKey(map, AFFINE_PREFIXES[0], layers);
}

fn hasAnyMxfp4SplitKey(map: std.json.ObjectMap, layers: u16) bool {
    var buf: [192]u8 = undefined;
    for (0..layers) |layer_usize| {
        const layer: u16 = @intCast(layer_usize);
        for ([_]Projection{ .gate, .up, .down }) |projection| {
            for ([_]Part{ .weight, .scales }) |part| {
                const key = mxfp4TensorKey(&buf, layer, projection, part) catch return true;
                if (map.get(key) != null) return true;
            }
            const bias = std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.switch_mlp.{s}_proj.biases", .{
                layer,
                @tagName(projection),
            }) catch return true;
            if (map.get(bias) != null) return true;
        }
    }
    return false;
}

fn hasAnyAlternativeExpertKey(map: std.json.ObjectMap, layers: u16, include_mxfp4_split: bool) bool {
    if (include_mxfp4_split and hasAnyMxfp4SplitKey(map, layers)) return true;
    var buf: [192]u8 = undefined;
    for (0..layers) |layer_usize| {
        const layer: u16 = @intCast(layer_usize);
        for (0..component_count) |ci| {
            const c: Component = @fromBackingInt(@intCast(ci));
            const key = tensorKey(&buf, layer, c) catch return true;
            if (map.get(key) != null) return true;
        }
    }
    return hasAnyExl3Key(map, layers);
}

fn layoutFromWeightMapWithFirstMoe(map: std.json.ObjectMap, layers: u16, first_moe_layer: u16, mimo: bool) ?Layout {
    if (layers == 0) return null;
    // Individual source keys are intentionally checked before every packed
    // layout. A complete bank beside any other routed bank is ambiguous.
    const individual_state = inspectMxfp4IndividualMap(map, layers, null, first_moe_layer);
    if (individual_state.malformed) return null;
    if (individual_state.seen) {
        if (individual_state.out_of_range or individual_state.dense_prefix or
            !mxfp4IndividualBankCompleteFromFirst(map, layers, null, first_moe_layer) or
            hasAnyAlternativeExpertKey(map, layers, true))
            return null;
        return .mxfp4_individual;
    }
    var buf: [192]u8 = undefined;
    if (mxfp4BankCompleteFromFirst(map, layers, first_moe_layer)) {
        return if (hasAnyAlternativeExpertKey(map, layers, false)) null else .mxfp4_split;
    }
    var fused = true;
    for (0..layers) |layer| {
        const gate = fusedTensorKey(&buf, @intCast(layer), false) catch return null;
        if (!stringAt(map, gate)) {
            fused = false;
            break;
        }
        const down = fusedTensorKey(&buf, @intCast(layer), true) catch return null;
        if (!stringAt(map, down)) {
            fused = false;
            break;
        }
    }
    if (fused) return .bf16_fused;
    // The nesting is the ARCH's: a pack the loader cannot address is not a pack.
    const exl3_ok = if (mimo)
        exl3BankComplete(map, EXL3_PREFIXES[1], first_moe_layer, layers)
    else
        exl3BankComplete(map, EXL3_PREFIXES[0], 0, layers);
    if (exl3_ok) return if (hasAnyAffineOrMxfp4ExpertKey(map, layers)) null else .exl3_k4;
    const affine_prefix = if (mimo) AFFINE_PREFIXES[1] else AFFINE_PREFIXES[0];
    const affine_first: u16 = if (mimo) first_moe_layer else 0;
    if (!affineBankComplete(map, affine_prefix, affine_first, layers)) return null;
    return .quantized_split;
}

pub fn layoutFromWeightMapForFirstMoe(map: std.json.ObjectMap, layers: u16, first_moe_layer: u16) ?Layout {
    return layoutFromWeightMapWithFirstMoe(map, layers, first_moe_layer, false);
}

pub fn layoutFromWeightMap(map: std.json.ObjectMap, layers: u16) ?Layout {
    return layoutFromWeightMapWithFirstMoe(map, layers, 1, false);
}

pub fn layoutFromIndexJsonWithFirstMoe(
    allocator: std.mem.Allocator,
    model_type: []const u8,
    raw: []const u8,
    layers: u16,
    first_moe_layer: u16,
) ?Layout {
    if (!isExpertStreamingArch(model_type)) return null;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const map = parsed.value.object.get("weight_map") orelse return null;
    if (map != .object) return null;
    const mimo = std.mem.eql(u8, model_type, "mimo_v2");
    const layout = layoutFromWeightMapWithFirstMoe(map.object, layers, first_moe_layer, mimo) orelse return null;
    if (layout == .mxfp4_split or layout == .mxfp4_individual) {
        return if (mimo) layout else null;
    }
    // EXL3 and affine stacked banks serve on both arches, each under its own
    // nesting; only the dense bf16 pack is qwen4's alone.
    if (layout == .exl3_k4 or layout == .quantized_split) return layout;
    return if (mimo) null else layout;
}

pub fn layoutFromIndexJson(allocator: std.mem.Allocator, model_type: []const u8, raw: []const u8, layers: u16) ?Layout {
    return layoutFromIndexJsonWithFirstMoe(allocator, model_type, raw, layers, 1);
}

pub fn layoutOfDirWithFirstMoe(
    allocator: std.mem.Allocator,
    io: std.Io,
    model_type: []const u8,
    model_dir: []const u8,
    layers: u16,
    first_moe_layer: u16,
) ?Layout {
    if (!isExpertStreamingArch(model_type)) return null;
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return null;
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return null;
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(64 * 1024 * 1024)) catch return null;
    defer allocator.free(raw);
    return layoutFromIndexJsonWithFirstMoe(allocator, model_type, raw, layers, first_moe_layer);
}

pub fn layoutOfDir(allocator: std.mem.Allocator, io: std.Io, model_type: []const u8, model_dir: []const u8, layers: u16) ?Layout {
    return layoutOfDirWithFirstMoe(allocator, io, model_type, model_dir, layers, 1);
}

const SourceHeader = struct {
    parsed: std.json.Parsed(std.json.Value),
    data_offset: u64,
    file_size: u64,
};

const SourceFile = struct {
    name: []u8,
    fd: std.c.fd_t,
    header: ?SourceHeader = null,
};

pub fn moeLayerCount(geometry: Geometry) u16 {
    if (geometry.first_moe_layer >= geometry.layers) return 0;
    return geometry.layers - geometry.first_moe_layer;
}

/// The nine leading-index banks of a streamed quantized expert pack, resolved
/// to per-expert byte spans. Per-tensor geometry comes from PACKED shapes and
/// dtypes against the activation dim, never from a model-wide quantization hint.
pub const QuantStore = struct {
    allocator: std.mem.Allocator,
    geometry: Geometry,
    layout: Layout,
    files: []SourceFile,
    spans: []SourceSpan,
    slot_bytes: [component_count]u64,
    rows: [component_count]u32,
    cols: [component_count]u32,
    dtypes: [component_count]io_mod.Dtype,
    geoms: [component_count]QuantGeom,

    fn sourceIndex(self: *const QuantStore, layer: u16, expert: u16, c: Component) usize {
        return ((@as(usize, layer) * self.geometry.experts) + expert) * component_count + @backingInt(c);
    }

    pub fn hasExpertLayer(self: *const QuantStore, layer: u16) bool {
        return layer < self.geometry.layers and layer >= self.geometry.first_moe_layer;
    }

    pub fn componentPresent(self: *const QuantStore, c: Component) bool {
        return !isMxfp4Layout(self.layout) or partOf(c) != .biases;
    }

    pub fn span(self: *const QuantStore, layer: u16, expert: u16, c: Component) SourceSpan {
        if (!self.hasExpertLayer(layer) or !self.componentPresent(c)) return .{ .file = 0, .offset = 0, .len = 0 };
        return self.spans[self.sourceIndex(layer, expert, c)];
    }

    pub fn slotBytes(self: *const QuantStore, c: Component) u64 {
        return self.slot_bytes[@backingInt(c)];
    }

    pub fn geomOf(self: *const QuantStore, c: Component) QuantGeom {
        return self.geoms[@backingInt(c)];
    }

    pub fn rowsOf(self: *const QuantStore, c: Component) u32 {
        return self.rows[@backingInt(c)];
    }

    pub fn colsOf(self: *const QuantStore, c: Component) u32 {
        return self.cols[@backingInt(c)];
    }

    pub fn dtypeOf(self: *const QuantStore, c: Component) io_mod.Dtype {
        return self.dtypes[@backingInt(c)];
    }

    pub fn expertBytes(self: *const QuantStore) u64 {
        var total: u64 = 0;
        for (self.slot_bytes) |b| total +|= b;
        return total;
    }

    pub fn open(allocator: std.mem.Allocator, model_dir: []const u8, geometry: Geometry) !QuantStore {
        return openForLayout(allocator, model_dir, geometry, .quantized_split);
    }

    pub fn openForLayout(
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        geometry: Geometry,
        chosen: Layout,
    ) !QuantStore {
        if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return error.InvalidExpertModelPath;
        if (geometry.layers == 0 or geometry.experts == 0 or geometry.hidden == 0 or geometry.intermediate == 0 or
            geometry.first_moe_layer > geometry.layers)
            return error.InvalidExpertGeometry;
        if (isMxfp4Layout(chosen) and
            (geometry.first_moe_layer == 0 or moeLayerCount(geometry) == 0))
            return error.Mxfp4DensePrefixMismatch;

        const io = std.Io.Threaded.global_single_threaded.io();
        var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
        defer dir.close(io);
        const index_raw = try dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(64 * 1024 * 1024));
        defer allocator.free(index_raw);
        const index_parsed = std.json.parseFromSlice(std.json.Value, allocator, index_raw, .{}) catch return error.InvalidSafetensorsIndex;
        defer index_parsed.deinit();
        if (index_parsed.value != .object) return error.InvalidSafetensorsIndex;
        const weight_map_value = index_parsed.value.object.get("weight_map") orelse return error.InvalidSafetensorsIndex;
        if (weight_map_value != .object) return error.InvalidSafetensorsIndex;
        const weight_map = weight_map_value.object;

        var files_list: std.ArrayList(SourceFile) = .empty;
        errdefer {
            for (files_list.items) |*file| deinitSourceFile(allocator, file);
            files_list.deinit(allocator);
        }
        const per_layer = std.math.mul(usize, geometry.experts, component_count) catch return error.InvalidExpertGeometry;
        const span_count = std.math.mul(usize, geometry.layers, per_layer) catch return error.InvalidExpertGeometry;
        const spans = try allocator.alloc(SourceSpan, span_count);
        errdefer allocator.free(spans);
        for (spans) |*span_value| span_value.* = .{ .file = 0, .offset = 0, .len = 0 };

        var store = QuantStore{
            .allocator = allocator,
            .geometry = geometry,
            .layout = chosen,
            .files = &.{},
            .spans = spans,
            .slot_bytes = @splat(0),
            .rows = @splat(0),
            .cols = @splat(0),
            .dtypes = @splat(.other),
            .geoms = @splat(.{ .bits = 0, .group_size = 0 }),
        };

        var key_buf: [192]u8 = undefined;
        const first_layer: u16 = if (isMxfp4Layout(chosen)) geometry.first_moe_layer else 0;
        if (chosen == .mxfp4_split) {
            // Banks before first_moe_layer must be absent. A dense layer with a
            // stray switch tensor is not safe to reinterpret as a routed bank.
            for (0..first_layer) |layer_usize| {
                const layer: u16 = @intCast(layer_usize);
                for ([_]Projection{ .gate, .up, .down }) |projection| {
                    for ([_]Part{ .weight, .scales }) |part| {
                        const key = mxfp4TensorKey(&key_buf, layer, projection, part) catch return error.InvalidExpertGeometry;
                        if (weight_map.get(key) != null) return error.Mxfp4DensePrefixMismatch;
                    }
                    const bias = std.fmt.bufPrint(&key_buf, "model.layers.{d}.mlp.switch_mlp.{s}_proj.biases", .{
                        layer,
                        @tagName(projection),
                    }) catch return error.InvalidExpertGeometry;
                    if (weight_map.get(bias) != null) return error.Mxfp4BiasUnsupported;
                }
            }
        }

        if (chosen == .mxfp4_individual) {
            try populateMxfp4IndividualStore(
                allocator,
                model_dir,
                weight_map,
                &files_list,
                &store,
            );
        } else {
            var metadata_ready = false;
            for (first_layer..geometry.layers) |layer_usize| {
                const layer: u16 = @intCast(layer_usize);
                for (0..component_count) |ci| {
                    const c: Component = @fromBackingInt(@intCast(ci));
                    if (chosen == .mxfp4_split and partOf(c) == .biases) {
                        const bias_key = std.fmt.bufPrint(&key_buf, "model.layers.{d}.mlp.switch_mlp.{s}_proj.biases", .{
                            layer,
                            @tagName(projectionOf(c)),
                        }) catch return error.InvalidExpertGeometry;
                        if (weight_map.get(bias_key) != null) return error.Mxfp4BiasUnsupported;
                        continue;
                    }
                    const key = if (chosen == .mxfp4_split)
                        mxfp4TensorKey(&key_buf, layer, projectionOf(c), partOf(c)) catch return error.InvalidExpertGeometry
                    else
                        tensorKey(&key_buf, layer, c) catch return error.InvalidExpertGeometry;
                    const mapped = weight_map.get(key) orelse return error.MissingExpertTensor;
                    if (mapped != .string) return error.InvalidSafetensorsIndex;
                    const file = try openSource(allocator, &files_list, model_dir, mapped.string);
                    const region = io_mod.tensorRegion(allocator, files_list.items[file].fd, key) catch |err| return switch (err) {
                        error.MissingSafetensorsTensor => error.MissingExpertTensor,
                        error.SafetensorsTensorOutOfBounds => error.ExpertTensorOutOfBounds,
                        else => error.InvalidExpertTensor,
                    };
                    if (region.rank != 3 or region.shape[0] != geometry.experts) return error.InvalidExpertTensor;
                    if (region.shape[1] == 0 or region.shape[1] > std.math.maxInt(u32)) return error.InvalidExpertTensor;
                    if (region.shape[2] == 0 or region.shape[2] > std.math.maxInt(u32)) return error.InvalidExpertTensor;
                    const elem: u64 = switch (region.dtype) {
                        .bf16 => 2,
                        .u8 => 1,
                        .u32 => 4,
                        .other => return error.InvalidExpertTensor,
                    };
                    const expected_rows = std.math.mul(u64, region.shape[0], region.shape[1]) catch return error.InvalidExpertTensor;
                    const expected_elems = std.math.mul(u64, expected_rows, region.shape[2]) catch return error.InvalidExpertTensor;
                    const expected_bytes = std.math.mul(u64, elem, expected_elems) catch return error.InvalidExpertTensor;
                    if (region.tensor_bytes != expected_bytes) return error.InvalidExpertTensor;
                    const per_expert = region.tensor_bytes / geometry.experts;
                    if (!metadata_ready) {
                        store.rows[ci] = @intCast(region.shape[1]);
                        store.cols[ci] = @intCast(region.shape[2]);
                        store.dtypes[ci] = region.dtype;
                        store.slot_bytes[ci] = per_expert;
                    } else if (store.rows[ci] != region.shape[1] or store.cols[ci] != region.shape[2] or
                        store.dtypes[ci] != region.dtype or store.slot_bytes[ci] != per_expert)
                    {
                        return error.MixedExpertBankGeometry;
                    }
                    const base = std.math.add(u64, region.data_offset, region.tensor_offset) catch return error.InvalidExpertTensor;
                    for (0..geometry.experts) |expert| {
                        spans[(layer_usize * geometry.experts + expert) * component_count + ci] = .{
                            .file = file,
                            .offset = std.math.add(u64, base, @as(u64, expert) * per_expert) catch return error.InvalidExpertTensor,
                            .len = per_expert,
                        };
                    }
                    // The first layer establishes all component metadata; later
                    // layers are checked against it above.
                    if (ci + 1 == component_count or
                        (chosen == .mxfp4_split and ci == @backingInt(Component.down_s)))
                        metadata_ready = true;
                }
            }
            if (!metadata_ready) return error.MissingExpertTensor;
        }
        for ([_]Projection{ .gate, .up, .down }) |p| {
            const in_dim: u64 = if (p == .down) geometry.intermediate else geometry.hidden;
            const out_rows: u64 = if (p == .down) geometry.hidden else geometry.intermediate;
            const w = weightOf(p);
            const sc = scalesOf(p);
            if (store.rows[@backingInt(w)] != out_rows or store.rows[@backingInt(sc)] != out_rows)
                return error.InvalidExpertTensor;
            if (isMxfp4Layout(chosen)) {
                if (store.dtypes[@backingInt(w)] != .u32 or store.dtypes[@backingInt(sc)] != .u8)
                    return error.InvalidExpertTensor;
                const geom = mxfp4GeomFromShapes(
                    store.cols[@backingInt(w)],
                    store.cols[@backingInt(sc)],
                    in_dim,
                    store.dtypes[@backingInt(w)],
                    store.dtypes[@backingInt(sc)],
                    false,
                ) orelse return error.UnsupportedExpertQuant;
                store.geoms[@backingInt(w)] = geom;
                store.geoms[@backingInt(sc)] = geom;
                if (chosen == .mxfp4_split) {
                    const bias_key = std.fmt.bufPrint(&key_buf, "model.layers.{d}.mlp.switch_mlp.{s}_proj.biases", .{
                        first_layer,
                        @tagName(p),
                    }) catch return error.InvalidExpertGeometry;
                    if (weight_map.get(bias_key) != null) return error.Mxfp4BiasUnsupported;
                }
            } else {
                if (store.dtypes[@backingInt(w)] != .u32 or store.dtypes[@backingInt(sc)] != .bf16)
                    return error.InvalidExpertTensor;
                const geom = affineGeomFromShapes(store.cols[@backingInt(w)], store.cols[@backingInt(sc)], in_dim) orelse
                    return error.UnsupportedExpertQuant;
                for ([_]Component{ w, sc, biasesOf(p) }) |c| store.geoms[@backingInt(c)] = geom;
                const b = biasesOf(p);
                if (store.dtypes[@backingInt(b)] != .bf16 or store.cols[@backingInt(b)] != store.cols[@backingInt(sc)] or
                    store.rows[@backingInt(b)] != out_rows) return error.InvalidExpertTensor;
            }
        }

        // Parsed headers are needed only while constructing the source spans.
        for (files_list.items) |*file| {
            if (file.header) |*header| header.parsed.deinit();
            file.header = null;
        }
        store.files = try files_list.toOwnedSlice(allocator);
        return store;
    }

    pub fn deinit(self: *QuantStore) void {
        for (self.files) |*file| deinitSourceFile(self.allocator, file);
        self.allocator.free(self.files);
        self.allocator.free(self.spans);
        self.* = undefined;
    }

    pub fn readSpan(self: *const QuantStore, span_value: SourceSpan, dst: []u8) !void {
        if (span_value.file >= self.files.len or span_value.len != dst.len) return error.InvalidExpertRead;
        try io_mod.readExact(self.files[span_value.file].fd, dst, span_value.offset);
    }

    /// The nine slices of one expert, concatenated in `Component` order.
    pub fn readExpert(self: *const QuantStore, layer: u16, expert: u16, dst: []u8) !void {
        if (layer >= self.geometry.layers or expert >= self.geometry.experts) return error.ExpertOutOfRange;
        if (dst.len != self.expertBytes()) return error.InvalidExpertRead;
        var at: usize = 0;
        for (0..component_count) |ci| {
            const c: Component = @fromBackingInt(@intCast(ci));
            if (!self.componentPresent(c)) continue;
            const s = self.span(layer, expert, c);
            const len: usize = @intCast(s.len);
            try self.readSpan(s, dst[at..][0..len]);
            at += len;
        }
    }
};

pub fn biasesOf(p: Projection) Component {
    return switch (p) {
        .gate => .gate_b,
        .up => .up_b,
        .down => .down_b,
    };
}

fn deinitSourceFile(allocator: std.mem.Allocator, file: *SourceFile) void {
    if (file.header) |*header| header.parsed.deinit();
    _ = std.c.close(file.fd);
    allocator.free(file.name);
}

fn cacheSourceHeader(allocator: std.mem.Allocator, file: *SourceFile) !void {
    if (file.header != null) return;
    var len_bytes: [8]u8 = undefined;
    try io_mod.readExact(file.fd, &len_bytes, 0);
    const header_len = std.mem.readInt(u64, &len_bytes, .little);
    if (header_len == 0 or header_len > 128 * 1024 * 1024) return error.InvalidSafetensorsHeader;
    const header_bytes = try allocator.alloc(u8, @intCast(header_len));
    defer allocator.free(header_bytes);
    try io_mod.readExact(file.fd, header_bytes, 8);
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, header_bytes, .{}) catch
        return error.InvalidSafetensorsHeader;
    errdefer parsed.deinit();
    if (parsed.value != .object) return error.InvalidSafetensorsHeader;

    var st: std.c.Stat = undefined;
    if (std.c.fstat(file.fd, &st) != 0 or st.size < 0) return error.InvalidSafetensorsHeader;
    const data_offset = std.math.add(u64, 8, header_len) catch return error.InvalidSafetensorsHeader;
    file.header = .{
        .parsed = parsed,
        .data_offset = data_offset,
        .file_size = @intCast(st.size),
    };
}

fn cachedTensorRegion(file: *const SourceFile, key: []const u8) !io_mod.TensorRegion {
    const header = file.header orelse return error.InvalidSafetensorsHeader;
    const value = header.parsed.value.object.get(key) orelse return error.MissingSafetensorsTensor;
    if (value != .object) return error.InvalidSafetensorsTensor;
    const object = value.object;
    const dtype = object.get("dtype") orelse return error.InvalidSafetensorsTensor;
    if (dtype != .string) return error.InvalidSafetensorsTensor;
    const dt: io_mod.Dtype = if (std.mem.eql(u8, dtype.string, "BF16"))
        .bf16
    else if (std.mem.eql(u8, dtype.string, "U8") or std.mem.eql(u8, dtype.string, "UINT8"))
        .u8
    else if (std.mem.eql(u8, dtype.string, "U32") or std.mem.eql(u8, dtype.string, "UINT32"))
        .u32
    else
        .other;
    const shape = object.get("shape") orelse return error.InvalidSafetensorsTensor;
    if (shape != .array or shape.array.items.len < 2 or shape.array.items.len > 3)
        return error.InvalidSafetensorsTensor;
    var dimensions: [3]u64 = .{ 0, 0, 0 };
    var elements: u64 = 1;
    for (shape.array.items, 0..) |dim, i| {
        if (dim != .integer or dim.integer <= 0) return error.InvalidSafetensorsTensor;
        dimensions[i] = @intCast(dim.integer);
        elements = std.math.mul(u64, elements, dimensions[i]) catch return error.InvalidSafetensorsTensor;
    }
    if (elements == 0) return error.InvalidSafetensorsTensor;
    const offsets = object.get("data_offsets") orelse return error.InvalidSafetensorsTensor;
    if (offsets != .array or offsets.array.items.len != 2) return error.InvalidSafetensorsTensor;
    const start = offsets.array.items[0];
    const end = offsets.array.items[1];
    if (start != .integer or end != .integer or start.integer < 0 or end.integer < start.integer)
        return error.InvalidSafetensorsTensor;
    const start_u: u64 = @intCast(start.integer);
    const end_u: u64 = @intCast(end.integer);
    const bytes = end_u - start_u;
    const absolute_end = std.math.add(u64, header.data_offset, end_u) catch return error.InvalidSafetensorsTensor;
    if (absolute_end > header.file_size) return error.SafetensorsTensorOutOfBounds;
    return .{
        .data_offset = header.data_offset,
        .tensor_offset = start_u,
        .tensor_bytes = bytes,
        .shape = dimensions,
        .rank = @intCast(shape.array.items.len),
        .dtype = dt,
    };
}

fn sourceTensorRegion(allocator: std.mem.Allocator, file: *SourceFile, key: []const u8) !io_mod.TensorRegion {
    try cacheSourceHeader(allocator, file);
    return cachedTensorRegion(file, key);
}

fn openSource(allocator: std.mem.Allocator, list: *std.ArrayList(SourceFile), dir_path: []const u8, name: []const u8) !u16 {
    for (list.items, 0..) |file, i| {
        if (std.mem.eql(u8, file.name, name)) return @intCast(i);
    }
    if (list.items.len >= std.math.maxInt(u16)) return error.TooManyExpertShards;
    const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ dir_path, name }, 0);
    defer allocator.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.MissingExpertShard;
    errdefer _ = std.c.close(fd);
    _ = std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1));
    const owned_name = try allocator.dupe(u8, name);
    errdefer allocator.free(owned_name);
    try list.append(allocator, .{ .name = owned_name, .fd = fd });
    return @intCast(list.items.len - 1);
}

fn validateMxfp4IndividualMap(
    map: std.json.ObjectMap,
    geometry: Geometry,
) !void {
    const state = inspectMxfp4IndividualMap(
        map,
        geometry.layers,
        geometry.experts,
        geometry.first_moe_layer,
    );
    if (!state.seen) return error.MissingExpertTensor;
    if (state.dense_prefix) return error.Mxfp4DensePrefixMismatch;
    if (state.malformed or state.out_of_range or hasAnyAlternativeExpertKey(map, geometry.layers, true))
        return error.InvalidSafetensorsIndex;
    if (!mxfp4IndividualBankCompleteFromFirst(
        map,
        geometry.layers,
        geometry.experts,
        geometry.first_moe_layer,
    ))
        return error.MissingExpertTensor;
}

fn populateMxfp4IndividualStore(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    weight_map: std.json.ObjectMap,
    files_list: *std.ArrayList(SourceFile),
    store: *QuantStore,
) !void {
    const geometry = store.geometry;
    try validateMxfp4IndividualMap(weight_map, geometry);

    var metadata_ready = false;
    var key_buf: [192]u8 = undefined;
    for (geometry.first_moe_layer..geometry.layers) |layer_usize| {
        const layer: u16 = @intCast(layer_usize);
        for (0..geometry.experts) |expert_usize| {
            const expert: u16 = @intCast(expert_usize);
            for ([_]Projection{ .gate, .up, .down }) |projection| {
                const in_dim: u64 = if (projection == .down) geometry.intermediate else geometry.hidden;
                const out_rows: u64 = if (projection == .down) geometry.hidden else geometry.intermediate;
                if (in_dim % 32 != 0) return error.UnsupportedExpertQuant;
                for ([_]Part{ .weight, .scales }) |part| {
                    const c: Component = if (part == .weight) weightOf(projection) else scalesOf(projection);
                    const key = mxfp4IndividualTensorKey(&key_buf, layer, expert, projection, part) catch
                        return error.InvalidExpertGeometry;
                    const mapped = weight_map.get(key) orelse return error.MissingExpertTensor;
                    if (mapped != .string) return error.InvalidSafetensorsIndex;
                    const file = try openSource(allocator, files_list, model_dir, mapped.string);
                    const region = sourceTensorRegion(allocator, &files_list.items[file], key) catch |err| return switch (err) {
                        error.MissingSafetensorsTensor => error.MissingExpertTensor,
                        error.SafetensorsTensorOutOfBounds => error.ExpertTensorOutOfBounds,
                        else => error.InvalidExpertTensor,
                    };
                    if (region.rank != 2 or region.dtype != .u8 or
                        region.shape[0] != out_rows)
                        return error.InvalidExpertTensor;
                    const raw_cols: u64 = if (part == .weight) in_dim / 2 else in_dim / 32;
                    const canonical_cols: u64 = if (part == .weight) in_dim / 8 else in_dim / 32;
                    if (region.shape[1] != raw_cols or raw_cols > std.math.maxInt(u32) or
                        canonical_cols > std.math.maxInt(u32))
                        return error.InvalidExpertTensor;
                    const expected_bytes = std.math.mul(u64, out_rows, raw_cols) catch return error.InvalidExpertTensor;
                    if (region.tensor_bytes != expected_bytes) return error.InvalidExpertTensor;
                    const canonical_dtype: io_mod.Dtype = if (part == .weight) .u32 else .u8;
                    const slot_bytes = region.tensor_bytes;
                    const ci = @backingInt(c);
                    if (!metadata_ready) {
                        store.rows[ci] = @intCast(out_rows);
                        store.cols[ci] = @intCast(canonical_cols);
                        store.dtypes[ci] = canonical_dtype;
                        store.slot_bytes[ci] = slot_bytes;
                    } else if (store.rows[ci] != out_rows or store.cols[ci] != canonical_cols or
                        store.dtypes[ci] != canonical_dtype or store.slot_bytes[ci] != slot_bytes)
                    {
                        return error.MixedExpertBankGeometry;
                    }
                    const base = std.math.add(u64, region.data_offset, region.tensor_offset) catch return error.InvalidExpertTensor;
                    store.spans[store.sourceIndex(layer, expert, c)] = .{
                        .file = file,
                        .offset = base,
                        .len = slot_bytes,
                    };
                    if (layer == geometry.first_moe_layer and expert == 0 and
                        projection == .down and part == .scales)
                        metadata_ready = true;
                }
            }
        }
    }
    if (!metadata_ready) return error.MissingExpertTensor;
}

/// Write a small raw MiMo expert source fixture. The payload intentionally
/// stores each expert as down, gate, up, unlike Component order, so tests prove
/// that spans come from each header entry rather than a guessed fixed stride.
pub fn writeTinyMxfp4IndividualCheckpoint(
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    experts: u16,
    hidden: u32,
    inter: u32,
) ![]u8 {
    if (experts == 0 or hidden == 0 or inter == 0 or hidden % 32 != 0 or inter % 32 != 0)
        return error.InvalidExpertGeometry;
    const io = std.Io.Threaded.global_single_threaded.io();
    const Plan = struct {
        offset: u64,
        len: u64,
        scale: bool,
    };
    var plans: std.ArrayList(Plan) = .empty;
    defer plans.deinit(allocator);
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(allocator);
    try header.append(allocator, '{');
    try index.appendSlice(allocator, "{\"weight_map\":{");

    var payload_len: u64 = 0;
    var key_buf: [192]u8 = undefined;
    var entry_count: usize = 0;
    for (0..experts) |expert_usize| {
        const expert: u16 = @intCast(expert_usize);
        for ([_]Projection{ .down, .gate, .up }) |projection| {
            const in_dim: u64 = if (projection == .down) inter else hidden;
            const rows: u64 = if (projection == .down) hidden else inter;
            for ([_]Part{ .weight, .scales }) |part| {
                const key = try mxfp4IndividualTensorKey(&key_buf, 1, expert, projection, part);
                const cols = if (part == .weight) in_dim / 2 else in_dim / 32;
                const len = std.math.mul(u64, rows, cols) catch return error.InvalidExpertGeometry;
                const sep: []const u8 = if (entry_count == 0) "" else ",";
                const entry = try std.fmt.allocPrint(
                    allocator,
                    "{s}\"{s}\":{{\"dtype\":\"U8\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}",
                    .{ sep, key, rows, cols, payload_len, payload_len + len },
                );
                defer allocator.free(entry);
                try header.appendSlice(allocator, entry);
                const mapping = try std.fmt.allocPrint(allocator, "{s}\"{s}\":\"mxfp4-individual.safetensors\"", .{ sep, key });
                defer allocator.free(mapping);
                try index.appendSlice(allocator, mapping);
                try plans.append(allocator, .{ .offset = payload_len, .len = len, .scale = part == .scales });
                payload_len = std.math.add(u64, payload_len, len) catch return error.InvalidExpertGeometry;
                entry_count += 1;
            }
        }
    }
    try header.append(allocator, '}');
    try index.appendSlice(allocator, "}}");

    const payload_size: usize = std.math.cast(usize, payload_len) orelse return error.InvalidExpertGeometry;
    const total_size = std.math.add(usize, 8 + header.items.len, payload_size) catch return error.InvalidExpertGeometry;
    const file_bytes = try allocator.alloc(u8, total_size);
    errdefer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.items.len, .little);
    @memcpy(file_bytes[8..][0..header.items.len], header.items);
    const payload = file_bytes[8 + header.items.len ..];
    for (plans.items, 0..) |plan, i| {
        const out = payload[@intCast(plan.offset)..][0..@intCast(plan.len)];
        for (out, 0..) |*byte, j| {
            const salt: usize = if (plan.scale) 0xA7 else 0x31;
            byte.* = @truncate(j * 17 + i * 29 + salt);
        }
    }
    try dir.writeFile(io, .{ .sub_path = "mxfp4-individual.safetensors", .data = file_bytes });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    return file_bytes;
}

test "quantized expert geometry solves bits and group size from packed shapes" {
    const t = std.testing;
    try t.expectEqual(QuantGeom{ .bits = 4, .group_size = 64 }, affineGeomFromShapes(320, 40, 2560).?);
    try t.expectEqual(QuantGeom{ .bits = 4, .group_size = 64 }, affineGeomFromShapes(80, 10, 640).?);
    try t.expectEqual(QuantGeom{ .bits = 8, .group_size = 64 }, affineGeomFromShapes(640, 40, 2560).?);
    try t.expect(affineGeomFromShapes(0, 40, 2560) == null);
    try t.expect(affineGeomFromShapes(320, 40, 0) == null);
    try t.expect(affineGeomFromShapes(321, 40, 2560) == null);
}

test "routed expert layout is read off the weight map" {
    const t = std.testing;
    const fused =
        \\{"weight_map":{"model.language_model.layers.0.mlp.experts.gate_up_proj":"a","model.language_model.layers.0.mlp.experts.down_proj":"a"}}
    ;
    const split =
        \\{"weight_map":{"language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight":"a","language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales":"a","language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases":"a","language_model.model.layers.0.mlp.switch_mlp.up_proj.weight":"a","language_model.model.layers.0.mlp.switch_mlp.up_proj.scales":"a","language_model.model.layers.0.mlp.switch_mlp.up_proj.biases":"a","language_model.model.layers.0.mlp.switch_mlp.down_proj.weight":"a","language_model.model.layers.0.mlp.switch_mlp.down_proj.scales":"a","language_model.model.layers.0.mlp.switch_mlp.down_proj.biases":"a"}}
    ;
    const dense =
        \\{"weight_map":{"language_model.model.layers.0.mlp.gate_proj.weight":"a"}}
    ;
    try t.expectEqual(Layout.bf16_fused, layoutFromIndexJson(t.allocator, "qwen4_exp", fused, 1).?);
    try t.expectEqual(Layout.quantized_split, layoutFromIndexJson(t.allocator, "qwen4_exp", split, 1).?);
    try t.expect(layoutFromIndexJson(t.allocator, "qwen4_exp", dense, 1) == null);
    try t.expect(layoutFromIndexJson(t.allocator, "qwen4_exp", split, 2) == null);
    try t.expect(isRoutedExpertKey(.quantized_split, "language_model.model.layers.3.mlp.switch_mlp.down_proj.scales"));
    try t.expect(!isRoutedExpertKey(.quantized_split, "language_model.model.layers.3.mlp.shared_expert.down_proj.scales"));
    try t.expect(isRoutedExpertKey(.bf16_fused, "model.language_model.layers.3.mlp.experts.down_proj"));
    const exl3 =
        \\{"weight_map":{"language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis":"a","language_model.model.layers.0.mlp.switch_mlp.gate_proj.suh":"a","language_model.model.layers.0.mlp.switch_mlp.gate_proj.svh":"a","language_model.model.layers.0.mlp.switch_mlp.up_proj.trellis":"a","language_model.model.layers.0.mlp.switch_mlp.up_proj.suh":"a","language_model.model.layers.0.mlp.switch_mlp.up_proj.svh":"a","language_model.model.layers.0.mlp.switch_mlp.down_proj.trellis":"a","language_model.model.layers.0.mlp.switch_mlp.down_proj.suh":"a","language_model.model.layers.0.mlp.switch_mlp.down_proj.svh":"a"}}
    ;
    try t.expectEqual(Layout.exl3_k4, layoutFromIndexJson(t.allocator, "qwen4_exp", exl3, 1).?);
    try t.expect(isRoutedExpertKey(.exl3_k4, "language_model.model.layers.3.mlp.switch_mlp.gate_proj.trellis"));
    try t.expect(isRoutedExpertKey(.exl3_k4, "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.suh"));
    try t.expect(!isRoutedExpertKey(.exl3_k4, "language_model.model.layers.3.mlp.shared_expert.down_proj.weight"));
}

test "exl3 top-k above reduce-bank is a named refusal" {
    const t = std.testing;
    try admitExl3TopK(10);
    try admitExl3TopK(16);
    try admitExl3TopK(32);
    try t.expectError(error.Exl3TopKExceedsReduceBank, admitExl3TopK(33));
}

test "exl3 expert_quant admits integer and fractional K under every served codebook and refuses the rest" {
    const t = std.testing;
    const cases = [_]struct { text: []const u8, n: u32 }{
        .{ .text = "2", .n = 32 },
        .{ .text = "2.5", .n = 40 },
        .{ .text = "2.75", .n = 44 },
        .{ .text = "3", .n = 48 },
        .{ .text = "3.5", .n = 56 },
        .{ .text = "4", .n = 64 },
    };
    for (cases) |c| {
        for ([_]expert_exl3.Codebook{ .mul1, .tiny, .mcg }) |cb| {
            var buf: [96]u8 = undefined;
            const raw = try std.fmt.bufPrint(&buf, "{{\"expert_quant\":{{\"format\":\"exl3\",\"k\":{s},\"codebook\":\"{s}\"}}}}", .{ c.text, @tagName(cb) });
            const ok = try std.json.parseFromSlice(std.json.Value, t.allocator, raw, .{});
            defer ok.deinit();
            const spec = try parseExpertQuant(ok.value.object);
            try t.expectEqual(c.n, spec.rate.n);
            try t.expectEqual(cb, spec.codebook);
        }
    }
    // 2.3 is not a multiple of 1/16; 4.5 and 6 are off the served range.
    for ([_][]const u8{ "2.3", "4.5", "6", "1", "2.0625" }) |bad| {
        var buf: [96]u8 = undefined;
        const raw = try std.fmt.bufPrint(&buf, "{{\"expert_quant\":{{\"format\":\"exl3\",\"k\":{s},\"codebook\":\"mul1\"}}}}", .{bad});
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, raw, .{});
        defer parsed.deinit();
        try t.expectError(error.ExpertLayoutUnsupported, parseExpertQuant(parsed.value.object));
    }
    const bad_cb = try std.json.parseFromSlice(std.json.Value, t.allocator,
        \\{"expert_quant":{"format":"exl3","k":4,"codebook":"mul2"}}
    , .{});
    defer bad_cb.deinit();
    try t.expectError(error.ExpertLayoutUnsupported, parseExpertQuant(bad_cb.value.object));
    const missing = try std.json.parseFromSlice(std.json.Value, t.allocator, "{}", .{});
    defer missing.deinit();
    try t.expectError(error.ExpertLayoutUnsupported, parseExpertQuant(missing.value.object));
}

test "exl3 kFromPackedDim maps last dim to n" {
    const t = std.testing;
    try t.expectEqual(@as(u32, 32), kFromPackedDim(32).?.n);
    try t.expectEqual(@as(u32, 40), kFromPackedDim(40).?.n);
    try t.expectEqual(@as(u32, 48), kFromPackedDim(48).?.n);
    try t.expectEqual(@as(u32, 64), kFromPackedDim(64).?.n);
    try t.expect(kFromPackedDim(80) == null);
    try t.expect(kFromPackedDim(16) == null);
    try t.expect(kFromPackedDim(41) == null);
}

test "exl3 mixed last dims map to per-tensor n including a fractional rate" {
    const t = std.testing;
    const last = [_]u64{ 48, 32, 64, 48, 40, 48 };
    var counts: [65]u32 = @splat(0);
    for (last) |d| {
        const rate = kFromPackedDim(d) orelse return error.TestUnexpectedResult;
        counts[rate.n] += 1;
    }
    try t.expectEqual(@as(u32, 1), counts[32]);
    try t.expectEqual(@as(u32, 1), counts[40]);
    try t.expectEqual(@as(u32, 3), counts[48]);
    try t.expectEqual(@as(u32, 1), counts[64]);
}

test "layout resolution is qwen4_exp only: the same index declares nothing for another arch" {
    const t = std.testing;
    const fused =
        \\{"weight_map":{"model.language_model.layers.0.mlp.experts.gate_up_proj":"a","model.language_model.layers.0.mlp.experts.down_proj":"a"}}
    ;
    const split =
        \\{"weight_map":{"language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight":"a","language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales":"a","language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases":"a","language_model.model.layers.0.mlp.switch_mlp.up_proj.weight":"a","language_model.model.layers.0.mlp.switch_mlp.up_proj.scales":"a","language_model.model.layers.0.mlp.switch_mlp.up_proj.biases":"a","language_model.model.layers.0.mlp.switch_mlp.down_proj.weight":"a","language_model.model.layers.0.mlp.switch_mlp.down_proj.scales":"a","language_model.model.layers.0.mlp.switch_mlp.down_proj.biases":"a"}}
    ;
    try t.expect(isExpertStreamingArch("qwen4_exp"));
    for ([_][]const u8{ "qwen4_exp_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next", "llama", "deepseek_v4", "" }) |mt| {
        try t.expect(!isExpertStreamingArch(mt));
        try t.expect(layoutFromIndexJson(t.allocator, mt, split, 1) == null);
        try t.expect(layoutFromIndexJson(t.allocator, mt, fused, 1) == null);
    }
}

test "mxfp4 geometry proves native U32/U8 biasless 4-bit group-32 storage" {
    const t = std.testing;
    try t.expectEqual(
        QuantGeom{ .bits = 4, .group_size = 32 },
        mxfp4GeomFromShapes(512, 128, 4096, .u32, .u8, false).?,
    );
    try t.expect(mxfp4GeomFromShapes(512, 128, 4096, .bf16, .u8, false) == null);
    try t.expect(mxfp4GeomFromShapes(512, 128, 4096, .u32, .bf16, false) == null);
    try t.expect(mxfp4GeomFromShapes(512, 128, 4096, .u32, .u8, true) == null);
    try t.expect(mxfp4GeomFromShapes(256, 128, 4096, .u32, .u8, false) == null);
    try t.expect(mxfp4GeomFromShapes(512, 64, 4096, .u32, .u8, false) == null);
}

test "mxfp4 layout requires the complete converter bank and never claims the raw HF source" {
    const t = std.testing;
    const mxfp4_index =
        \\{"weight_map":{
        \\ "model.layers.1.mlp.switch_mlp.gate_proj.weight":"experts.safetensors",
        \\ "model.layers.1.mlp.switch_mlp.gate_proj.scales":"experts.safetensors",
        \\ "model.layers.1.mlp.switch_mlp.up_proj.weight":"experts.safetensors",
        \\ "model.layers.1.mlp.switch_mlp.up_proj.scales":"experts.safetensors",
        \\ "model.layers.1.mlp.switch_mlp.down_proj.weight":"experts.safetensors",
        \\ "model.layers.1.mlp.switch_mlp.down_proj.scales":"experts.safetensors"
        \\}}
    ;
    const missing_scale =
        \\{"weight_map":{
        \\ "model.layers.1.mlp.switch_mlp.gate_proj.weight":"experts.safetensors",
        \\ "model.layers.1.mlp.switch_mlp.gate_proj.scales":"experts.safetensors",
        \\ "model.layers.1.mlp.switch_mlp.up_proj.weight":"experts.safetensors",
        \\ "model.layers.1.mlp.switch_mlp.up_proj.scales":"experts.safetensors",
        \\ "model.layers.1.mlp.switch_mlp.down_proj.weight":"experts.safetensors"
        \\}}
    ;
    const raw_hf =
        \\{"weight_map":{
        \\ "model.layers.1.mlp.experts.0.gate_proj.weight":"experts.safetensors"
        \\}}
    ;
    try t.expectEqual(Layout.mxfp4_split, layoutFromIndexJson(t.allocator, "mimo_v2", mxfp4_index, 2).?);
    try t.expect(layoutFromIndexJson(t.allocator, "mimo_v2", missing_scale, 2) == null);
    try t.expect(layoutFromIndexJson(t.allocator, "mimo_v2", raw_hf, 2) == null);
    try t.expect(layoutFromIndexJson(t.allocator, "qwen4_exp", mxfp4_index, 2) == null);
    try t.expect(isRoutedExpertKey(.mxfp4_split, "model.layers.1.mlp.switch_mlp.down_proj.scales"));
    try t.expect(!isRoutedExpertKey(.mxfp4_split, "model.layers.1.mlp.experts.0.down_proj.weight"));
}

test "mxfp4 individual source detection requires every expert pair and rejects mixed families" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const raw_file = try writeTinyMxfp4IndividualCheckpoint(t.allocator, tmp.dir, 2, 32, 32);
    defer t.allocator.free(raw_file);
    const index = try tmp.dir.readFileAlloc(t.io, "model.safetensors.index.json", t.allocator, .limited(8 * 1024 * 1024));
    defer t.allocator.free(index);

    try t.expectEqual(
        Layout.mxfp4_individual,
        layoutFromIndexJsonWithFirstMoe(t.allocator, "mimo_v2", index, 2, 1).?,
    );
    try t.expect(layoutFromIndexJsonWithFirstMoe(t.allocator, "qwen4_exp", index, 2, 1) == null);
    try t.expect(isRoutedExpertKey(.mxfp4_individual, "model.layers.1.mlp.experts.0.gate_proj.weight"));
    try t.expect(isRoutedExpertKey(.mxfp4_individual, "model.layers.1.mlp.experts.0.gate_proj.weight_scale"));
    try t.expect(!isRoutedExpertKey(.mxfp4_individual, "model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv"));
    try t.expect(!isRoutedExpertKey(.mxfp4_individual, "model.layers.1.mlp.experts.gate_up_proj"));

    var key_buf: [192]u8 = undefined;
    const missing_key = try mxfp4IndividualTensorKey(&key_buf, 1, 0, .gate, .scales);
    const missing_needle = try std.fmt.allocPrint(
        t.allocator,
        "\"{s}\":\"mxfp4-individual.safetensors\",",
        .{missing_key},
    );
    defer t.allocator.free(missing_needle);
    const missing = try std.mem.replaceOwned(u8, t.allocator, index, missing_needle, "");
    defer t.allocator.free(missing);
    try t.expect(layoutFromIndexJsonWithFirstMoe(t.allocator, "mimo_v2", missing, 2, 1) == null);

    const bad_scale_key = "model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv";
    const bad_scale_index = try std.fmt.allocPrint(
        t.allocator,
        "{s},\"{s}\":\"mxfp4-individual.safetensors\"{s}",
        .{ index[0 .. index.len - 2], bad_scale_key, "}}" },
    );
    defer t.allocator.free(bad_scale_index);
    try t.expect(layoutFromIndexJsonWithFirstMoe(t.allocator, "mimo_v2", bad_scale_index, 2, 1) == null);

    const dense_prefix_key = "model.layers.0.mlp.experts.0.gate_proj.weight";
    const dense_prefix_index = try std.fmt.allocPrint(
        t.allocator,
        "{s},\"{s}\":\"mxfp4-individual.safetensors\"{s}",
        .{ index[0 .. index.len - 2], dense_prefix_key, "}}" },
    );
    defer t.allocator.free(dense_prefix_index);
    try t.expect(layoutFromIndexJsonWithFirstMoe(t.allocator, "mimo_v2", dense_prefix_index, 2, 1) == null);

    const mixed_key = "model.layers.1.mlp.switch_mlp.gate_proj.weight";
    const mixed_index = try std.fmt.allocPrint(
        t.allocator,
        "{s},\"{s}\":\"mxfp4-individual.safetensors\"{s}",
        .{ index[0 .. index.len - 2], mixed_key, "}}" },
    );
    defer t.allocator.free(mixed_index);
    try t.expect(layoutFromIndexJsonWithFirstMoe(t.allocator, "mimo_v2", mixed_index, 2, 1) == null);
}

test "mxfp4 individual source preserves permuted raw offsets and canonical slab geometry" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const raw_file = try writeTinyMxfp4IndividualCheckpoint(t.allocator, tmp.dir, 2, 32, 32);
    defer t.allocator.free(raw_file);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const geometry = Geometry{ .layers = 2, .experts = 2, .hidden = 32, .intermediate = 32, .first_moe_layer = 1 };
    var store = try QuantStore.openForLayout(t.allocator, path_buf[0..path_len], geometry, .mxfp4_individual);
    defer store.deinit();

    try t.expectEqual(Layout.mxfp4_individual, store.layout);
    try t.expect(!store.hasExpertLayer(0));
    try t.expect(store.hasExpertLayer(1));
    try t.expectEqual(@as(u64, 512), store.slotBytes(.gate_w));
    try t.expectEqual(@as(u64, 32), store.slotBytes(.gate_s));
    try t.expectEqual(@as(u32, 32), store.rowsOf(.gate_w));
    try t.expectEqual(@as(u32, 4), store.colsOf(.gate_w));
    try t.expectEqual(io_mod.Dtype.u32, store.dtypeOf(.gate_w));
    try t.expectEqual(io_mod.Dtype.u8, store.dtypeOf(.gate_s));
    try t.expectEqual(QuantGeom{ .bits = 4, .group_size = 32 }, store.geomOf(.gate_w));
    try t.expect(!store.componentPresent(.gate_b));
    try t.expect(store.span(1, 1, .down_w).offset < store.span(1, 1, .gate_w).offset);
    try t.expect(store.span(1, 1, .gate_w).offset < store.span(1, 1, .up_w).offset);

    const total: usize = @intCast(store.expertBytes());
    const got = try t.allocator.alloc(u8, total);
    defer t.allocator.free(got);
    try store.readExpert(1, 1, got);
    var at: usize = 0;
    for (0..component_count) |ci| {
        const c: Component = @fromBackingInt(@intCast(ci));
        if (!store.componentPresent(c)) continue;
        const span_value = store.span(1, 1, c);
        const len: usize = @intCast(span_value.len);
        const source = raw_file[@intCast(span_value.offset)..][0..len];
        try t.expectEqualSlices(u8, source, got[at..][0..len]);
        at += len;
    }
    try t.expectEqual(total, at);
}

test "mxfp4 individual source rejects missing and malformed weights or scales" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const io = t.io;
    const raw_file = try writeTinyMxfp4IndividualCheckpoint(t.allocator, tmp.dir, 2, 32, 32);
    defer t.allocator.free(raw_file);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    const model_path = path_buf[0..path_len];
    const geometry = Geometry{ .layers = 2, .experts = 2, .hidden = 32, .intermediate = 32, .first_moe_layer = 1 };
    const index = try tmp.dir.readFileAlloc(io, "model.safetensors.index.json", t.allocator, .limited(8 * 1024 * 1024));
    defer t.allocator.free(index);

    var key_buf: [192]u8 = undefined;
    const scale_key = try mxfp4IndividualTensorKey(&key_buf, 1, 0, .gate, .scales);
    const missing_needle = try std.fmt.allocPrint(t.allocator, "\"{s}\":\"mxfp4-individual.safetensors\",", .{scale_key});
    defer t.allocator.free(missing_needle);
    const missing_index = try std.mem.replaceOwned(u8, t.allocator, index, missing_needle, "");
    defer t.allocator.free(missing_index);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = missing_index });
    try t.expectError(
        error.MissingExpertTensor,
        QuantStore.openForLayout(t.allocator, model_path, geometry, .mxfp4_individual),
    );
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index });

    const weight_key = try mxfp4IndividualTensorKey(&key_buf, 1, 0, .gate, .weight);
    const old_weight = try std.fmt.allocPrint(t.allocator, "\"{s}\":{{\"dtype\":\"U8\",\"shape\":[32,16]", .{weight_key});
    defer t.allocator.free(old_weight);
    const new_weight = try std.fmt.allocPrint(t.allocator, "\"{s}\":{{\"dtype\":\"U32\",\"shape\":[32,16]", .{weight_key});
    defer t.allocator.free(new_weight);
    const malformed_weight = try std.mem.replaceOwned(u8, t.allocator, raw_file, old_weight, new_weight);
    defer t.allocator.free(malformed_weight);
    try tmp.dir.writeFile(io, .{ .sub_path = "mxfp4-individual.safetensors", .data = malformed_weight });
    try t.expectError(
        error.InvalidExpertTensor,
        QuantStore.openForLayout(t.allocator, model_path, geometry, .mxfp4_individual),
    );
    try tmp.dir.writeFile(io, .{ .sub_path = "mxfp4-individual.safetensors", .data = raw_file });

    const old_scale = try std.fmt.allocPrint(t.allocator, "\"{s}\":{{\"dtype\":\"U8\",\"shape\":[32,1]", .{scale_key});
    defer t.allocator.free(old_scale);
    const new_scale = try std.fmt.allocPrint(t.allocator, "\"{s}\":{{\"dtype\":\"U8\",\"shape\":[32,2]", .{scale_key});
    defer t.allocator.free(new_scale);
    const malformed_scale = try std.mem.replaceOwned(u8, t.allocator, raw_file, old_scale, new_scale);
    defer t.allocator.free(malformed_scale);
    try tmp.dir.writeFile(io, .{ .sub_path = "mxfp4-individual.safetensors", .data = malformed_scale });
    try t.expectError(
        error.InvalidExpertTensor,
        QuantStore.openForLayout(t.allocator, model_path, geometry, .mxfp4_individual),
    );
}

/// A synthetic MiMo index: dense layer 0, affine routed banks from layer 1 on.
fn mimoAffineIndexJson(allocator: std.mem.Allocator, layers: u16, first_moe: u16) ![]u8 {
    var b: std.ArrayList(u8) = .empty;
    errdefer b.deinit(allocator);
    try b.appendSlice(allocator, "{\"weight_map\":{\"model.layers.0.mlp.gate_proj.weight\":\"a\"");
    var layer: u16 = first_moe;
    while (layer < layers) : (layer += 1) {
        for ([_][]const u8{ "gate", "up", "down" }) |proj| {
            for ([_][]const u8{ "weight", "scales", "biases" }) |part| {
                try b.print(allocator, ",\"model.layers.{d}.mlp.switch_mlp.{s}_proj.{s}\":\"a\"", .{ layer, proj, part });
            }
        }
    }
    try b.appendSlice(allocator, "}}");
    return b.toOwnedSlice(allocator);
}

test "a mixed-width MiMo affine pack solves (bits, group_size) per layer and projection" {
    const t = std.testing;
    // hidden 4096, inter 2048: the four widths the imatrix allocator emits,
    // read off the PACKED shapes the shards carry — the config states none.
    const gate = struct {
        fn at(w_cols: u64, s_cols: u64) ?QuantGeom {
            return affineGeomFromShapes(w_cols, s_cols, 4096);
        }
    }.at;
    const down = struct {
        fn at(w_cols: u64, s_cols: u64) ?QuantGeom {
            return affineGeomFromShapes(w_cols, s_cols, 2048);
        }
    }.at;
    try t.expectEqual(QuantGeom{ .bits = 2, .group_size = 128 }, gate(256, 32).?);
    try t.expectEqual(QuantGeom{ .bits = 2, .group_size = 128 }, down(128, 16).?);
    try t.expectEqual(QuantGeom{ .bits = 3, .group_size = 128 }, gate(384, 32).?);
    try t.expectEqual(QuantGeom{ .bits = 2, .group_size = 64 }, gate(256, 64).?);
    try t.expectEqual(QuantGeom{ .bits = 4, .group_size = 64 }, gate(512, 64).?);
    try t.expectEqual(QuantGeom{ .bits = 4, .group_size = 64 }, down(256, 32).?);
}

test "mimo_v2 affine routed banks resolve as a layout under the arch's own nesting" {
    const t = std.testing;
    const index = try mimoAffineIndexJson(t.allocator, 3, 1);
    defer t.allocator.free(index);
    try t.expectEqual(Layout.quantized_split, layoutFromIndexJsonWithFirstMoe(t.allocator, "mimo_v2", index, 3, 1).?);
    try t.expect(isRoutedExpertKey(.quantized_split, "model.layers.1.mlp.switch_mlp.gate_proj.weight"));
    try t.expect(isRoutedExpertKey(.quantized_split, "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"));
    try t.expect(!isRoutedExpertKey(.quantized_split, "model.layers.1.mlp.gate.weight"));
    // A MoE layer short of its bank is not a pack, and the dense prefix stays dense.
    const short = try mimoAffineIndexJson(t.allocator, 3, 2);
    defer t.allocator.free(short);
    try t.expect(layoutFromIndexJsonWithFirstMoe(t.allocator, "mimo_v2", short, 3, 1) == null);
    // qwen4 nests under `language_model.`, so the same map is not its pack.
    try t.expect(layoutFromIndexJsonWithFirstMoe(t.allocator, "qwen4_exp", index, 3, 1) == null);
}

/// A synthetic MiMo index: dense layer 0, EXL3 routed banks from layer 1 on.
fn mimoExl3IndexJson(allocator: std.mem.Allocator, layers: u16, first_moe: u16) ![]u8 {
    var b: std.ArrayList(u8) = .empty;
    errdefer b.deinit(allocator);
    try b.appendSlice(allocator, "{\"weight_map\":{\"model.layers.0.mlp.gate_proj.weight\":\"a\"");
    var layer: u16 = first_moe;
    while (layer < layers) : (layer += 1) {
        for ([_][]const u8{ "gate", "up", "down" }) |proj| {
            for ([_][]const u8{ "trellis", "suh", "svh" }) |part| {
                try b.print(allocator, ",\"model.layers.{d}.mlp.switch_mlp.{s}_proj.{s}\":\"a\"", .{ layer, proj, part });
            }
        }
    }
    try b.appendSlice(allocator, "}}");
    return b.toOwnedSlice(allocator);
}

test "mimo_v2 EXL3 stacked experts resolve as a layout and keep the dense prefix" {
    const t = std.testing;
    const index = try mimoExl3IndexJson(t.allocator, 3, 1);
    defer t.allocator.free(index);
    try t.expectEqual(Layout.exl3_k4, layoutFromIndexJsonWithFirstMoe(t.allocator, "mimo_v2", index, 3, 1).?);
    try t.expect(isRoutedExpertKey(.exl3_k4, "model.layers.1.mlp.switch_mlp.gate_proj.trellis"));
    try t.expect(!isRoutedExpertKey(.exl3_k4, "model.layers.1.mlp.gate.weight"));
    // A MoE layer short of its bank is not a pack.
    const short = try mimoExl3IndexJson(t.allocator, 3, 2);
    defer t.allocator.free(short);
    try t.expect(layoutFromIndexJsonWithFirstMoe(t.allocator, "mimo_v2", short, 3, 1) == null);
    // The qwen4 nesting stays its own probe.
    try t.expect(layoutFromIndexJsonWithFirstMoe(t.allocator, "qwen4_exp", index, 3, 1) == null);
}

test "mimo_v2 EXL3 beside an MXFP4 bank is a mixed pack and resolves to nothing" {
    const t = std.testing;
    const index = try mimoExl3IndexJson(t.allocator, 2, 1);
    defer t.allocator.free(index);
    var key_buf: [192]u8 = undefined;
    const mxfp4 = try mxfp4TensorKey(&key_buf, 1, .gate, .weight);
    const mixed = try std.fmt.allocPrint(t.allocator, "{s},\"{s}\":\"a\"}}}}", .{ index[0 .. index.len - 2], mxfp4 });
    defer t.allocator.free(mixed);
    try t.expect(layoutFromIndexJsonWithFirstMoe(t.allocator, "mimo_v2", mixed, 2, 1) == null);
    const individual = try mxfp4IndividualTensorKey(&key_buf, 1, 0, .gate, .weight);
    const mixed2 = try std.fmt.allocPrint(t.allocator, "{s},\"{s}\":\"a\"}}}}", .{ index[0 .. index.len - 2], individual });
    defer t.allocator.free(mixed2);
    try t.expect(layoutFromIndexJsonWithFirstMoe(t.allocator, "mimo_v2", mixed2, 2, 1) == null);
}

test "mimo_v2 is the only additional expert streaming architecture" {
    const t = std.testing;
    try t.expect(isExpertStreamingArch("qwen4_exp"));
    try t.expect(isExpertStreamingArch("mimo_v2"));
    try t.expect(!isExpertStreamingArch("mimo_v2_text"));
    try t.expect(!isExpertStreamingArch("qwen3_5_moe"));
}

test "real quantized pack resolves nine regions per layer and the per expert bill" {
    const t = std.testing;
    const path = "/Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit";
    var dir = std.Io.Dir.openDirAbsolute(t.io, path, .{}) catch return error.SkipZigTest;
    defer dir.close(t.io);
    var store = try QuantStore.open(t.allocator, path, .{ .layers = 48, .experts = 512, .hidden = 2560, .intermediate = 640 });
    defer store.deinit();

    try t.expectEqual(@as(u64, 819_200), store.slotBytes(.gate_w));
    try t.expectEqual(@as(u64, 51_200), store.slotBytes(.gate_s));
    try t.expectEqual(@as(u64, 51_200), store.slotBytes(.gate_b));
    try t.expectEqual(@as(u64, 819_200), store.slotBytes(.down_w));
    try t.expectEqual(@as(u64, 51_200), store.slotBytes(.down_s));
    try t.expectEqual(@as(u64, 2_764_800), store.expertBytes());
    try t.expectEqual(QuantGeom{ .bits = 4, .group_size = 64 }, store.geomOf(.gate_w));
    try t.expectEqual(QuantGeom{ .bits = 4, .group_size = 64 }, store.geomOf(.down_w));
    try t.expectEqual(@as(u32, 640), store.rowsOf(.gate_w));
    try t.expectEqual(@as(u32, 320), store.colsOf(.gate_w));
    try t.expectEqual(@as(u32, 2560), store.rowsOf(.down_w));
    try t.expectEqual(@as(u32, 80), store.colsOf(.down_w));

    // The direct arm re-derives every offset from the shard header, so a wrong
    // stride or a swapped component fails here instead of agreeing with itself.
    const layer: u16 = 3;
    const expert: u16 = 129;
    const total: usize = @intCast(store.expertBytes());
    const through = try t.allocator.alloc(u8, total);
    defer t.allocator.free(through);
    const direct = try t.allocator.alloc(u8, total);
    defer t.allocator.free(direct);
    try store.readExpert(layer, expert, through);
    const index_raw = try dir.readFileAlloc(t.io, "model.safetensors.index.json", t.allocator, .limited(64 * 1024 * 1024));
    defer t.allocator.free(index_raw);
    const index_parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, index_raw, .{});
    defer index_parsed.deinit();
    const weight_map = index_parsed.value.object.get("weight_map").?.object;
    var at: usize = 0;
    for (0..component_count) |ci| {
        const c: Component = @fromBackingInt(@intCast(ci));
        var key_buf: [192]u8 = undefined;
        const key = try tensorKey(&key_buf, layer, c);
        const shard = try std.fmt.allocPrintSentinel(t.allocator, "{s}/{s}", .{ path, weight_map.get(key).?.string }, 0);
        defer t.allocator.free(shard);
        const fd = try io_mod.openHinted(shard, .{});
        defer _ = std.c.close(fd);
        const region = try io_mod.tensorRegion(t.allocator, fd, key);
        const per = region.tensor_bytes / region.shape[0];
        const offset = region.data_offset + region.tensor_offset + @as(u64, expert) * per;
        try t.expectEqual(per, store.slotBytes(c));
        try t.expectEqual(offset, store.span(layer, expert, c).offset);
        const len: usize = @intCast(per);
        try io_mod.readExact(fd, direct[at..][0..len], offset);
        at += len;
    }
    try t.expectEqual(total, at);
    try t.expectEqualSlices(u8, direct, through);

    // A collapsed stride or a duplicated component would still agree above.
    const neighbour = try t.allocator.alloc(u8, total);
    defer t.allocator.free(neighbour);
    try store.readExpert(layer, expert + 1, neighbour);
    try t.expect(!std.mem.eql(u8, through, neighbour));
    const gate_len: usize = @intCast(store.slotBytes(.gate_w));
    const up_at: usize = @intCast(store.slotBytes(.gate_w) + store.slotBytes(.gate_s) + store.slotBytes(.gate_b));
    try t.expect(!std.mem.eql(u8, through[0..gate_len], through[up_at..][0..gate_len]));
}

test "an eight bit tensor beside four bit ones solves to its own width" {
    const t = std.testing;
    const path = "/Users/beam/llm/models/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit";
    var dir = std.Io.Dir.openDirAbsolute(t.io, path, .{}) catch return error.SkipZigTest;
    dir.close(t.io);
    const fd = try io_mod.openHinted(path ++ "/model-00051.safetensors", .{});
    defer _ = std.c.close(fd);
    const w = try io_mod.tensorRegion(t.allocator, fd, "language_model.model.layers.3.mlp.shared_expert.gate_proj.weight");
    const sc = try io_mod.tensorRegion(t.allocator, fd, "language_model.model.layers.3.mlp.shared_expert.gate_proj.scales");
    try t.expectEqual(io_mod.Dtype.u32, w.dtype);
    try t.expectEqual(io_mod.Dtype.bf16, sc.dtype);
    try t.expectEqual(QuantGeom{ .bits = 8, .group_size = 64 }, affineGeomFromShapes(w.shape[1], sc.shape[1], 2560).?);
}
