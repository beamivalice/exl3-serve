//! Per-model settings (`~/.sushi/model-settings.json`): context size, KV
//! quant and MTP that follow the MODEL, applied at every load construction
//! site. Keyed by the model's absolute path (dir, or the `.gguf` file).
//! The app edits the file; the server owns applying it. A malformed file is
//! logged and treated as empty: a settings typo must never stop a load.
const std = @import("std");
const kv_quant = @import("kv_quant.zig");
const log = @import("log");
const mtp_acceptance = @import("mtp_acceptance.zig");

/// Where a load's value for one key came from. A request's own field outranks all three.
pub const Source = enum { flag, model_settings, default };

pub fn Pick(comptime T: type) type {
    return struct { value: T, source: Source };
}

/// An explicit launch flag outranks the file, which outranks the default. `flag` null = not given.
pub fn pick(comptime T: type, flag: ?T, setting: ?T, default: T) Pick(T) {
    if (flag) |f| return .{ .value = f, .source = .flag };
    if (setting) |s| return .{ .value = s, .source = .model_settings };
    return .{ .value = default, .source = .default };
}

pub fn sourceLabel(source: Source, flag_name: []const u8) []const u8 {
    return switch (source) {
        .flag => flag_name,
        .model_settings => "model-settings.json",
        .default => "default",
    };
}

/// A launch value counts as a flag only when the operator passed it.
pub fn launchFlag(comptime T: type, value: T, explicit: bool) ?T {
    return if (explicit) value else null;
}

/// The manual context: `--ctx-size` > `ctx_size` > auto. 0 = auto, i.e. not given, on both sides.
pub fn contextPick(flag: u32, setting: u32) Pick(u32) {
    return pick(u32, if (flag > 0) flag else null, if (setting > 0) setting else null, 0);
}

/// Whether a load arms the MTP head, and whether requests then default to it on any arch.
pub const MtpChoice = struct {
    on: bool,
    source: Source,

    pub fn resolve(flag: ?bool, setting: ?bool, default: bool) MtpChoice {
        const p = pick(bool, flag, setting, default);
        return .{ .on = p.value, .source = p.source };
    }

    /// The engine default arms the head but leaves the request default to the arch
    /// (`server.defaultEnableMtp`); an operator's `on` forces it for MoE targets too.
    pub fn forced(self: MtpChoice) bool {
        return self.on and self.source != .default;
    }

    pub fn label(self: MtpChoice) []const u8 {
        return if (self.on) "on" else "off";
    }

    pub fn sourceName(self: MtpChoice) []const u8 {
        return sourceLabel(self.source, if (self.on) "--mtp" else "--no-mtp");
    }
};

/// Log token for a manual context; 0 is auto.
pub fn contextLabel(buf: []u8, ctx: u32) []const u8 {
    if (ctx == 0) return "auto";
    return std.fmt.bufPrint(buf, "{d}", .{ctx}) catch "auto";
}

/// The flag an acceptance mode would have been launched with.
pub fn acceptanceFlagName(mode: mtp_acceptance.Mode) []const u8 {
    return switch (mode) {
        .exact => "default",
        .typical => "--mtp-typical",
        .tokenv3 => "--mtp-tokenv3",
    };
}

pub const Override = struct {
    ctx_size: ?u32 = null,
    kv_quant: ?kv_quant.KVQuantConfig = null,
    mtp: ?bool = null,
    mtp_acceptance: ?mtp_acceptance.Mode = null,
    ssd_budget_gb: ?u32 = null,

    pub fn isEmpty(o: Override) bool {
        return o.ctx_size == null and o.kv_quant == null and o.mtp == null and
            o.mtp_acceptance == null and o.ssd_budget_gb == null;
    }
};

pub const Settings = struct {
    parsed: ?std.json.Parsed(std.json.Value) = null,

    pub fn deinit(self: *Settings) void {
        if (self.parsed) |*p| p.deinit();
        self.parsed = null;
    }

    pub fn lookup(self: *const Settings, model_path: []const u8) Override {
        const p = self.parsed orelse return .{};
        const root = switch (p.value) {
            .object => |o| o,
            else => return .{},
        };
        const want = trimSlash(model_path);
        var it = root.iterator();
        while (it.next()) |kv| {
            if (!std.mem.eql(u8, trimSlash(kv.key_ptr.*), want)) continue;
            return fromValue(kv.value_ptr.*);
        }
        return .{};
    }
};

fn trimSlash(p: []const u8) []const u8 {
    var s = p;
    while (s.len > 1 and s[s.len - 1] == '/') s = s[0 .. s.len - 1];
    return s;
}

fn fromValue(v: std.json.Value) Override {
    const obj = switch (v) {
        .object => |o| o,
        else => return .{},
    };
    var o: Override = .{};
    if (obj.get("ctx_size")) |c| switch (c) {
        .integer => |i| if (i > 0 and i <= std.math.maxInt(u32)) {
            o.ctx_size = @intCast(i);
        },
        else => {},
    };
    if (obj.get("kv_quant")) |k| o.kv_quant = kv_quant.KVQuantConfig.fromJsonValue(k);
    if (obj.get("mtp")) |m| switch (m) {
        .bool => |b| o.mtp = b,
        else => {},
    };
    if (obj.get("mtp_acceptance")) |a| switch (a) {
        .string => |name| o.mtp_acceptance = mtp_acceptance.fromName(name),
        else => {},
    };
    if (obj.get("ssd_budget_gb")) |g| switch (g) {
        .integer => |i| if (i > 0 and i <= std.math.maxInt(u32)) {
            o.ssd_budget_gb = @intCast(i);
        },
        else => {},
    };
    return o;
}

pub fn parse(alloc: std.mem.Allocator, body: []const u8) !Settings {
    return .{ .parsed = try std.json.parseFromSlice(std.json.Value, alloc, body, .{}) };
}

/// Missing file = empty. Unreadable or malformed = empty, logged.
pub fn load(alloc: std.mem.Allocator, io: std.Io, path: []const u8) Settings {
    const body = std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 20)) catch |err| {
        if (err != error.FileNotFound) log.warn("[model-settings] {s}: unreadable ({s}), ignored\n", .{ path, @errorName(err) });
        return .{};
    };
    defer alloc.free(body);
    return parse(alloc, body) catch |err| {
        log.warn("[model-settings] {s}: malformed ({s}), ignored\n", .{ path, @errorName(err) });
        return .{};
    };
}

pub fn defaultPath(buf: []u8) []const u8 {
    const home = std.mem.span(std.c.getenv("HOME") orelse "/tmp");
    return std.fmt.bufPrint(buf, "{s}/.sushi/model-settings.json", .{home}) catch "";
}

/// The one call load sites make: read the default file, look the model up, log a hit.
pub fn overrideFor(alloc: std.mem.Allocator, io: std.Io, model_path: []const u8) Override {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var s = load(alloc, io, defaultPath(&buf));
    defer s.deinit();
    const o = s.lookup(model_path);
    if (!o.isEmpty()) log.info("[model-settings] {s}: ctx={d} kv={s} mtp={s} accept={s} ssd_budget_gb={d}\n", .{
        model_path,
        o.ctx_size orelse 0,
        if (o.kv_quant) |k| k.wireName() else "default",
        if (o.mtp) |m| (if (m) "on" else "off") else "default",
        if (o.mtp_acceptance) |a| mtp_acceptance.name(a) else "default",
        o.ssd_budget_gb orelse 0,
    });
    return o;
}

test "model_settings: parse + lookup with and without trailing slash" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a/": {"ctx_size": 65536, "kv_quant": "8", "mtp": false}, "/m/b": {"kv_quant": 4}}
    );
    defer s.deinit();
    const a = s.lookup("/m/a");
    try std.testing.expectEqual(@as(?u32, 65536), a.ctx_size);
    try std.testing.expectEqual(@as(u8, 8), a.kv_quant.?.bits);
    try std.testing.expectEqual(@as(?bool, false), a.mtp);
    const b = s.lookup("/m/b/");
    try std.testing.expectEqual(@as(?u32, null), b.ctx_size);
    try std.testing.expectEqual(@as(u8, 4), b.kv_quant.?.bits);
    try std.testing.expectEqual(@as(?bool, null), b.mtp);
    try std.testing.expect(s.lookup("/m/c").isEmpty());
}

test "model_settings: mtp_acceptance names a mode at its default threshold" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"mtp_acceptance": "typical"}, "/m/b": {"mtp_acceptance": "tokenv3"}, "/m/c": {"mtp_acceptance": "exact"}, "/m/d": {"mtp_acceptance": "fast"}}
    );
    defer s.deinit();
    try std.testing.expectEqual(@as(f32, 0.2), s.lookup("/m/a").mtp_acceptance.?.typical.delta);
    try std.testing.expectEqual(@as(f32, 0.95), s.lookup("/m/b").mtp_acceptance.?.tokenv3);
    try std.testing.expect(s.lookup("/m/c").mtp_acceptance.? == .exact);
    try std.testing.expect(s.lookup("/m/d").isEmpty());
}

test "model_settings: bad values ignored, bad JSON = empty" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"ctx_size": 0, "kv_quant": "16", "mtp": "yes", "future": 1}}
    );
    defer s.deinit();
    try std.testing.expect(s.lookup("/m/a").isEmpty());
    try std.testing.expectError(error.SyntaxError, parse(std.testing.allocator, "{nope"));
    var empty = load(std.testing.allocator, std.testing.io, "/nonexistent/model-settings.json");
    defer empty.deinit();
    try std.testing.expect(empty.lookup("/m/a").isEmpty());
}

test "model_settings: ssd_budget_gb rides the same file as the other keys" {
    var s = try parse(std.testing.allocator,
        \\{"/m/a": {"ssd_budget_gb": 60}, "/m/b": {"ssd_budget_gb": 0}, "/m/c": {"ssd_budget_gb": "60"}, "/m/d": {"ctx_size": 4096}}
    );
    defer s.deinit();
    try std.testing.expectEqual(@as(?u32, 60), s.lookup("/m/a").ssd_budget_gb);
    try std.testing.expect(s.lookup("/m/b").isEmpty());
    try std.testing.expect(s.lookup("/m/c").isEmpty());
    try std.testing.expectEqual(@as(?u32, null), s.lookup("/m/d").ssd_budget_gb);
    try std.testing.expect(!s.lookup("/m/a").isEmpty());
}

test "an explicit --mtp / --no-mtp outranks the per-model mtp, which outranks the default" {
    const t = std.testing;
    const no_mtp = MtpChoice.resolve(false, true, true);
    try t.expect(!no_mtp.on);
    try t.expectEqualStrings("--no-mtp", no_mtp.sourceName());
    try t.expect(!no_mtp.forced());
    const mtp = MtpChoice.resolve(true, false, true);
    try t.expect(mtp.on and mtp.forced());
    try t.expectEqualStrings("--mtp", mtp.sourceName());
    const file = MtpChoice.resolve(null, true, true);
    try t.expect(file.on and file.forced());
    try t.expectEqualStrings("model-settings.json", file.sourceName());
    const engine = MtpChoice.resolve(null, null, true);
    try t.expect(engine.on and !engine.forced());
    try t.expectEqualStrings("default", engine.sourceName());
    try t.expectEqualStrings("off", MtpChoice.resolve(null, false, true).label());
}

test "an explicit --ctx-size outranks the per-model ctx_size; 0 is auto on both sides" {
    const t = std.testing;
    const flag = contextPick(16384, 4096);
    try t.expectEqual(@as(u32, 16384), flag.value);
    try t.expectEqualStrings("--ctx-size", sourceLabel(flag.source, "--ctx-size"));
    const file = contextPick(0, 4096);
    try t.expectEqual(@as(u32, 4096), file.value);
    try t.expectEqual(Source.model_settings, file.source);
    const auto = contextPick(0, 0);
    try t.expectEqual(@as(u32, 0), auto.value);
    try t.expectEqual(Source.default, auto.source);
}

test "an explicit --mtp-typical / --mtp-tokenv3 outranks the per-model mtp_acceptance" {
    const t = std.testing;
    const Mode = mtp_acceptance.Mode;
    const typical: Mode = .{ .typical = .{ .delta = 0.3 } };
    const flag = pick(Mode, typical, .{ .tokenv3 = 0.95 }, .exact);
    try t.expectEqual(@as(f32, 0.3), flag.value.typical.delta);
    try t.expectEqualStrings("--mtp-typical", sourceLabel(flag.source, acceptanceFlagName(flag.value)));
    const file = pick(Mode, null, .{ .tokenv3 = 0.95 }, .exact);
    try t.expectEqual(@as(f32, 0.95), file.value.tokenv3);
    try t.expectEqual(Source.model_settings, file.source);
    const engine = pick(Mode, null, null, .exact);
    try t.expect(engine.value == .exact);
    try t.expectEqual(Source.default, engine.source);
}
