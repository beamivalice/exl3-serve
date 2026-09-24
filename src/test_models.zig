//! Where a test that needs a real pack finds it: `$SUSHI_MODELS_DIR/<name>`, else `$HOME/.sushi/models/<name>`.
//! A test skips when the pack is absent.
const std = @import("std");

/// Absolute path of the pack directory `name`; SkipZigTest when no absolute models root is known.
pub fn packPath(buf: []u8, name: []const u8) error{SkipZigTest}![]const u8 {
    return join(buf, envSpan("SUSHI_MODELS_DIR"), envSpan("HOME"), name) orelse error.SkipZigTest;
}

fn envSpan(key: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(key)) |p| std.mem.span(p) else null;
}

fn join(buf: []u8, models_dir: ?[]const u8, home: ?[]const u8, name: []const u8) ?[]const u8 {
    if (models_dir) |root| if (root.len > 0) {
        if (!std.fs.path.isAbsolute(root)) return null;
        return std.fmt.bufPrint(buf, "{s}/{s}", .{ root, name }) catch null;
    };
    const h = home orelse return null;
    if (!std.fs.path.isAbsolute(h)) return null;
    return std.fmt.bufPrint(buf, "{s}/.sushi/models/{s}", .{ h, name }) catch null;
}

test "a pack path prefers SUSHI_MODELS_DIR, falls back to ~/.sushi/models, and refuses a relative root" {
    const t = std.testing;
    var buf: [256]u8 = undefined;
    try t.expectEqualStrings("/packs/Qwen", join(&buf, "/packs", "/Users/user", "Qwen").?);
    try t.expectEqualStrings("/Users/user/.sushi/models/Qwen", join(&buf, null, "/Users/user", "Qwen").?);
    try t.expectEqualStrings("/Users/user/.sushi/models/Qwen", join(&buf, "", "/Users/user", "Qwen").?);
    try t.expect(join(&buf, "packs", "/Users/user", "Qwen") == null);
    try t.expect(join(&buf, null, null, "Qwen") == null);
    var tiny: [8]u8 = undefined;
    try t.expect(join(&tiny, "/packs", null, "Qwen3.8-Flash-Next") == null);
}
