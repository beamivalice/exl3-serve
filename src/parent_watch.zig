//! `--parent-pid <pid>`: a host that runs sushi as a guest engine names its own pid, and sushi shuts down the way
//! SIGTERM does once that process is gone, so a crashed host cannot leave it holding GPU memory.

const std = @import("std");
const log = @import("log.zig");

const pid_t = std.posix.pid_t;

pub fn parseArg(s: []const u8) error{InvalidParentPid}!pid_t {
    const pid = std.fmt.parseInt(pid_t, s, 10) catch return error.InvalidParentPid;
    if (pid <= 1) return error.InvalidParentPid;
    return pid;
}

pub const Watch = struct {
    pid: pid_t,
    /// The watched pid was our parent when the watch began: a reparent then proves it died even after its pid is
    /// reused by another process.
    was_parent: bool,

    pub fn init(pid: pid_t) Watch {
        return .{ .pid = pid, .was_parent = std.posix.getppid() == pid };
    }

    pub fn gone(w: Watch, exists: bool, ppid_now: pid_t) bool {
        if (!exists) return true;
        return w.was_parent and ppid_now != w.pid;
    }
};

/// A process owned by another user answers EPERM, which still means it exists.
pub fn processExists(pid: pid_t) bool {
    std.posix.kill(pid, @enumFromInt(0)) catch |err| return err != error.ProcessNotFound;
    return true;
}

/// Polls once a second on its own thread. On the parent's exit it signals this whole process with SIGTERM: before
/// the serve loop installs its handler that ends a model load, after it the loop shuts down as for any SIGTERM.
pub fn start(pid: pid_t) !void {
    const t = try std.Thread.spawn(.{}, watchLoop, .{Watch.init(pid)});
    t.detach();
}

fn watchLoop(w: Watch) void {
    const tick = std.c.timespec{ .sec = 1, .nsec = 0 };
    while (!w.gone(processExists(w.pid), std.posix.getppid())) {
        _ = std.c.nanosleep(&tick, null);
    }
    log.info("[parent-pid] {d} is gone; shutting down\n", .{w.pid});
    std.posix.kill(std.c.getpid(), .TERM) catch {};
}

test "parent_watch: a pid that no longer exists is gone" {
    const w: Watch = .{ .pid = 4242, .was_parent = false };
    try std.testing.expect(w.gone(false, 1));
    try std.testing.expect(!w.gone(true, 1));
}

test "parent_watch: a watched parent that reparented us is gone even if its pid exists again" {
    const w: Watch = .{ .pid = 4242, .was_parent = true };
    try std.testing.expect(!w.gone(true, 4242));
    try std.testing.expect(w.gone(true, 1));
}

test "parent_watch: a watched non-parent is judged by existence alone" {
    const w: Watch = .{ .pid = 4242, .was_parent = false };
    try std.testing.expect(!w.gone(true, 1));
}

test "parent_watch: probes this process as alive and an impossible pid as gone" {
    try std.testing.expect(processExists(std.c.getpid()));
    try std.testing.expect(!processExists(std.math.maxInt(pid_t)));
}

test "parent_watch: the flag takes a pid above 1" {
    try std.testing.expectEqual(@as(pid_t, 4242), try parseArg("4242"));
    try std.testing.expectError(error.InvalidParentPid, parseArg("1"));
    try std.testing.expectError(error.InvalidParentPid, parseArg("0"));
    try std.testing.expectError(error.InvalidParentPid, parseArg("-7"));
    try std.testing.expectError(error.InvalidParentPid, parseArg("host"));
}
