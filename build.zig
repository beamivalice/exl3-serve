const std = @import("std");
const builtin = @import("builtin");

comptime {
    // 0.17.0 isn't tagged stable yet (homebrew still ships 0.16.0) — a nightly
    // build from ziglang.org/download is required until it is. 0.16.0's
    // bundled libc++ fails to compile against the macOS 27 beta SDK
    // (`use of undeclared identifier 'INFINITY'` in its vendored <random>);
    // fixed upstream by 0.17.0-dev, which is why the floor moved.
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 17) {
        @compileError(std.fmt.comptimePrint(
            "sushi requires Zig 0.17 (nightly until 0.17.0 stable ships) (have {d}.{d}.{d}). Grab a nightly from https://ziglang.org/download/.",
            .{ builtin.zig_version.major, builtin.zig_version.minor, builtin.zig_version.patch },
        ));
    }
}

pub fn build(b: *std.Build) void {
    // Pin LC_BUILD_VERSION minos to macOS 26.2 — the honest floor: the linked
    // libmlx is built at deployment target 26.2 (NAX kernels, scripts/
    // build-mlx.sh), so on older macOS the binary can't run anyway; failing at
    // the binary with a clear dyld version error beats "loading" and dying on
    // the dylib. Guard: tests/test_mlx_staged_nax.sh (binary minos check).
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .os_version_min = .{ .semver = .{ .major = 26, .minor = 2, .patch = 0 } },
        },
    });
    const optimize = b.standardOptimizeOption(.{});

    // Setting any non-default target field disables Zig's native macOS SDK detection,
    // so we resolve the SDK path ourselves and surface its frameworks dir.
    const macos_sdk_frameworks: ?[]const u8 = blk: {
        if (target.result.os.tag != .macos) break :blk null;
        var code: u8 = undefined;
        const stdout = b.runAllowFail(
            &.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" },
            &code,
            .inherit,
        ) catch break :blk null;
        const sdk = std.mem.trim(u8, stdout, " \n\r\t");
        if (sdk.len == 0) break :blk null;
        break :blk b.fmt("{s}/System/Library/Frameworks", .{sdk});
    };

    if (target.result.os.tag == .macos) {
        verifyBrewDeps(b);
        verifyMlxStage(b);
    }

    if (builtin.os.tag != .macos) return;

    // Version: SemVer, build.zig.zon's `.version` unless the release workflow
    // passes the tag's version (release.sh checks the two agree).
    const version = b.option([]const u8, "version", "SemVer version string (default: build.zig.zon)") orelse @import("build.zig.zon").version;
    _ = std.SemanticVersion.parse(version) catch {
        std.debug.print("[sushi] -Dversion={s} is not a SemVer version (MAJOR.MINOR.PATCH[-pre])\n", .{version});
        std.process.exit(1);
    };

    // Engine-version pins surfaced by `sushi --version` (the macOS app spawns
    // it and parses the output — see src/version.zig). These are the versions
    // that have NO runtime query API (MLX reports itself at runtime):
    //   --mlx-c-version  pinned mlx-c submodule version; defaults from the
    //                    lib/mlx/.version stamp (written by scripts/build-mlx.sh)
    // The mlx submodule commit rides along for `sushi --guest-manifest`.
    const mlx_c_version = b.option([]const u8, "mlx-c-version", "Pinned mlx-c version") orelse readMlxPin(b, "mlxc=") orelse "unknown";
    const mlx_sha = readMlxPin(b, "mlx=") orelse "";

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", version);
    build_options.addOption([]const u8, "mlx_c_version", mlx_c_version);
    build_options.addOption([]const u8, "mlx_sha", mlx_sha);
    const git_sha = b.option([]const u8, "git-sha", "Engine build id for the round-cost table: a release sha stands for the executable bytes, which are then not hashed; the MLX dylib and metallib fingerprints are always mixed in") orelse "";
    build_options.addOption([]const u8, "git_sha", git_sha);

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "jinja_c", .module = addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize) },
            .{ .name = "stb", .module = addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize) },
            .{ .name = "webp", .module = addCHeaderModule(b, .{ .cwd_relative = "/opt/homebrew/include/webp/decode.h" }, .{ .cwd_relative = "/opt/homebrew/include" }, target, optimize) },
        },
    });

    // Jinja2 template engine (wangzhaode/jinja.cpp + nlohmann/json; see NOTICE).
    // Pre-compiled as a static library with system clang++ (C++17 requires system libc++).
    // Rebuild with: cd lib/jinja_cpp && for f in jinja_wrapper caps lexer parser runtime jinja_string value; do clang++ -std=c++17 -O2 -DNDEBUG -I . -c $f.cpp -o obj/$f.o; done && ar rcs libjinja.a obj/*.o
    mod.addObjectFile(b.path("lib/jinja_cpp/libjinja.a"));
    mod.addIncludePath(b.path("lib/jinja_cpp"));

    // stb_image for JPEG/PNG decoding in the vision pipeline
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    mod.addIncludePath(b.path("lib"));

    // ANE prefill-MLP offload (perf-plan-aug-17 P5): objc bridge to the
    // private AppleNeuralEngine framework (dlopen'd at runtime — the probe
    // returns unavailable on machines/OSes without it) + the per-layer MLP
    // MIL program builder. See lib/ane/ + src/ane.zig; provenance in NOTICE.
    addAneSources(b, mod);

    // mlx + mlx-c: self-built from the pinned submodules (lib/mlx-src,
    // lib/mlxc-src) into lib/mlx by scripts/build-mlx.sh, with NAX kernels
    // enabled (the Homebrew bottle ships without them). MUST come before the
    // /opt/homebrew lib path so a leftover brew mlx-c can never win the link.
    addMlxLib(b, mod);
    // webp include/lib paths (homebrew)
    mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    mod.linkSystemLibrary("webp", .{});

    if (macos_sdk_frameworks) |fw_path| {
        mod.addFrameworkPath(.{ .cwd_relative = fw_path });
    }
    mod.linkFramework("IOKit", .{});
    mod.linkFramework("CoreFoundation", .{});
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("Metal", .{});
    mod.linkFramework("IOSurface", .{});

    const exe = b.addExecutable(.{
        .name = "sushi",
        .root_module = mod,
    });

    // Ensure Mach-O header has room for install_name_tool path changes — the
    // release tarball rewires @rpath/libmlxc.dylib to @executable_path.
    exe.headerpad_max_install_names = true;

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    const run_step = b.step("run", "Run sushi");
    run_step.dependOn(&run_cmd.step);

    // Unit tests — reuses the same module config (mlx-c, jinja_cpp, etc.)
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "jinja_c", .module = addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize) },
            .{ .name = "stb", .module = addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize) },
            .{ .name = "webp", .module = addCHeaderModule(b, .{ .cwd_relative = "/opt/homebrew/include/webp/decode.h" }, .{ .cwd_relative = "/opt/homebrew/include" }, target, optimize) },
        },
    });

    test_mod.addObjectFile(b.path("lib/jinja_cpp/libjinja.a"));
    test_mod.addIncludePath(b.path("lib/jinja_cpp"));
    test_mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    test_mod.addIncludePath(b.path("lib"));
    addAneSources(b, test_mod);
    test_mod.linkSystemLibrary("c++", .{});
    addMlxLib(b, test_mod);
    test_mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    test_mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    test_mod.linkSystemLibrary("webp", .{});

    if (macos_sdk_frameworks) |fw_path| {
        test_mod.addFrameworkPath(.{ .cwd_relative = fw_path });
    }
    test_mod.linkFramework("IOKit", .{});
    test_mod.linkFramework("CoreFoundation", .{});
    test_mod.linkFramework("Foundation", .{});
    test_mod.linkFramework("Metal", .{});
    test_mod.linkFramework("IOSurface", .{});

    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this substring");
    const qwen_preprocess_fixture = b.option(
        []const u8,
        "qwen-preprocess-fixture",
        "CPU reference fixture for the gated Qwen preprocessing parity test",
    );
    const unit_tests = b.addTest(.{
        .root_module = test_mod,
        .filters = if (test_filter) |f| &.{f} else &.{},
    });

    const test_build = b.step("test-build", "Compile unit tests without running them");
    test_build.dependOn(&b.addInstallArtifact(unit_tests, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);

    const run_unit_tests = b.addRunArtifact(unit_tests);
    if (qwen_preprocess_fixture) |fixture| {
        run_unit_tests.setEnvironmentVariable("QWEN_PREPROCESS_FIXTURE", fixture);
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/manifest.json", .{fixture}) });
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/source_rgb.bin", .{fixture}) });
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/pixel_values.bin", .{fixture}) });
    }
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}

/// Translates a single C header into an importable module (`@import("name")`
/// at the call site) via `addTranslateC`, replacing an inline `@cImport` —
/// removed as a language builtin in 0.17.0-dev.
fn addCHeaderModule(
    b: *std.Build,
    header_path: std.Build.LazyPath,
    include_dir: std.Build.LazyPath,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const translate = b.addTranslateC(.{
        .root_source_file = header_path,
        .target = target,
        .optimize = optimize,
    });
    translate.addIncludePath(include_dir);
    return translate.createModule();
}

/// ANE prefill offload sources (lib/ane): the private-framework bridge and
/// the per-layer MLP program builder, both ARC objc. Runtime-probed —
/// compiling them in costs nothing on machines without the framework.
fn addAneSources(b: *std.Build, module: *std.Build.Module) void {
    const objc_flags = &[_][]const u8{
        "-O3",
        "-fobjc-arc",
        "-Wno-deprecated-declarations",
    };
    module.addCSourceFile(.{ .file = b.path("lib/ane/ane_bridge.m"), .flags = objc_flags });
    module.addCSourceFile(.{ .file = b.path("lib/ane/ane_mlp.m"), .flags = objc_flags });
    module.addIncludePath(b.path("lib/ane"));
}

fn buildRootHandle(b: *std.Build) std.Io.Dir {
    return b.root.root_dir.handle;
}

/// Link the self-built mlx + mlx-c staged in lib/mlx by scripts/build-mlx.sh
/// (pinned submodules lib/mlx-src + lib/mlxc-src, deployment target 26.2 so
/// MLX's NAX kernels are compiled in — the Homebrew bottle ships without them
/// and hard-wires is_nax_available() false even on M5). Install names are
/// @rpath/...; the build-tree rpath resolves them in dev, release.yml
/// rewrites them to @executable_path and re-signs for the release tarball.
/// Guard test: tests/test_mlx_staged_nax.sh.
fn addMlxLib(b: *std.Build, module: *std.Build.Module) void {
    module.addIncludePath(b.path("lib/mlx/include"));
    module.addLibraryPath(b.path("lib/mlx/lib"));
    // use_pkg_config = .no: a leftover Homebrew mlx-c ships an mlx-c.pc that
    // would otherwise hijack this link — we want exactly the staged
    // NAX-enabled pair.
    module.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
    // @loader_path resolves against the BINARY's own location at launch, not
    // the launching process's cwd, so this stays correct from any launch cwd
    // and survives copying the whole zig-out + lib tree elsewhere. Two entries
    // because the installed exe (zig-out/bin/) and the `zig build test` binary
    // (.zig-cache/o/<hash>/) sit at different depths under the build root; dyld
    // tries every LC_RPATH in order and skips the one that does not resolve.
    module.addRPath(.{ .cwd_relative = "@loader_path/../../lib/mlx/lib" });
    module.addRPath(.{ .cwd_relative = "@loader_path/../../../lib/mlx/lib" });
}

/// Configure-time check that scripts/build-mlx.sh has staged the pinned
/// mlx/mlx-c build. Mirrors verifyBrewDeps: fail loudly with the fix, never
/// let the linker produce a confusing -lmlxc error (or silently pick up a
/// leftover brew copy from /opt/homebrew/lib).
fn verifyMlxStage(b: *std.Build) void {
    const stage_ok = blk: {
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/libmlxc.dylib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/mlx.metallib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/.version", .{}) catch break :blk false;
        break :blk true;
    };
    if (!stage_ok) {
        std.debug.print(
            "\n[sushi] lib/mlx is not staged (self-built mlx + mlx-c). Run:\n" ++
                "  git submodule update --init lib/mlx-src lib/mlxc-src && ./scripts/build-mlx.sh\n\n",
            .{},
        );
        std.process.exit(1);
    }
}

/// A pinned revision (`key` "mlx=" or "mlxc=") from lib/mlx/.version, written
/// by scripts/build-mlx.sh as "mlx=<sha> mlxc=<sha> target=<ver>". Returns
/// null when not staged yet.
fn readMlxPin(b: *std.Build, key: []const u8) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "lib/mlx/.version",
        b.allocator,
        .limited(256),
    ) catch return null;
    var it = std.mem.tokenizeScalar(u8, std.mem.trim(u8, bytes, " \t\r\n"), ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, key)) return b.dupe(tok[key.len..]);
    }
    return null;
}

const BrewDep = struct { name: []const u8, min: std.SemanticVersion };

const required_brew_deps = [_]BrewDep{
    // mlx + mlx-c are NOT brew deps anymore: they are pinned submodules built
    // by scripts/build-mlx.sh (see addMlxLib) so the NAX kernels ship enabled.
    .{ .name = "webp", .min = .{ .major = 1, .minor = 6, .patch = 0 } },
};

fn verifyBrewDeps(b: *std.Build) void {
    for (required_brew_deps) |dep| {
        var code: u8 = undefined;
        const stdout = b.runAllowFail(
            &.{ "brew", "list", "--versions", dep.name },
            &code,
            .inherit,
        ) catch {
            std.debug.print(
                "\n[sushi] missing Homebrew dependency '{s}' (>= {d}.{d}.{d}). Install with: brew install webp\n\n",
                .{ dep.name, dep.min.major, dep.min.minor, dep.min.patch },
            );
            std.process.exit(1);
        };
        const trimmed = std.mem.trim(u8, stdout, " \n\r\t");
        const space = std.mem.indexOfScalar(u8, trimmed, ' ') orelse {
            std.debug.print("[sushi] cannot parse `brew list --versions {s}` output: {s}\n", .{ dep.name, trimmed });
            std.process.exit(1);
        };
        var ver_str = trimmed[space + 1 ..];
        // Strip Homebrew revision suffix (e.g., "0.6.0_2" -> "0.6.0").
        if (std.mem.indexOfScalar(u8, ver_str, '_')) |us| ver_str = ver_str[0..us];
        const have = std.SemanticVersion.parse(ver_str) catch {
            std.debug.print("[sushi] cannot parse '{s}' version '{s}'\n", .{ dep.name, ver_str });
            std.process.exit(1);
        };
        if (have.order(dep.min) == .lt) {
            std.debug.print(
                "\n[sushi] Homebrew '{s}' is {d}.{d}.{d}; need >= {d}.{d}.{d}. Run: brew upgrade {s}\n\n",
                .{ dep.name, have.major, have.minor, have.patch, dep.min.major, dep.min.minor, dep.min.patch, dep.name },
            );
            std.process.exit(1);
        }
    }
}
