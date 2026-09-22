const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const exl3 = @import("expert_exl3.zig");
const io_util = @import("io_util.zig");

var ubench_mute: bool = false;
var ubench_env: ?bool = null;
var pair_splits_force: ?u32 = null;
var swiglu_maxabs_env: ?bool = null;
var swiglu_maxabs_dumped: u32 = 0;

fn diagEnvValueOn(raw: ?[*:0]const u8) bool {
    const v = raw orelse return false;
    return v[0] != 0 and v[0] != '0';
}

fn pairSplitCount() u32 {
    if (pair_splits_force) |v| return v;
    return 2;
}

/// A pair-GEMV threadgroup prepares its own slice of x, so the split must cut
/// the hidden dim on a 128-wide Hadamard block boundary.
fn pairSplitCountFor(in_dim: c_int) u32 {
    const n = pairSplitCount();
    if (@rem(in_dim, @as(c_int, @intCast(n)) * 128) != 0) return 1;
    return n;
}

pub fn setPairSplitsForTest(n: ?u32) void {
    pair_splits_force = n;
}

fn exl3UbenchOn() bool {
    if (ubench_mute) return false;
    if (ubench_env) |v| return v;
    const v = diagEnvValueOn(std.c.getenv("MLX_SERVE_EXL3_LAYER_UBENCH"));
    ubench_env = v;
    return v;
}

fn benchPrint(comptime fmt: []const u8, args: anytype) void {
    if (!exl3UbenchOn()) return;
    std.debug.print(fmt, args);
}

fn ubenchEval(a: mlx.mlx_array, name: []const u8) !void {
    if (!exl3UbenchOn()) return;
    const io = std.Io.Threaded.global_single_threaded.io();
    var sw = io_util.Stopwatch.init(io);
    try mlx.check(mlx.mlx_array_eval(a));
    const ns = sw.read();
    benchPrint("[exl3-ubench] {s} {d:.3} ms\n", .{ name, @as(f64, @floatFromInt(ns)) / 1e6 });
    log.info("[exl3-ubench] {s} {d:.3} ms\n", .{ name, @as(f64, @floatFromInt(ns)) / 1e6 });
}

fn swigluMaxabsOn() bool {
    if (swiglu_maxabs_env) |v| return v;
    const v = diagEnvValueOn(std.c.getenv("MLX_SERVE_EXL3_SWIGLU_MAXABS"));
    swiglu_maxabs_env = v;
    return v;
}

fn dumpAbsMax(s: mlx.mlx_stream, a: mlx.mlx_array, name: []const u8) !void {
    var ab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ab);
    try mlx.check(mlx.mlx_abs(&ab, a, s));
    var mx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx);
    try mlx.check(mlx.mlx_max(&mx, ab, false, s));
    try mlx.check(mlx.mlx_array_eval(mx));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, mx));
    log.info("[exl3-maxabs] {s} {d:.6}\n", .{ name, v });
}

pub const DECODE_ROWS_MAX: usize = 16;

/// The trellis rate a packed last dim carries: n halfwords per 256-weight tile,
/// K = n/16. Every kernel keys on n, never on an integer K.
fn packedRate(last: c_int) !exl3.Rate {
    if (last < 0) return error.BadExl3Shape;
    return exl3.kFromPackedDim(@intCast(last)) orelse error.BadExl3Shape;
}

pub fn usesPrefillArm(rows: usize) bool {
    return rows > DECODE_ROWS_MAX;
}

pub const UNION_EXPERTS: usize = 512;

pub fn unionUnique(eids: []const u32) u32 {
    var seen: [UNION_EXPERTS]u8 = @splat(0);
    var n: u32 = 0;
    for (eids) |e| {
        if (e >= UNION_EXPERTS) continue;
        if (seen[e] == 0) {
            seen[e] = 1;
            n += 1;
        }
    }
    return n;
}

pub fn unionMultiplicity(eids: []const u32, counts: []u32) u32 {
    var unique: u32 = 0;
    for (eids) |e| {
        if (e >= counts.len) continue;
        if (counts[e] == 0) unique += 1;
        counts[e] += 1;
    }
    return unique;
}

var union_hist_env: ?bool = null;

fn unionHistOn() bool {
    if (union_hist_env) |v| return v;
    const v = diagEnvValueOn(std.c.getenv("MLX_SERVE_EXL3_UNION_HIST"));
    union_hist_env = v;
    return v;
}

pub fn dumpUnionHist(slots_u: mlx.mlx_array, S: usize, K: usize) !void {
    if (!unionHistOn()) return;
    if (S < 2 or K == 0) return;
    // The caller swallows this error, so the latch this eval may raise is ours
    // to drop: left standing it becomes the next decode tick's `MlxFailure`.
    const had_error = mlx.errorPending();
    mlx.check(mlx.mlx_array_eval(slots_u)) catch |e| {
        mlx.dropLatchedErrorUnless(had_error);
        return e;
    };
    const n = S * K;
    const ptr = mlx.mlx_array_data_uint32(slots_u) orelse return;
    const slice = ptr[0..n];
    var counts: [UNION_EXPERTS]u32 = @splat(0);
    const unique = unionMultiplicity(slice, &counts);
    var shared: u32 = 0;
    var max_m: u32 = 0;
    for (counts) |c| {
        if (c >= 2) shared += 1;
        if (c > max_m) max_m = c;
    }
    log.info("[exl3-union] S={d} K={d} assignments={d} unique={d} shared={d} max_mult={d}\n", .{ S, K, n, unique, shared, max_m });
}

/// Rows a routed-expert call sees: the product of every leading dim, never the
/// activation width.
pub fn rowsOfShape(shape: []const c_int) usize {
    if (shape.len < 2) return 1;
    var n: usize = 1;
    for (shape[0 .. shape.len - 1]) |d| n *= @intCast(@max(d, 0));
    return n;
}

pub const RunTable = struct {
    start: []u32,
    len: []u32,
    eid: []u32,
    n: u32,
};

pub fn buildRuns(alloc: std.mem.Allocator, experts: []const u32) !RunTable {
    if (experts.len == 0) return .{ .start = &.{}, .len = &.{}, .eid = &.{}, .n = 0 };
    var n: u32 = 1;
    var i: usize = 1;
    while (i < experts.len) : (i += 1) {
        if (experts[i] != experts[i - 1]) n += 1;
    }
    const start = try alloc.alloc(u32, n);
    const len = try alloc.alloc(u32, n);
    const eid = try alloc.alloc(u32, n);
    var r: u32 = 0;
    start[0] = 0;
    eid[0] = experts[0];
    i = 1;
    while (i < experts.len) : (i += 1) {
        if (experts[i] != experts[i - 1]) {
            len[r] = @intCast(i - start[r]);
            r += 1;
            start[r] = @intCast(i);
            eid[r] = experts[i];
        }
    }
    len[r] = @intCast(experts.len - start[r]);
    return .{ .start = start, .len = len, .eid = eid, .n = n };
}

const GEMV_SOURCE: [:0]const u8 =
    \\uint gid = uint(thread_position_in_grid.x);
    \\if (gid >= uint(ODIM)) return;
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_HW = N;
    \\constexpr uint word_count = N / 2u;
    \\constexpr uint IN_TILES = uint(IDIM) / TILE;
    \\constexpr uint OUT_TILES = uint(ODIM) / TILE;
    \\const uint tn = gid / TILE;
    \\const uint local = gid % TILE;
    \\float acc = 0.0f;
    \\for (uint tk = 0u; tk < IN_TILES; tk++) {
    \\  const device ushort* tile = trellis + (tk * OUT_TILES + tn) * PACKED_HW;
    \\  ushort cw[256];
    \\  for (uint th = 0u; th < 128u; th++) {
    \\    const exl3_win wv = exl3_pair_window(th * 2u, N);
    \\    const uint a = uint(tile[wv.i0 * 2u]) | (uint(tile[wv.i0 * 2u + 1u]) << 16u);
    \\    const uint b = uint(tile[wv.i1 * 2u]) | (uint(tile[wv.i1 * 2u + 1u]) << 16u);
    \\    const ulong merged = (ulong(a) << 32) | ulong(b);
    \\    const uint funnel = uint(merged >> wv.sh);
    \\    cw[th * 2u] = ushort((funnel >> wv.fresh) & 0xffffu);
    \\    cw[th * 2u + 1u] = ushort(funnel & 0xffffu);
    \\  }
    \\  for (uint slot = 0u; slot < 256u; slot++) {
    \\    const uint lane = slot / 8u;
    \\    const uint s = slot % 8u;
    \\    const uint row0 = (lane & 3u) * 2u;
    \\    const uint col0 = lane >> 2u;
    \\    uint pos;
    \\    switch (s) {
    \\      case 0u: pos = row0 * 16u + col0; break;
    \\      case 1u: pos = (row0 + 1u) * 16u + col0; break;
    \\      case 2u: pos = (row0 + 8u) * 16u + col0; break;
    \\      case 3u: pos = (row0 + 9u) * 16u + col0; break;
    \\      case 4u: pos = row0 * 16u + col0 + 8u; break;
    \\      case 5u: pos = (row0 + 1u) * 16u + col0 + 8u; break;
    \\      case 6u: pos = (row0 + 8u) * 16u + col0 + 8u; break;
    \\      default: pos = (row0 + 9u) * 16u + col0 + 8u; break;
    \\    }
    \\    if ((pos % 16u) != local) continue;
    \\    const float w = exl3_decode1(uint(cw[slot]));
    \\    acc += float(x[tk * TILE + (pos / 16u)]) * w;
    \\  }
    \\}
    \\y[gid] = half(acc);
;

const GEMM_SORTED_SOURCE: [:0]const u8 =
    \\threadgroup float partial[2 * 256];
    \\uint ot = uint(threadgroup_position_in_grid.x);
    \\uint win = uint(threadgroup_position_in_grid.y);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\uint lane = uint(thread_index_in_simdgroup);
    \\uint lid = uint(thread_index_in_threadgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_HW = N;
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\const uint start = wstarts[win];
    \\const uint n = wnlive[win];
    \\if (n == 0u || n > uint(WIN)) return;
    \\const uint pg = sg >> 1u;
    \\const uint th = sg & 1u;
    \\const uint first = th * 128u + lane * 4u;
    \\uint pos[4];
    \\uint irow[4];
    \\for (uint s = 0u; s < 4u; s++) {
    \\  const uint src = first + s;
    \\  const uint ln = src >> 3u;
    \\  const uint sl = src & 7u;
    \\  const uint prow = (ln & 3u) * 2u;
    \\  const uint col0 = ln >> 2u;
    \\  uint p;
    \\  switch (sl) {
    \\    case 0u: p = prow * 16u + col0; break;
    \\    case 1u: p = (prow + 1u) * 16u + col0; break;
    \\    case 2u: p = (prow + 8u) * 16u + col0; break;
    \\    case 3u: p = (prow + 9u) * 16u + col0; break;
    \\    case 4u: p = prow * 16u + col0 + 8u; break;
    \\    case 5u: p = (prow + 1u) * 16u + col0 + 8u; break;
    \\    case 6u: p = (prow + 8u) * 16u + col0 + 8u; break;
    \\    default: p = (prow + 9u) * 16u + col0 + 8u; break;
    \\  }
    \\  pos[s] = p;
    \\  irow[s] = p >> 4u;
    \\}
    \\uint row = start;
    \\const uint end = start + n;
    \\while (row < end) {
    \\const uint run0 = row;
    \\const uint eid = uint(eids[row]);
    \\uint run_end = row + 1u;
    \\while (run_end < end && uint(eids[run_end]) == eid) run_end++;
    \\const uint nlive = run_end - row;
    \\float4 acc[8][4];
    \\for (uint g = 0u; g < 8u; g++) {
    \\  acc[g][0] = float4(0.0f);
    \\  acc[g][1] = float4(0.0f);
    \\  acc[g][2] = float4(0.0f);
    \\  acc[g][3] = float4(0.0f);
    \\}
    \\  for (uint tk = pg; tk < IT; tk += 2u) {
    \\    const device uint* words = (const device uint*)(trellis + ((((size_t)eid * (size_t)IT + tk) * (size_t)OT + ot) * PACKED_HW));
    \\    const exl3_win wlo_v = exl3_pair_window(first, N);
    \\    const exl3_win whi_v = exl3_pair_window(first + 2u, N);
    \\    const ulong m0 = ((ulong)words[wlo_v.i0] << 32) | (ulong)words[wlo_v.i1];
    \\    const uint f0 = uint(m0 >> wlo_v.sh);
    \\    const ulong m1 = ((ulong)words[whi_v.i0] << 32) | (ulong)words[whi_v.i1];
    \\    const uint f1 = uint(m1 >> whi_v.sh);
    \\    const uint2 lo = uint2((f0 >> wlo_v.fresh) & 0xffffu, f0 & 0xffffu);
    \\    const uint2 hi = uint2((f1 >> whi_v.fresh) & 0xffffu, f1 & 0xffffu);
    \\    const float2 wlo = exl3_decode2(lo);
    \\    const float2 whi = exl3_decode2(hi);
    \\    const float4 wt = float4(wlo.x, wlo.y, whi.x, whi.y);
    \\    const uint ib = tk * TILE;
    \\    for (uint g = 0u; g < 8u; g++) {
    \\      if (g * 4u >= nlive) break;
    \\      const uint base_r = run0 + g * 4u;
    \\      for (uint s = 0u; s < 4u; s++) {
    \\        const size_t col = (size_t)(ib + irow[s]);
    \\        float4 a = float4(0.0f);
    \\        a.x = float(x[(size_t)base_r * (size_t)(IDIM) + col]);
    \\        if (g * 4u + 1u < nlive) a.y = float(x[(size_t)(base_r + 1u) * (size_t)(IDIM) + col]);
    \\        if (g * 4u + 2u < nlive) a.z = float(x[(size_t)(base_r + 2u) * (size_t)(IDIM) + col]);
    \\        if (g * 4u + 3u < nlive) a.w = float(x[(size_t)(base_r + 3u) * (size_t)(IDIM) + col]);
    \\        acc[g][s] = fma(a, float4(wt[s]), acc[g][s]);
    \\      }
    \\    }
    \\  }
    \\  for (uint g = 0u; g < 8u; g++) {
    \\    for (uint r = 0u; r < 4u; r++) {
    \\      const uint rr = g * 4u + r;
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\      for (uint s = 0u; s < 4u; s++) {
    \\        partial[pg * 256u + pos[s]] = acc[g][s][r];
    \\      }
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\      if (lid < 16u && rr < nlive) {
    \\        float sum = 0.0f;
    \\        for (uint pr = 0u; pr < 16u; pr++) {
    \\          const uint p = pr * 16u + lid;
    \\          sum += partial[p] + partial[256u + p];
    \\        }
    \\        y[(size_t)(run0 + rr) * (size_t)(ODIM) + ot * TILE + lid] = half(sum);
    \\      }
    \\    }
    \\  }
    \\  row = run_end;
    \\}
;
const GEMM_NAX_INCLUDES: [:0]const u8 =
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace metal;
    \\using namespace mpp::tensor_ops;
    \\
;
const GEMM_NAX_FRAGS: [:0]const u8 =
    \\using nfrag = vec<half, 8>;
    \\static inline nfrag nax_wfrag(const device uint *words, uint lane) {
    \\  const uint source0 = (lane & 16u) + ((lane & 7u) << 1u);
    \\  const uint2 current = *(const device uint2 *)(words + source0);
    \\  const uint previous = words[(source0 + 31u) & 31u];
    \\  const uint slot = ((lane >> 3u) & 1u) * 2u;
    \\  const ulong w0 = ((ulong)previous << 32) | (ulong)current.x;
    \\  const ulong w1 = ((ulong)current.x << 32) | (ulong)current.y;
    \\  const uint s0 = 28u - slot * 4u;
    \\  const uint s1 = 28u - (slot + 4u) * 4u;
    \\  const half2 p00 = exl3_pairh(uint2(uint(w0 >> s0) & 0xffffu, uint(w0 >> (s0 - 4u)) & 0xffffu));
    \\  const half2 p01 = exl3_pairh(uint2(uint(w1 >> s0) & 0xffffu, uint(w1 >> (s0 - 4u)) & 0xffffu));
    \\  const half2 p10 = exl3_pairh(uint2(uint(w0 >> s1) & 0xffffu, uint(w0 >> (s1 - 4u)) & 0xffffu));
    \\  const half2 p11 = exl3_pairh(uint2(uint(w1 >> s1) & 0xffffu, uint(w1 >> (s1 - 4u)) & 0xffffu));
    \\  return nfrag(p00.x, p00.y, p01.x, p01.y, p10.x, p10.y, p11.x, p11.y);
    \\}
    \\template<uint N>
    \\static inline half2 nax_funnel_pair(const device uint *words, uint oracle_thread) {
    \\  const exl3_win w = exl3_pair_window(2u * oracle_thread, N);
    \\  const ulong concat = ((ulong)words[w.i0] << 32) | (ulong)words[w.i1];
    \\  const uint funnel = uint(concat >> w.sh);
    \\  return exl3_pairh(uint2((funnel >> w.fresh) & 0xffffu, funnel & 0xffffu));
    \\}
    \\template<uint N>
    \\static inline nfrag nax_wfrag_k(const device uint *words, uint lane) {
    \\  const uint tau_0 = 64u * (lane >> 4u) + ((lane & 7u) << 3u) + ((lane >> 3u) & 1u);
    \\  const half2 p0 = nax_funnel_pair<N>(words, tau_0);
    \\  const half2 p1 = nax_funnel_pair<N>(words, tau_0 + 2u);
    \\  const half2 p2 = nax_funnel_pair<N>(words, tau_0 + 4u);
    \\  const half2 p3 = nax_funnel_pair<N>(words, tau_0 + 6u);
    \\  return nfrag(p0.x, p0.y, p2.x, p2.y, p1.x, p1.y, p3.x, p3.y);
    \\}
    \\static inline short2 nax_origin(uint lane) {
    \\  const short qid = short(lane >> 2u);
    \\  return short2(short(((qid & 2) | short(lane & 1u)) * 4), short((qid & 4) | short((lane >> 1u) & 3u)));
    \\}
    \\template <typename D>
    \\static inline void nax_zero(thread D &dst) {
    \\  for (uint s = 0u; s < dst.get_capacity(); s++) dst[s] = 0.0f;
    \\}
    \\// `rem` is the block's live row count: the 16-row tile is padded with zeros
    \\// so a short run still runs the full mma. The half4 read is 8-byte aligned
    \\// because IDIM is a multiple of 16 and origin.x a multiple of 4.
    \\template <typename L, typename X>
    \\static inline void nax_left(thread L &left, const device X *x, uint row, uint idim, uint kbase, short2 origin, uint rem) {
    \\  using x4 = vec<X, 4>;
    \\  const size_t r0 = (size_t)(row + uint(origin.y)) * (size_t)idim + kbase + uint(origin.x);
    \\  const size_t r1 = r0 + 8u * (size_t)idim;
    \\  const x4 v0 = (uint(origin.y) < rem) ? *(const device x4 *)(x + r0) : x4(0);
    \\  const x4 v1 = (uint(origin.y) + 8u < rem) ? *(const device x4 *)(x + r1) : x4(0);
    \\  for (uint c = 0u; c < 4u; c++) {
    \\    left[c] = v0[c];
    \\    left[4 + c] = v1[c];
    \\  }
    \\}
    \\template <typename D, typename Y>
    \\static inline void nax_store(const thread D &dst, device Y *y, uint row, uint odim, uint obase, short2 origin, uint rem) {
    \\  const size_t o0 = (size_t)(row + uint(origin.y)) * (size_t)odim;
    \\  const size_t o1 = o0 + 8u * (size_t)odim;
    \\  const bool l0 = uint(origin.y) < rem;
    \\  const bool l1 = uint(origin.y) + 8u < rem;
    \\  for (uint ct = 0u; ct < 2u; ct++) {
    \\    for (uint c = 0u; c < 4u; c++) {
    \\      const uint col = obase + ct * 16u + uint(origin.x) + c;
    \\      if (col >= odim) continue;
    \\      if (l0) y[o0 + col] = Y(dst[ct * 8u + c]);
    \\      if (l1) y[o1 + col] = Y(dst[ct * 8u + 4u + c]);
    \\    }
    \\  }
    \\}
;

const GEMM_NAX_SOURCE: [:0]const u8 =
    \\uint win = uint(threadgroup_position_in_grid.y);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\uint lane = uint(thread_index_in_simdgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_HW = N;
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\const uint start = wstarts[win];
    \\const uint n = wnlive[win];
    \\if (n == 0u || n > uint(WIN)) return;
    \\constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    \\matmul2d<desc, execution_simdgroup> op;
    \\auto left = op.get_left_input_cooperative_tensor<half, half, float>();
    \\auto right = op.get_right_input_cooperative_tensor<half, half, float>();
    \\auto destination = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
    \\auto dest_hi = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
    \\const short2 origin = nax_origin(lane);
    \\const uint output_base = uint(threadgroup_position_in_grid.x) * 128u + sg * 32u;
    \\uint row = start;
    \\const uint end = start + n;
    \\while (row < end) {
    \\const uint run0 = row;
    \\const uint eid = uint(eids[row]);
    \\uint run_end = row + 1u;
    \\while (run_end < end && uint(eids[run_end]) == eid) run_end++;
    \\const uint nlive = run_end - row;
    \\const uint n_hi = (nlive > TILE) ? (nlive - TILE) : 0u;
    \\nax_zero(destination);
    \\if (n_hi > 0u) nax_zero(dest_hi);
    \\const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW);
    \\for (uint tk = 0u; tk < IT; tk++) {
    \\  const uint kbase = tk * TILE;
    \\  const device uint *words0 = trellis_e + ((size_t)tk * (size_t)OT + output_base / TILE) * PACKED_W;
    \\  nfrag w0, w1;
    \\  if (N == 64u) {
    \\    w0 = nax_wfrag(words0, lane);
    \\    w1 = nax_wfrag(words0 + PACKED_W, lane);
    \\  } else {
    \\    w0 = nax_wfrag_k<N>(words0, lane);
    \\    w1 = nax_wfrag_k<N>(words0 + PACKED_W, lane);
    \\  }
    \\  for (short s = 0; s < 8; s++) {
    \\    right[s] = w0[s];
    \\    right[8 + s] = w1[s];
    \\  }
    \\  nax_left(left, x, run0, uint(IDIM), kbase, origin, nlive);
    \\  op.run(left, right, destination);
    \\  if (n_hi > 0u) {
    \\    nax_left(left, x, run0 + TILE, uint(IDIM), kbase, origin, n_hi);
    \\    op.run(left, right, dest_hi);
    \\  }
    \\}
    \\nax_store(destination, y, run0, uint(ODIM), output_base, origin, nlive);
    \\if (n_hi > 0u) nax_store(dest_hi, y, run0 + TILE, uint(ODIM), output_base, origin, n_hi);
    \\row = run_end;
    \\}
;
const TOKEN_PREPARE_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint eid = uint(slots[slot]);
    \\const uint orig = uint(order[slot]);
    \\const uint row = orig / uint(TOPK);
    \\const uint base = block * 128u;
    \\const size_t xb = (size_t)row * (size_t)(IDIM) + base;
    \\const size_t yb = (size_t)slot * (size_t)(IDIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(IDIM) + base;
    \\float4 v = float4(
    \\  float(x[xb + lane]) * float(suh[sb + lane]),
    \\  float(x[xb + lane + 32u]) * float(suh[sb + lane + 32u]),
    \\  float(x[xb + lane + 64u]) * float(suh[sb + lane + 64u]),
    \\  float(x[xb + lane + 96u]) * float(suh[sb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\const float sc = 0.08838834764831845f;
    \\y[yb + lane] = half((s0 + s2) * sc);
    \\y[yb + lane + 32u] = half((s1 + s3) * sc);
    \\y[yb + lane + 64u] = half((s0 - s2) * sc);
    \\y[yb + lane + 96u] = half((s1 - s3) * sc);
;

const TOKEN_PAIR_PREPARE_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint eid = uint(slots[slot]);
    \\const uint orig = uint(order[slot]);
    \\const uint row = orig / uint(TOPK);
    \\const uint base = block * 128u;
    \\const size_t xb = (size_t)row * (size_t)(IDIM) + base;
    \\const size_t yb = (size_t)slot * (size_t)(IDIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(IDIM) + base;
    \\const float x0 = float(x[xb + lane]);
    \\const float x1 = float(x[xb + lane + 32u]);
    \\const float x2 = float(x[xb + lane + 64u]);
    \\const float x3 = float(x[xb + lane + 96u]);
    \\float4 v = float4(
    \\  x0 * float(suhg[sb + lane]),
    \\  x1 * float(suhg[sb + lane + 32u]),
    \\  x2 * float(suhg[sb + lane + 64u]),
    \\  x3 * float(suhg[sb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float sc = 0.08838834764831845f;
    \\float s0 = v.x + v.y;
    \\float s1 = v.x - v.y;
    \\float s2 = v.z + v.w;
    \\float s3 = v.z - v.w;
    \\yg[yb + lane] = half((s0 + s2) * sc);
    \\yg[yb + lane + 32u] = half((s1 + s3) * sc);
    \\yg[yb + lane + 64u] = half((s0 - s2) * sc);
    \\yg[yb + lane + 96u] = half((s1 - s3) * sc);
    \\v = float4(
    \\  x0 * float(suhu[sb + lane]),
    \\  x1 * float(suhu[sb + lane + 32u]),
    \\  x2 * float(suhu[sb + lane + 64u]),
    \\  x3 * float(suhu[sb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\s0 = v.x + v.y;
    \\s1 = v.x - v.y;
    \\s2 = v.z + v.w;
    \\s3 = v.z - v.w;
    \\yu[yb + lane] = half((s0 + s2) * sc);
    \\yu[yb + lane + 32u] = half((s1 + s3) * sc);
    \\yu[yb + lane + 64u] = half((s0 - s2) * sc);
    \\yu[yb + lane + 96u] = half((s1 - s3) * sc);
;

const TOKEN_SCATTER_SOURCE: [:0]const u8 =
    \\uint col = uint(thread_position_in_grid.x);
    \\uint slot = uint(thread_position_in_grid.y);
    \\if (col >= uint(DIM)) return;
    \\const uint orig = uint(order[slot]);
    \\y[(size_t)orig * (size_t)(DIM) + col] = x[(size_t)slot * (size_t)(DIM) + col];
;

const TOKEN_REDUCE_SOURCE: [:0]const u8 =
    \\uint col = uint(thread_position_in_grid.x);
    \\uint row = uint(thread_position_in_grid.y);
    \\if (col >= uint(ODIM)) return;
    \\half acc = half(0.0f);
    \\for (uint k = 0u; k < uint(TOPK); k++) {
    \\  const uint orig = row * uint(TOPK) + k;
    \\  const uint si = uint(inv[orig]);
    \\  const half p = half(float(d[(size_t)si * (size_t)(ODIM) + col]) * float(half(sc[orig])));
    \\  acc += p;
    \\}
    \\y[(size_t)row * (size_t)(ODIM) + col] = acc;
;

const PREPARE_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint eid = uint(slots[slot]);
    \\const uint base = block * 128u;
    \\const size_t xb = (size_t)slot * (size_t)(IDIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(IDIM) + base;
    \\float4 v = float4(
    \\  float(x[xb + lane]) * float(suh[sb + lane]),
    \\  float(x[xb + lane + 32u]) * float(suh[sb + lane + 32u]),
    \\  float(x[xb + lane + 64u]) * float(suh[sb + lane + 64u]),
    \\  float(x[xb + lane + 96u]) * float(suh[sb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\const float sc = 0.08838834764831845f;
    \\y[xb + lane] = half((s0 + s2) * sc);
    \\y[xb + lane + 32u] = half((s1 + s3) * sc);
    \\y[xb + lane + 64u] = half((s0 - s2) * sc);
    \\y[xb + lane + 96u] = half((s1 - s3) * sc);
;

const FINISH_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint eid = uint(slots[slot]);
    \\const uint base = block * 128u;
    \\const size_t xb = (size_t)slot * (size_t)(ODIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(ODIM) + base;
    \\float4 v = float4(
    \\  float(inner[xb + lane]),
    \\  float(inner[xb + lane + 32u]),
    \\  float(inner[xb + lane + 64u]),
    \\  float(inner[xb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\const float sc = 0.08838834764831845f;
    \\y[xb + lane] = half((s0 + s2) * sc * float(svh[sb + lane]));
    \\y[xb + lane + 32u] = half((s1 + s3) * sc * float(svh[sb + lane + 32u]));
    \\y[xb + lane + 64u] = half((s0 - s2) * sc * float(svh[sb + lane + 64u]));
    \\y[xb + lane + 96u] = half((s1 - s3) * sc * float(svh[sb + lane + 96u]));
;

var gemm_sorted_kernel: KernelSlots = no_kernels;
var prepare_kernel: ?mlx.mlx_fast_metal_kernel = null;
var finish_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gemv_kernel: KernelSlots = no_kernels;
var gemv_engaged: bool = false;

/// How the next dispatch decodes: the pack's codebook and codeword window.
/// A weight kernel is built per (codebook, window), so the slots below are
/// indexed by both; the model's own pair is asserted at each dispatch entry,
/// since several packs can be resident.
var active_decode: exl3.Decode = .mul1;
pub fn setDecodeParams(dec: exl3.Decode) void {
    active_decode = dec;
}
const N_CB = exl3.Codebook.count;
const N_WIN = exl3.Window.count;
const KernelSlots = [N_CB][N_WIN]?mlx.mlx_fast_metal_kernel;
const no_kernels: KernelSlots = @splat(@splat(null));
fn cbIndex(comptime cb: exl3.Codebook) usize {
    return @intFromEnum(cb);
}
fn winSuffix(comptime win: exl3.Window) [:0]const u8 {
    return comptime if (win == .w16) "" else std.fmt.comptimePrint("_w{d}", .{win.bits()});
}
fn cbSuffix(comptime cb: exl3.Codebook) [:0]const u8 {
    return switch (cb) {
        .mul1 => "",
        .mcg => "_mcg",
        .tiny => "_tiny",
    };
}

/// The one decode every weight kernel calls, two codewords to two halves. A
/// pack whose search hashed a narrower window masks the sliding window down to
/// it first; w16 masks nothing and emits the source it always did.
fn codebookHelpers(comptime cb: exl3.Codebook, comptime win: exl3.Window) [:0]const u8 {
    const body: [:0]const u8 = comptime switch (cb) {
        .mul1 =>
        \\  const uint2 mixed = cw * uint2(0x83DCD12Du);
        \\  const uint2 pair_sums = (mixed & uint2(0x00FF00FFu)) + ((mixed >> uint2(8u)) & uint2(0x00FF00FFu));
        \\  const uint2 byte_sum = uint2(0x6400u) + (pair_sums & uint2(0xFFFFu)) + (pair_sums >> uint2(16u));
        \\  const half2 h = as_type<half2>(ushort2(byte_sum & uint2(0xFFFFu)));
        \\  return fma(h, as_type<half2>(ushort2(0x1EEEu)), as_type<half2>(ushort2(0xC931u)));
        ,
        .mcg =>
        \\  const uint2 r = ((cw * uint2(0xCBAC1FEDu)) & uint2(0x8FFF8FFFu)) ^ uint2(0x3B603B60u);
        \\  const half4 h = as_type<half4>(r);
        \\  return half2(h.x + h.y, h.z + h.w);
        ,
        .tiny =>
        \\  const uint2 r = ((cw * uint2(0xCBAC1FEDu)) & uint2(0x8FFF8FFFu)) + uint2(0x32003100u);
        \\  const half4 h = as_type<half4>(r);
        \\  return half2(h.x + h.y, h.z + h.w);
        ,
    };
    const mask: [:0]const u8 = comptime if (win == .w16) "" else std.fmt.comptimePrint("  cw &= uint2(0x{X}u);\n", .{win.mask()});
    const pair = comptime "static inline half2 exl3_pairh(uint2 cw) {\n" ++ mask ++ body ++ "\n}\n";
    return comptime pair ++
        \\static inline float2 exl3_decode2(uint2 cw) { return float2(exl3_pairh(cw)); }
        \\static inline float exl3_decode1(uint cw) { return exl3_decode2(uint2(cw, 0u)).x; }
        \\// Weight t's 16-bit codeword is the window ending at floor((t+1)*K), with
        \\// K = N/16; the pair (t0, t0+1) shares one 32-bit funnel read.
        \\struct exl3_win { uint i0; uint i1; uint sh; uint fresh; };
        \\static inline exl3_win exl3_pair_window(uint t0, uint N) {
        \\  const uint e0 = ((t0 + 1u) * N) >> 4u;
        \\  const uint e1 = ((t0 + 2u) * N) >> 4u;
        \\  const uint b0 = e0 + 16u * N - 16u;
        \\  const uint b2 = e1 + 16u * N;
        \\  const uint j0 = b0 >> 5u;
        \\  const uint j1 = (b2 - 1u) >> 5u;
        \\  exl3_win w;
        \\  w.i0 = j0 % (N >> 1u);
        \\  w.i1 = j1 % (N >> 1u);
        \\  w.sh = (j1 + 1u) * 32u - b2;
        \\  w.fresh = e1 - e0;
        \\  return w;
        \\}
        \\
    ;
}

fn naxHeader(comptime cb: exl3.Codebook, comptime win: exl3.Window) [:0]const u8 {
    return comptime GEMM_NAX_INCLUDES ++ codebookHelpers(cb, win) ++ GEMM_NAX_FRAGS;
}

/// A weight kernel under the active decode parameters: its own slot, name and
/// header.
fn codebookKernel(slots: *KernelSlots, comptime base: [:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    switch (active_decode.codebook) {
        inline else => |cb| switch (active_decode.window) {
            inline else => |win| return getNamedKernel(
                &slots[cbIndex(cb)][win.index()],
                comptime base ++ cbSuffix(cb) ++ winSuffix(win),
                ins,
                outs,
                source,
                codebookHelpers(cb, win),
            ),
        },
    }
}

const GemvKey = struct { in_dim: c_int, out_dim: c_int, n: u32 };

fn CfgCache(comptime Key: type, comptime CAP: usize) type {
    return struct {
        const Self = @This();
        keys: [CAP]Key = @splat(std.mem.zeroes(Key)),
        cfgs: [CAP]?mlx.mlx_fast_metal_kernel_config = @splat(null),
        used: [CAP]u64 = @splat(0),
        tick: u64 = 0,

        fn get(self: *Self, key: Key) ?mlx.mlx_fast_metal_kernel_config {
            for (self.cfgs, 0..) |c, i| {
                if (c != null and std.meta.eql(self.keys[i], key)) {
                    self.tick += 1;
                    self.used[i] = self.tick;
                    return c.?;
                }
            }
            return null;
        }

        fn put(self: *Self, key: Key, cfg: mlx.mlx_fast_metal_kernel_config) void {
            var victim: usize = 0;
            var oldest: u64 = std.math.maxInt(u64);
            for (self.cfgs, 0..) |c, i| {
                if (c == null) {
                    victim = i;
                    break;
                }
                if (self.used[i] < oldest) {
                    oldest = self.used[i];
                    victim = i;
                }
            }
            if (self.cfgs[victim]) |old| _ = mlx.mlx_fast_metal_kernel_config_free(old);
            self.cfgs[victim] = cfg;
            self.keys[victim] = key;
            self.tick += 1;
            self.used[victim] = self.tick;
        }
    };
}

const IndexedKey = struct { in_dim: c_int, out_dim: c_int, topk: c_int, n: u32 };
const UnaryKey = struct { dim: c_int, topk: c_int };
const GemmSortedKey = struct { in_dim: c_int, out_dim: c_int, rows: c_int, win: c_int, n: u32 };

const GEMM_WINDOW_ROWS: c_int = 32;

/// Rows one run-window may carry. `GEMM_SORTED_SOURCE` accumulates `acc[8][4]`
/// and the NAX body has two 16-row destinations, so past this the kernels'
/// own `n > WIN` guard still admits the window and the extra rows go unwritten.
const GEMM_WINDOW_MAX_ROWS: c_int = 32;

var gemm_win_cached: ?c_int = null;
var gemm_align_cached: ?bool = null;

/// null = not a window these kernels run, so the default stands.
fn resolveGemmWindowRows(raw: ?[]const u8) ?c_int {
    const v = raw orelse return null;
    const n = std.fmt.parseInt(c_int, v, 10) catch return null;
    if (n < 1 or n > GEMM_WINDOW_MAX_ROWS) return null;
    return n;
}

fn gemmWindowRows() c_int {
    if (gemm_win_cached) |v| return v;
    const v = blk: {
        const p = std.c.getenv("MLX_SERVE_EXL3_GEMM_WIN") orelse break :blk GEMM_WINDOW_ROWS;
        const raw = std.mem.span(p);
        if (resolveGemmWindowRows(raw)) |n| break :blk n;
        log.warn("[exl3] MLX_SERVE_EXL3_GEMM_WIN={s} is not a window in 1..{d}; keeping {d}\n", .{ raw, GEMM_WINDOW_MAX_ROWS, GEMM_WINDOW_ROWS });
        break :blk GEMM_WINDOW_ROWS;
    };
    gemm_win_cached = v;
    return v;
}

fn gemmWindowAligned() bool {
    if (gemm_align_cached) |v| return v;
    var on = true;
    if (std.c.getenv("MLX_SERVE_EXL3_WIN_ALIGN")) |p| {
        const v = std.mem.span(p);
        if (v.len > 0 and v[0] == '0') on = false;
    }
    gemm_align_cached = on;
    return on;
}
var indexed_coop_cfgs: CfgCache(IndexedKey, 8) = .{};
var prepare_cfgs: CfgCache(UnaryKey, 8) = .{};
var finish_cfgs: CfgCache(UnaryKey, 8) = .{};
var gemm_sorted_cfgs: CfgCache(GemmSortedKey, 8) = .{};
var gemm_nax_cfgs: CfgCache(GemmSortedKey, 8) = .{};
var gemm_nax_kernel: KernelSlots = no_kernels;
var gemm_nax_failed: bool = false;
var gemm_nax_cached: ?bool = null;
const PairPrepKey = struct { in_dim: c_int, nslots: c_int, topk: c_int };
const PairGemvKey = struct { in_dim: c_int, out_dim: c_int, nslots: c_int, nsplit: c_int, topk: c_int, n: u32 };
const DownFusedKey = struct { in_dim: c_int, out_dim: c_int, nslots: c_int, nsplit: c_int, n: u32 };
const MidKey = struct { dim: c_int, nslots: c_int };
const ReduceKey = struct { out_dim: c_int, rows: c_int, topk: c_int };
const DecodeReduceKey = struct { out_dim: c_int, rows: c_int, topk: c_int, dtype: mlx.mlx_dtype };
var pair_gemv_cfgs: CfgCache(PairGemvKey, 8) = .{};
var mid_cfgs: CfgCache(MidKey, 8) = .{};
var reduce_cfgs: CfgCache(DecodeReduceKey, 8) = .{};
var down_fused_cfgs: CfgCache(DownFusedKey, 8) = .{};
var token_prep_cfgs: CfgCache(PairPrepKey, 8) = .{};
var token_pair_prep_cfgs: CfgCache(PairPrepKey, 8) = .{};
var token_reduce_cfgs: CfgCache(ReduceKey, 8) = .{};
const ScatterKey = struct { dim: c_int, nslots: c_int };
var token_scatter_cfgs: CfgCache(ScatterKey, 8) = .{};
var token_prepare_kernel: ?mlx.mlx_fast_metal_kernel = null;
var token_pair_prepare_kernel: ?mlx.mlx_fast_metal_kernel = null;
var token_scatter_kernel: ?mlx.mlx_fast_metal_kernel = null;
var token_reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;
var fused_dispatches: u32 = 0;
var apply_host_ns: u64 = 0;
var apply_host_n: u32 = 0;
var apply_host_layers: u32 = 0;
var apply_host_dumps: u32 = 0;
var apply_ubench_env: ?bool = null;

fn applyUbenchOn() bool {
    if (apply_ubench_env) |v| return v;
    const v = diagEnvValueOn(std.c.getenv("MLX_SERVE_DECODE_TICK_UBENCH"));
    apply_ubench_env = v;
    return v;
}

pub fn resetFusedDispatchCount() void {
    fused_dispatches = 0;
}

pub fn fusedDispatchCount() u32 {
    return fused_dispatches;
}

fn gpuArch(buf: []u8) ?[]const u8 {
    var dev = mlx.mlx_device{ .ctx = null };
    if (mlx.mlx_get_default_device(&dev) != 0) return null;
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) != 0) return null;
    var cstr: [*:0]const u8 = undefined;
    if (mlx.mlx_device_info_get_string(&cstr, info, "architecture") != 0) return null;
    const arch = std.mem.span(cstr);
    if (arch.len == 0 or arch.len > buf.len) return null;
    @memcpy(buf[0..arch.len], arch);
    return buf[0..arch.len];
}

fn gemmNaxOn() bool {
    if (gemm_nax_failed) return false;
    if (std.c.getenv("MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK")) |p| {
        const v = std.mem.span(p);
        if (v.len > 0 and v[0] == '1') return false;
    }
    if (gemm_nax_cached) |v| return v;
    var buf: [128]u8 = undefined;
    const arch = gpuArch(&buf) orelse {
        gemm_nax_cached = false;
        return false;
    };
    var i: usize = 0;
    while (i + 2 < arch.len) : (i += 1) {
        const a = arch[i] | 32;
        const b = arch[i + 1] | 32;
        if (a == 'g' and b == '1' and arch[i + 2] >= '7' and arch[i + 2] <= '9') {
            gemm_nax_cached = true;
            return true;
        }
    }
    gemm_nax_cached = false;
    return false;
}

fn getGemmNaxKernel() !mlx.mlx_fast_metal_kernel {
    switch (active_decode.codebook) {
        inline else => |cb| switch (active_decode.window) {
            inline else => |win| {
                if (gemm_nax_kernel[cbIndex(cb)][win.index()]) |k| return k;
                const kernel = buildNaxGemmKernel(GEMM_NAX_SOURCE, naxHeader(cb, win), comptime "mlxserve_exl3_k4_gemm_nax" ++ cbSuffix(cb) ++ winSuffix(win)) orelse {
                    gemm_nax_failed = true;
                    log.info("[exl3-gemm] NAX arm declined (kernel probe failed); the sorted GEMM serves prefill\n", .{});
                    return error.MetalKernelCompileFailed;
                };
                gemm_nax_kernel[cbIndex(cb)][win.index()] = kernel;
                return kernel;
            },
        },
    }
}

/// Metal JIT-compiles a kernel at its first EVAL, not at apply, so a source the
/// toolchain rejects is proven here on a one-tile problem before any prefill
/// depends on it. null = the arm is unusable; the latch it raised is dropped.
fn buildNaxGemmKernel(source: [:0]const u8, header: [:0]const u8, name: [*:0]const u8) ?mlx.mlx_fast_metal_kernel {
    const input_names = [_][*:0]const u8{ "x", "trellis", "eids", "wstarts", "wnlive" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, true, false);
    if (kernel.ctx == null) return null;
    if (probeNaxGemm(kernel)) return kernel;
    _ = mlx.mlx_fast_metal_kernel_free(kernel);
    return null;
}

fn probeNaxGemm(kernel: mlx.mlx_fast_metal_kernel) bool {
    const s = mlx.gpuStream();
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const in_dim: c_int = 16;
    const out_dim: c_int = 128;
    if (mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, out_dim }, 2, .float16) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_dim, 1, 1) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IDIM", in_dim) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ODIM", out_dim) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WIN", gemmWindowRows()) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NHW", 64) != 0) return false;
    const xh = std.mem.zeroes([16]u16);
    const x = mlx.mlx_array_new_data(&xh, &[_]c_int{ 1, in_dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(x);
    const th = std.mem.zeroes([8 * 64]u16);
    const trellis = mlx.mlx_array_new_data(&th, &[_]c_int{ 1, 1, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trellis);
    const zero = [_]u32{0};
    const one = [_]u32{1};
    const eids = mlx.mlx_array_new_data(&zero, &[_]c_int{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eids);
    const starts = mlx.mlx_array_new_data(&zero, &[_]c_int{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(starts);
    const nlive = mlx.mlx_array_new_data(&one, &[_]c_int{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(nlive);
    const inputs = [_]mlx.mlx_array{ x, trellis, eids, starts, nlive };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    if (mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, inputs_vec, cfg, s) != 0) return false;
    if (mlx.mlx_vector_array_size(outputs) != 1) return false;
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    if (mlx.mlx_vector_array_get(&out, outputs, 0) != 0) return false;
    if (mlx.mlx_array_eval(out) != 0) return false;
    return !mlx.errorPending();
}

fn getGemmSortedKernel() !mlx.mlx_fast_metal_kernel {
    const ins = [_][*:0]const u8{ "x", "trellis", "eids", "wstarts", "wnlive" };
    const outs = [_][*:0]const u8{"y"};
    return codebookKernel(&gemm_sorted_kernel, "mlxserve_exl3_k4_gemm_sorted", &ins, &outs, GEMM_SORTED_SOURCE);
}

const WindowTable = struct { starts: mlx.mlx_array, nlives: mlx.mlx_array, nwin: c_int };

var gemm_sel_logged: bool = false;

fn logGemmSelector(win: c_int, aligned: bool, nwin: c_int, n: c_int) void {
    if (n < 2048) return;
    if (gemm_sel_logged) return;
    gemm_sel_logged = true;
    log.info("[exl3-gemm] win={d} aligned={d} nwin={d} mixed={d} n={d}\n", .{
        win,
        @intFromBool(aligned),
        nwin,
        @as(u32, if (aligned) 0 else 1),
        n,
    });
}

fn buildWindowTable(s: mlx.mlx_stream, eids: mlx.mlx_array, n: c_int, win: c_int) !WindowTable {
    return buildWindowTableHost(s, eids, n, win);
}

fn gemmWindowTable(s: mlx.mlx_stream, eids: mlx.mlx_array, n: c_int, win: c_int, aligned: bool) !WindowTable {
    return if (aligned) buildWindowTable(s, eids, n, win) else buildStrideTable(s, n, win);
}

fn buildWindowTableHost(s: mlx.mlx_stream, eids: mlx.mlx_array, n: c_int, win: c_int) !WindowTable {
    const ids = try std.heap.page_allocator.alloc(u32, @intCast(n));
    defer std.heap.page_allocator.free(ids);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, eids, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    switch (mlx.mlx_array_dtype(contig)) {
        .uint32 => {
            const p = mlx.mlx_array_data_uint32(contig) orelse return error.F16Unreadable;
            @memcpy(ids, p[0..ids.len]);
        },
        .int32 => {
            const p = mlx.mlx_array_data_int32(contig) orelse return error.F16Unreadable;
            for (ids, 0..) |*d, i| d.* = @intCast(p[i]);
        },
        else => return error.BadExl3Shape,
    }
    const runs = try buildRuns(std.heap.page_allocator, ids);
    defer std.heap.page_allocator.free(runs.start);
    defer std.heap.page_allocator.free(runs.len);
    defer std.heap.page_allocator.free(runs.eid);
    const w: u32 = @intCast(win);
    var nwin_u: u32 = 0;
    var r: u32 = 0;
    while (r < runs.n) : (r += 1) {
        nwin_u += (runs.len[r] + w - 1) / w;
    }
    const sh = try std.heap.page_allocator.alloc(u32, nwin_u);
    defer std.heap.page_allocator.free(sh);
    const lh = try std.heap.page_allocator.alloc(u32, nwin_u);
    defer std.heap.page_allocator.free(lh);
    var k: u32 = 0;
    r = 0;
    while (r < runs.n) : (r += 1) {
        var off: u32 = 0;
        while (off < runs.len[r]) {
            const live = @min(w, runs.len[r] - off);
            sh[k] = runs.start[r] + off;
            lh[k] = live;
            k += 1;
            off += live;
        }
    }
    const starts_raw = mlx.mlx_array_new_data(sh.ptr, &[_]c_int{@intCast(nwin_u)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(starts_raw);
    const nlives_raw = mlx.mlx_array_new_data(lh.ptr, &[_]c_int{@intCast(nwin_u)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(nlives_raw);
    var starts = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(starts);
    var nlives = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(nlives);
    try mlx.check(mlx.mlx_contiguous(&starts, starts_raw, false, s));
    try mlx.check(mlx.mlx_contiguous(&nlives, nlives_raw, false, s));
    try mlx.check(mlx.mlx_array_eval(starts));
    try mlx.check(mlx.mlx_array_eval(nlives));
    if (exl3UbenchOn()) {
        benchPrint("[exl3-ubench] win_table_host nwin={d} n={d} win={d} eval=eids\n", .{ nwin_u, n, win });
    }
    return .{ .starts = starts, .nlives = nlives, .nwin = @intCast(nwin_u) };
}

fn buildStrideTable(s: mlx.mlx_stream, n: c_int, win: c_int) !WindowTable {
    const w: u32 = @intCast(win);
    const nn: u32 = @intCast(n);
    const nwin_u = (nn + w - 1) / w;
    const sh = try std.heap.page_allocator.alloc(u32, nwin_u);
    defer std.heap.page_allocator.free(sh);
    const lh = try std.heap.page_allocator.alloc(u32, nwin_u);
    defer std.heap.page_allocator.free(lh);
    var i: u32 = 0;
    while (i < nwin_u) : (i += 1) {
        const st = i * w;
        sh[i] = st;
        lh[i] = @min(w, nn - st);
    }
    const starts_raw = mlx.mlx_array_new_data(sh.ptr, &[_]c_int{@intCast(nwin_u)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(starts_raw);
    const nlives_raw = mlx.mlx_array_new_data(lh.ptr, &[_]c_int{@intCast(nwin_u)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(nlives_raw);
    var starts = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(starts);
    var nlives = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(nlives);
    try mlx.check(mlx.mlx_contiguous(&starts, starts_raw, false, s));
    try mlx.check(mlx.mlx_contiguous(&nlives, nlives_raw, false, s));
    try mlx.check(mlx.mlx_array_eval(starts));
    try mlx.check(mlx.mlx_array_eval(nlives));
    if (exl3UbenchOn()) {
        benchPrint("[exl3-ubench] win_table_stride nwin={d} n={d} win={d}\n", .{ nwin_u, n, win });
    }
    return .{ .starts = starts, .nlives = nlives, .nwin = @intCast(nwin_u) };
}

fn windowStats(ids: []const u32, win: u32, aligned: bool) struct { nwin: u32, mixed: u32, decodes: u32 } {
    if (aligned) {
        const runs = buildRuns(std.heap.page_allocator, ids) catch return .{ .nwin = 0, .mixed = 0, .decodes = 0 };
        defer std.heap.page_allocator.free(runs.start);
        defer std.heap.page_allocator.free(runs.len);
        defer std.heap.page_allocator.free(runs.eid);
        var nwin: u32 = 0;
        var r: u32 = 0;
        while (r < runs.n) : (r += 1) {
            nwin += (runs.len[r] + win - 1) / win;
        }
        return .{ .nwin = nwin, .mixed = 0, .decodes = nwin };
    }
    const nwin = (@as(u32, @intCast(ids.len)) + win - 1) / win;
    var mixed: u32 = 0;
    var decodes: u32 = 0;
    var w: u32 = 0;
    while (w < nwin) : (w += 1) {
        const st = w * win;
        const nlive = @min(win, @as(u32, @intCast(ids.len)) - st);
        var runs_here: u32 = 1;
        var i: u32 = 1;
        while (i < nlive) : (i += 1) {
            if (ids[st + i] != ids[st + i - 1]) runs_here += 1;
        }
        decodes += runs_here;
        if (runs_here > 1) mixed += 1;
    }
    return .{ .nwin = nwin, .mixed = mixed, .decodes = decodes };
}

pub fn innerGemmSorted(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    trellis: mlx.mlx_array,
    eids: mlx.mlx_array,
) !mlx.mlx_array {
    return innerGemmSortedWinAlign(s, x, trellis, eids, gemmWindowRows(), gemmWindowAligned());
}

fn innerGemmSortedWin(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    trellis: mlx.mlx_array,
    eids: mlx.mlx_array,
    win: c_int,
) !mlx.mlx_array {
    return innerGemmSortedWinAlign(s, x, trellis, eids, win, true);
}

fn innerGemmSortedWinAlign(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    trellis: mlx.mlx_array,
    eids: mlx.mlx_array,
    win: c_int,
    aligned: bool,
) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    if (xsh.len != 2) return error.BadExl3Shape;
    if (win <= 0 or win > GEMM_WINDOW_MAX_ROWS) return error.BadExl3Shape;
    const tab = try gemmWindowTable(s, eids, xsh[0], win, aligned);
    defer _ = mlx.mlx_array_free(tab.starts);
    defer _ = mlx.mlx_array_free(tab.nlives);
    return innerGemmSortedTable(s, x, trellis, eids, win, aligned, tab);
}

/// The window table depends only on the sorted slots, so a layer's three
/// projections share one: each build is a host eval that drains the GPU.
fn innerGemmSortedTable(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    trellis: mlx.mlx_array,
    eids: mlx.mlx_array,
    win: c_int,
    aligned: bool,
    tab: WindowTable,
) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(trellis);
    if (xsh.len != 2 or tsh.len != 4) return error.BadExl3Shape;
    if (win <= 0 or win > GEMM_WINDOW_MAX_ROWS) return error.BadExl3Shape;
    const n = xsh[0];
    const in_dim = xsh[1];
    const out_dim = tsh[2] * 16;
    const out_tiles = tsh[2];
    const rate = try packedRate(tsh[3]);
    if (tsh[1] * 16 != in_dim) return error.BadExl3Shape;
    if (tab.nwin <= 0) return error.BadExl3Shape;
    logGemmSelector(win, aligned, tab.nwin, n);
    const key = GemmSortedKey{ .in_dim = in_dim, .out_dim = out_dim, .rows = n, .win = win, .n = rate.n };
    if (gemmNaxOn() and @rem(out_dim, 128) == 0) {
        // The fallback below answers a NAX build or dispatch that failed, so the
        // failure's latch is ours to drop: left standing it becomes the next
        // decode tick's `MlxFailure`.
        const had_error = mlx.errorPending();
        if (getGemmNaxKernel()) |nk| {
            const ncfg = gemm_nax_cfgs.get(key) orelse blk: {
                const c = mlx.mlx_fast_metal_kernel_config_new();
                errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ n, out_dim }, 2, .float16));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "WIN", win));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
                gemm_nax_cfgs.put(key, c);
                break :blk c;
            };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(ncfg, out_dim, tab.nwin, 1));
            const ninputs = [_]mlx.mlx_array{ x, trellis, eids, tab.starts, tab.nlives };
            const ninputs_vec = mlx.mlx_vector_array_new_data(&ninputs, ninputs.len);
            defer _ = mlx.mlx_vector_array_free(ninputs_vec);
            var noutputs = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(noutputs);
            if (mlx.mlx_fast_metal_kernel_apply(&noutputs, nk, ninputs_vec, ncfg, s) == 0 and mlx.mlx_vector_array_size(noutputs) == 1) {
                var nout = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(nout);
                try mlx.check(mlx.mlx_vector_array_get(&nout, noutputs, 0));
                return nout;
            }
            gemm_nax_failed = true;
        } else |_| {
            gemm_nax_failed = true;
        }
        mlx.dropLatchedErrorUnless(had_error);
    }
    const cfg = gemm_sorted_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ n, out_dim }, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "WIN", win));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        gemm_sorted_cfgs.put(key, c);
        break :blk c;
    };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_tiles * 128, tab.nwin, 1));
    const inputs = [_]mlx.mlx_array{ x, trellis, eids, tab.starts, tab.nlives };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, try getGemmSortedKernel(), inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

fn getPrepareKernel() !mlx.mlx_fast_metal_kernel {
    if (prepare_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "suh", "slots" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_exl3_prepare_h128",
        in_vec,
        out_vec,
        PREPARE_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    prepare_kernel = kernel;
    return kernel;
}

fn getFinishKernel() !mlx.mlx_fast_metal_kernel {
    if (finish_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "inner", "svh", "slots" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_exl3_finish_h128",
        in_vec,
        out_vec,
        FINISH_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    finish_kernel = kernel;
    return kernel;
}

fn applyUnary(
    s: mlx.mlx_stream,
    kernel: mlx.mlx_fast_metal_kernel,
    inputs: []const mlx.mlx_array,
    cfg: mlx.mlx_fast_metal_kernel_config,
) !mlx.mlx_array {
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

pub fn prepareIndexed(s: mlx.mlx_stream, x_in: mlx.mlx_array, suh: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const topk = mlx.getShape(slots)[0];
    const xsh = mlx.getShape(x_in);
    const in_dim = xsh[xsh.len - 1];
    // The kernel reads `topk` rows of x, so a rank-2 x must already carry them.
    if (xsh.len > 2 or (xsh.len == 2 and xsh[0] != topk)) return error.BadExl3Shape;
    var x = x_in;
    var owned = false;
    if (xsh.len == 1) {
        var b = mlx.mlx_array_new();
        const shape = [_]c_int{ topk, in_dim };
        try mlx.check(mlx.mlx_broadcast_to(&b, x_in, &shape, 2, s));
        x = b;
        owned = true;
    }
    defer if (owned) {
        _ = mlx.mlx_array_free(x);
    };
    const blocks = @divExact(in_dim, 128);
    const key = UnaryKey{ .dim = in_dim, .topk = topk };
    const cfg = prepare_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const out_shape = [_]c_int{ topk, in_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, topk, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        prepare_cfgs.put(key, c);
        break :blk c;
    };
    return applyUnary(s, try getPrepareKernel(), &.{ x, suh, slots }, cfg);
}

pub fn finishIndexed(s: mlx.mlx_stream, inner: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const sh = mlx.getShape(inner);
    const topk = sh[0];
    const out_dim = sh[1];
    const blocks = @divExact(out_dim, 128);
    const key = UnaryKey{ .dim = out_dim, .topk = topk };
    const cfg = finish_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const out_shape = [_]c_int{ topk, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, topk, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        finish_cfgs.put(key, c);
        break :blk c;
    };
    return applyUnary(s, try getFinishKernel(), &.{ inner, svh, slots }, cfg);
}

fn getGemvKernel() !mlx.mlx_fast_metal_kernel {
    const ins = [_][*:0]const u8{ "x", "trellis" };
    const outs = [_][*:0]const u8{"y"};
    return codebookKernel(&gemv_kernel, "mlxserve_exl3_k4_mcg_gemv", &ins, &outs, GEMV_SOURCE);
}

var inner_gemv_cfgs: CfgCache(GemvKey, 8) = .{};

fn gemvConfig(in_dim: c_int, out_dim: c_int, rate: exl3.Rate) !mlx.mlx_fast_metal_kernel_config {
    const key = GemvKey{ .in_dim = in_dim, .out_dim = out_dim, .n = rate.n };
    if (inner_gemv_cfgs.get(key)) |c| return c;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_shape = [_]c_int{out_dim};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 1, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_dim, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IDIM", in_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ODIM", out_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NHW", @intCast(rate.n)));
    inner_gemv_cfgs.put(key, cfg);
    return cfg;
}

const INDEXED_COOP_SOURCE: [:0]const u8 =
    \\threadgroup float partial[4 * 256];
    \\uint ot = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\uint split = uint(threadgroup_position_in_grid.z);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\uint lane = uint(thread_index_in_simdgroup);
    \\uint lid = uint(thread_index_in_threadgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_HW = N;
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\constexpr uint SGS = 4u;
    \\constexpr uint SPLITS = 1u;
    \\const uint eid = uint(slots[slot]);
    \\const uint tiles_per_split = (IT + SPLITS - 1u) / SPLITS;
    \\const uint tk0 = split * tiles_per_split;
    \\const uint tk1 = min(tk0 + tiles_per_split, IT);
    \\const uint prow = (lane & 3u) * 2u;
    \\const uint pcol = lane >> 2u;
    \\uint pos[8];
    \\pos[0] = prow * 16u + pcol;
    \\pos[1] = (prow + 1u) * 16u + pcol;
    \\pos[2] = (prow + 8u) * 16u + pcol;
    \\pos[3] = (prow + 9u) * 16u + pcol;
    \\pos[4] = prow * 16u + pcol + 8u;
    \\pos[5] = (prow + 1u) * 16u + pcol + 8u;
    \\pos[6] = (prow + 8u) * 16u + pcol + 8u;
    \\pos[7] = (prow + 9u) * 16u + pcol + 8u;
    \\const uint row0 = pos[0] >> 4u;
    \\const uint row1 = pos[1] >> 4u;
    \\const uint row2 = pos[2] >> 4u;
    \\const uint row3 = pos[3] >> 4u;
    \\float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    \\const size_t xb = (size_t)slot * (size_t)(IDIM);
    \\const size_t expert_stride = (size_t)IT * (size_t)OT * (size_t)PACKED_HW;
    \\const device uint* trellis_e = (const device uint*)(trellis + (size_t)eid * expert_stride);
    \\if (N == 64u) {
    \\for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    \\  const device uint* words = trellis_e + ((size_t)tk * (size_t)OT + ot) * 32u;
    \\  const ulong merged = ((ulong)words[(lane + 31u) & 31u] << 32) | (ulong)words[lane];
    \\  const float in0 = float(x[xb + tk * TILE + row0]);
    \\  const float in1 = float(x[xb + tk * TILE + row1]);
    \\  const float in2 = float(x[xb + tk * TILE + row2]);
    \\  const float in3 = float(x[xb + tk * TILE + row3]);
    \\  const uint sh[8] = {28u, 24u, 20u, 16u, 12u, 8u, 4u, 0u};
    \\  const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\  for (uint p = 0u; p < 4u; p++) {
    \\    const uint2 cw = uint2(uint(merged >> sh[p * 2u]), uint(merged >> sh[p * 2u + 1u])) & uint2(0xffffu);
    \\    const float2 w = exl3_decode2(cw);
    \\    acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);
    \\    acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);
    \\  }
    \\}
    \\} else {
    \\  uint w0[4];
    \\  uint w1[4];
    \\  uint shv[4];
    \\  uint frv[4];
    \\  for (uint p = 0u; p < 4u; p++) {
    \\    const exl3_win wv = exl3_pair_window(lane * 8u + p * 2u, N);
    \\    w0[p] = wv.i0;
    \\    w1[p] = wv.i1;
    \\    shv[p] = wv.sh;
    \\    frv[p] = wv.fresh;
    \\  }
    \\  for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    \\    const device uint* words = trellis_e + ((size_t)tk * (size_t)OT + ot) * PACKED_W;
    \\    const float in0 = float(x[xb + tk * TILE + row0]);
    \\    const float in1 = float(x[xb + tk * TILE + row1]);
    \\    const float in2 = float(x[xb + tk * TILE + row2]);
    \\    const float in3 = float(x[xb + tk * TILE + row3]);
    \\    const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\    for (uint p = 0u; p < 4u; p++) {
    \\      const ulong merged = ((ulong)words[w0[p]] << 32) | (ulong)words[w1[p]];
    \\      const uint funnel = uint(merged >> shv[p]);
    \\      const uint2 cw = uint2((funnel >> frv[p]) & 0xffffu, funnel & 0xffffu);
    \\      const float2 w = exl3_decode2(cw);
    \\      acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);
    \\      acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);
    \\    }
    \\  }
    \\}
    \\for (uint si = 0u; si < 8u; si++) {
    \\  partial[sg * 256u + pos[si]] = acc[si];
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (lid < 16u) {
    \\  float sum = 0.0f;
    \\  for (uint r = 0u; r < 16u; r++) {
    \\    const uint p = r * 16u + lid;
    \\    for (uint g = 0u; g < SGS; g++) {
    \\      sum += partial[g * 256u + p];
    \\    }
    \\  }
    \\  y[(size_t)slot * (size_t)(ODIM) + ot * TILE + lid] = half(sum);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
;

var indexed_coop_kernel: KernelSlots = no_kernels;

fn getIndexedCoopKernel() !mlx.mlx_fast_metal_kernel {
    const ins = [_][*:0]const u8{ "x", "trellis", "slots" };
    const outs = [_][*:0]const u8{"y"};
    return codebookKernel(&indexed_coop_kernel, "mlxserve_exl3_k4_mul1_gemv_indexed", &ins, &outs, INDEXED_COOP_SOURCE);
}

pub fn indexedGemvCoopF16(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(trellis);
    const ssh = mlx.getShape(slots);
    if ((xsh.len != 1 and xsh.len != 2) or tsh.len != 4 or ssh.len != 1) return error.BadExl3Shape;
    const in_dim = xsh[xsh.len - 1];
    const out_dim = tsh[2] * 16;
    const out_tiles = tsh[2];
    const topk = ssh[0];
    const rate = try packedRate(tsh[3]);
    if (tsh[1] * 16 != in_dim) return error.BadExl3Shape;
    const key = IndexedKey{ .in_dim = in_dim, .out_dim = out_dim, .topk = topk, .n = rate.n };
    const cfg = indexed_coop_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const out_shape = [_]c_int{ topk, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, out_tiles * 128, topk, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        indexed_coop_cfgs.put(key, c);
        break :blk c;
    };
    const inputs_arr = [_]mlx.mlx_array{ x, trellis, slots };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getIndexedCoopKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

const DOWN_FUSED_SOURCE: [:0]const u8 =
    \\threadgroup float partial[4 * 16];
    \\threadgroup half prepared[uint(IDIM)];
    \\uint ot = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\uint split = uint(threadgroup_position_in_grid.z);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\uint lane = uint(thread_index_in_simdgroup);
    \\uint lid = uint(thread_index_in_threadgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_HW = N;
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\constexpr uint SGS = 4u;
    \\constexpr uint SPLITS = 1u;
    \\const uint eid = uint(slots[slot]);
    \\const float sc = 0.08838834764831845f;
    \\const uint nblocks = uint(IDIM) / 128u;
    \\for (uint block = sg; block < nblocks; block += SGS) {
    \\  const uint base = block * 128u;
    \\  const size_t sb = (size_t)eid * (size_t)(IDIM) + base;
    \\  float4 v = float4(0.0f, 0.0f, 0.0f, 0.0f);
    \\  for (uint sp = 0u; sp < uint(NSPLIT); sp++) {
    \\    const size_t xb = ((size_t)slot * uint(NSPLIT) + sp) * (size_t)(IDIM) + base;
    \\    v += float4(float(ig[xb + lane]), float(ig[xb + lane + 32u]), float(ig[xb + lane + 64u]), float(ig[xb + lane + 96u]));
    \\  }
    \\  for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\    const float p0 = simd_shuffle_xor(v.x, bit);
    \\    const float p1 = simd_shuffle_xor(v.y, bit);
    \\    const float p2 = simd_shuffle_xor(v.z, bit);
    \\    const float p3 = simd_shuffle_xor(v.w, bit);
    \\    const bool lower = (lane & bit) == 0u;
    \\    v.x = lower ? v.x + p0 : p0 - v.x;
    \\    v.y = lower ? v.y + p1 : p1 - v.y;
    \\    v.z = lower ? v.z + p2 : p2 - v.z;
    \\    v.w = lower ? v.w + p3 : p3 - v.w;
    \\  }
    \\  float s0 = v.x + v.y;
    \\  float s1 = v.x - v.y;
    \\  float s2 = v.z + v.w;
    \\  float s3 = v.z - v.w;
    \\  const float g0 = (s0 + s2) * sc * float(svhg[sb + lane]);
    \\  const float g1 = (s1 + s3) * sc * float(svhg[sb + lane + 32u]);
    \\  const float g2 = (s0 - s2) * sc * float(svhg[sb + lane + 64u]);
    \\  const float g3 = (s1 - s3) * sc * float(svhg[sb + lane + 96u]);
    \\  v = float4(0.0f, 0.0f, 0.0f, 0.0f);
    \\  for (uint sp = 0u; sp < uint(NSPLIT); sp++) {
    \\    const size_t xbu = ((size_t)slot * uint(NSPLIT) + sp) * (size_t)(IDIM) + base;
    \\    v += float4(float(iu[xbu + lane]), float(iu[xbu + lane + 32u]), float(iu[xbu + lane + 64u]), float(iu[xbu + lane + 96u]));
    \\  }
    \\  for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\    const float p0 = simd_shuffle_xor(v.x, bit);
    \\    const float p1 = simd_shuffle_xor(v.y, bit);
    \\    const float p2 = simd_shuffle_xor(v.z, bit);
    \\    const float p3 = simd_shuffle_xor(v.w, bit);
    \\    const bool lower = (lane & bit) == 0u;
    \\    v.x = lower ? v.x + p0 : p0 - v.x;
    \\    v.y = lower ? v.y + p1 : p1 - v.y;
    \\    v.z = lower ? v.z + p2 : p2 - v.z;
    \\    v.w = lower ? v.w + p3 : p3 - v.w;
    \\  }
    \\  s0 = v.x + v.y;
    \\  s1 = v.x - v.y;
    \\  s2 = v.z + v.w;
    \\  s3 = v.z - v.w;
    \\  const float u0 = (s0 + s2) * sc * float(svhu[sb + lane]);
    \\  const float u1 = (s1 + s3) * sc * float(svhu[sb + lane + 32u]);
    \\  const float u2 = (s0 - s2) * sc * float(svhu[sb + lane + 64u]);
    \\  const float u3 = (s1 - s3) * sc * float(svhu[sb + lane + 96u]);
    \\  const float ysig0 = 1 / (1 + exp(abs(g0)));
    \\  const float ysig1 = 1 / (1 + exp(abs(g1)));
    \\  const float ysig2 = 1 / (1 + exp(abs(g2)));
    \\  const float ysig3 = 1 / (1 + exp(abs(g3)));
    \\  const float sig0 = (g0 < 0) ? ysig0 : 1 - ysig0;
    \\  const float sig1 = (g1 < 0) ? ysig1 : 1 - ysig1;
    \\  const float sig2 = (g2 < 0) ? ysig2 : 1 - ysig2;
    \\  const float sig3 = (g3 < 0) ? ysig3 : 1 - ysig3;
    \\  const float silu0 = g0 * sig0;
    \\  const float silu1 = g1 * sig1;
    \\  const float silu2 = g2 * sig2;
    \\  const float silu3 = g3 * sig3;
    \\  const float h0 = silu0 * u0;
    \\  const float h1 = silu1 * u1;
    \\  const float h2 = silu2 * u2;
    \\  const float h3 = silu3 * u3;
    \\  v = float4(h0 * float(suhd[sb + lane]), h1 * float(suhd[sb + lane + 32u]), h2 * float(suhd[sb + lane + 64u]), h3 * float(suhd[sb + lane + 96u]));
    \\  for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\    const float p0 = simd_shuffle_xor(v.x, bit);
    \\    const float p1 = simd_shuffle_xor(v.y, bit);
    \\    const float p2 = simd_shuffle_xor(v.z, bit);
    \\    const float p3 = simd_shuffle_xor(v.w, bit);
    \\    const bool lower = (lane & bit) == 0u;
    \\    v.x = lower ? v.x + p0 : p0 - v.x;
    \\    v.y = lower ? v.y + p1 : p1 - v.y;
    \\    v.z = lower ? v.z + p2 : p2 - v.z;
    \\    v.w = lower ? v.w + p3 : p3 - v.w;
    \\  }
    \\  s0 = v.x + v.y;
    \\  s1 = v.x - v.y;
    \\  s2 = v.z + v.w;
    \\  s3 = v.z - v.w;
    \\  prepared[base + lane] = half((s0 + s2) * sc);
    \\  prepared[base + lane + 32u] = half((s1 + s3) * sc);
    \\  prepared[base + lane + 64u] = half((s0 - s2) * sc);
    \\  prepared[base + lane + 96u] = half((s1 - s3) * sc);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const uint tiles_per_split = (IT + SPLITS - 1u) / SPLITS;
    \\const uint tk0 = split * tiles_per_split;
    \\const uint tk1 = min(tk0 + tiles_per_split, IT);
    \\const uint prow = (lane & 3u) * 2u;
    \\const uint pcol = lane >> 2u;
    \\const uint row0 = prow;
    \\const uint row1 = prow + 1u;
    \\const uint row2 = prow + 8u;
    \\const uint row3 = prow + 9u;
    \\float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    \\const device uint* trellis_e = (const device uint*)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW);
    \\if (N == 64u) {
    \\for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    \\  const device uint* words = trellis_e + ((size_t)tk * (size_t)OT + ot) * 32u;
    \\  const ulong merged = ((ulong)words[(lane + 31u) & 31u] << 32) | (ulong)words[lane];
    \\  const float in0 = float(prepared[tk * TILE + row0]);
    \\  const float in1 = float(prepared[tk * TILE + row1]);
    \\  const float in2 = float(prepared[tk * TILE + row2]);
    \\  const float in3 = float(prepared[tk * TILE + row3]);
    \\  const uint sh[8] = {28u, 24u, 20u, 16u, 12u, 8u, 4u, 0u};
    \\  const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\  for (uint p = 0u; p < 4u; p++) {
    \\    const uint2 cw = uint2(uint(merged >> sh[p * 2u]), uint(merged >> sh[p * 2u + 1u])) & uint2(0xffffu);
    \\    const float2 w = exl3_decode2(cw);
    \\    acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);
    \\    acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);
    \\  }
    \\}
    \\} else {
    \\  uint w0[4];
    \\  uint w1[4];
    \\  uint shv[4];
    \\  uint frv[4];
    \\  for (uint p = 0u; p < 4u; p++) {
    \\    const exl3_win wv = exl3_pair_window(lane * 8u + p * 2u, N);
    \\    w0[p] = wv.i0;
    \\    w1[p] = wv.i1;
    \\    shv[p] = wv.sh;
    \\    frv[p] = wv.fresh;
    \\  }
    \\  for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    \\    const device uint* words = trellis_e + ((size_t)tk * (size_t)OT + ot) * PACKED_W;
    \\    const float in0 = float(prepared[tk * TILE + row0]);
    \\    const float in1 = float(prepared[tk * TILE + row1]);
    \\    const float in2 = float(prepared[tk * TILE + row2]);
    \\    const float in3 = float(prepared[tk * TILE + row3]);
    \\    const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\    for (uint p = 0u; p < 4u; p++) {
    \\      const ulong merged = ((ulong)words[w0[p]] << 32) | (ulong)words[w1[p]];
    \\      const uint funnel = uint(merged >> shv[p]);
    \\      const uint2 cw = uint2((funnel >> frv[p]) & 0xffffu, funnel & 0xffffu);
    \\      const float2 w = exl3_decode2(cw);
    \\      acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);
    \\      acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);
    \\    }
    \\  }
    \\}
    \\float clo = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    \\float chi = (acc[4] + acc[5]) + (acc[6] + acc[7]);
    \\clo += simd_shuffle_xor(clo, 1u);
    \\chi += simd_shuffle_xor(chi, 1u);
    \\clo += simd_shuffle_xor(clo, 2u);
    \\chi += simd_shuffle_xor(chi, 2u);
    \\if ((lane & 3u) == 0u) {
    \\  partial[sg * 16u + pcol] = clo;
    \\  partial[sg * 16u + pcol + 8u] = chi;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (lid < 16u) {
    \\  float sum = 0.0f;
    \\  for (uint g = 0u; g < SGS; g++) {
    \\    sum += partial[g * 16u + lid];
    \\  }
    \\  y[(size_t)slot * (size_t)(ODIM) + ot * TILE + lid] = half(sum);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
;

fn downGemvFusedMid(
    s: mlx.mlx_stream,
    ig: mlx.mlx_array,
    iu: mlx.mlx_array,
    trellis: mlx.mlx_array,
    svhg: mlx.mlx_array,
    svhu: mlx.mlx_array,
    suhd: mlx.mlx_array,
    slots: mlx.mlx_array,
    in_dim: c_int,
    out_dim: c_int,
    nslots: c_int,
) !mlx.mlx_array {
    const tsh = mlx.getShape(trellis);
    const rate = try packedRate(tsh[tsh.len - 1]);
    const out_tiles = @divExact(out_dim, 16);
    const nsplit: c_int = @intCast(pairSplitCountFor(out_dim));
    const key = DownFusedKey{ .in_dim = in_dim, .out_dim = out_dim, .nslots = nslots, .nsplit = nsplit, .n = rate.n };
    const cfg = down_fused_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, out_tiles * 128, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NSPLIT", nsplit));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        down_fused_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "ig", "iu", "trellis", "svhg", "svhu", "suhd", "slots" };
    const outs = [_][*:0]const u8{"y"};
    const kernel = try codebookKernel(&down_fused_kernel, "mlxserve_exl3_k4_down_fused", &ins, &outs, DOWN_FUSED_SOURCE);
    const ov = try applyOuts(s, kernel, &.{ ig, iu, trellis, svhg, svhu, suhd, slots }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

pub fn projectIndexed(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const prepared = try prepareIndexed(s, x, suh, slots);
    defer _ = mlx.mlx_array_free(prepared);
    const inner = try indexedGemvCoopF16(s, prepared, trellis, slots);
    defer _ = mlx.mlx_array_free(inner);
    return finishIndexed(s, inner, svh, slots);
}

/// The suh scale + H128 Hadamard that feeds this GEMV is applied here, over the
/// k range this threadgroup owns, rather than in a dispatch of its own: the tile
/// loop then reads its operands from threadgroup memory instead of device.
const PAIR_GEMV_SOURCE: [:0]const u8 =
    \\threadgroup float partial[4 * 16];
    \\uint ot = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\uint split = uint(threadgroup_position_in_grid.z);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\uint lane = uint(thread_index_in_simdgroup);
    \\uint lid = uint(thread_index_in_threadgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_HW = N;
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\constexpr uint SGS = 4u;
    \\const uint tiles_per_split = (IT + uint(NSPLIT) - 1u) / uint(NSPLIT);
    \\const uint tk0 = split * tiles_per_split;
    \\const uint tk1 = min(tk0 + tiles_per_split, IT);
    \\const uint eid = uint(slots[slot]);
    \\constexpr uint KSPAN = uint(IDIM) / uint(NSPLIT);
    \\threadgroup half prepared[2u * KSPAN];
    \\{
    \\  const uint xrow = slot / uint(TOPK);
    \\  const float psc = 0.08838834764831845f;
    \\  for (uint b = sg; b < KSPAN / 128u; b += SGS) {
    \\    const uint pbase = tk0 * TILE + b * 128u;
    \\    const size_t xr = (size_t)xrow * (size_t)(IDIM) + pbase;
    \\    const size_t sr = (size_t)eid * (size_t)(IDIM) + pbase;
    \\    const float x0 = float(x[xr + lane]);
    \\    const float x1 = float(x[xr + lane + 32u]);
    \\    const float x2 = float(x[xr + lane + 64u]);
    \\    const float x3 = float(x[xr + lane + 96u]);
    \\    for (uint pj = 0u; pj < 2u; pj++) {
    \\      const device half *suh = (pj == 0u) ? suhg : suhu;
    \\      float4 v = float4(
    \\        x0 * float(suh[sr + lane]),
    \\        x1 * float(suh[sr + lane + 32u]),
    \\        x2 * float(suh[sr + lane + 64u]),
    \\        x3 * float(suh[sr + lane + 96u]));
    \\      for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\        const float p0 = simd_shuffle_xor(v.x, bit);
    \\        const float p1 = simd_shuffle_xor(v.y, bit);
    \\        const float p2 = simd_shuffle_xor(v.z, bit);
    \\        const float p3 = simd_shuffle_xor(v.w, bit);
    \\        const bool lower = (lane & bit) == 0u;
    \\        v.x = lower ? v.x + p0 : p0 - v.x;
    \\        v.y = lower ? v.y + p1 : p1 - v.y;
    \\        v.z = lower ? v.z + p2 : p2 - v.z;
    \\        v.w = lower ? v.w + p3 : p3 - v.w;
    \\      }
    \\      const float s0 = v.x + v.y;
    \\      const float s1 = v.x - v.y;
    \\      const float s2 = v.z + v.w;
    \\      const float s3 = v.z - v.w;
    \\      const uint pw = pj * KSPAN + b * 128u;
    \\      prepared[pw + lane] = half((s0 + s2) * psc);
    \\      prepared[pw + lane + 32u] = half((s1 + s3) * psc);
    \\      prepared[pw + lane + 64u] = half((s0 - s2) * psc);
    \\      prepared[pw + lane + 96u] = half((s1 - s3) * psc);
    \\    }
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const uint prow = (lane & 3u) * 2u;
    \\const uint pcol = lane >> 2u;
    \\const uint row0 = prow;
    \\const uint row1 = prow + 1u;
    \\const uint row2 = prow + 8u;
    \\const uint row3 = prow + 9u;
    \\for (uint proj = 0u; proj < 2u; proj++) {
    \\  const uint xb = proj * KSPAN;
    \\  const device ushort *trellis = (proj == 0u) ? tg : tu;
    \\  device float *y = (proj == 0u) ? yg : yu;
    \\  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    \\  const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW);
    \\  if (N == 64u) {
    \\  for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    \\    const device uint *words = trellis_e + ((size_t)tk * (size_t)OT + ot) * 32u;
    \\    const ulong merged = ((ulong)words[(lane + 31u) & 31u] << 32) | (ulong)words[lane];
    \\    const float in0 = float(prepared[xb + (tk - tk0) * TILE + row0]);
    \\    const float in1 = float(prepared[xb + (tk - tk0) * TILE + row1]);
    \\    const float in2 = float(prepared[xb + (tk - tk0) * TILE + row2]);
    \\    const float in3 = float(prepared[xb + (tk - tk0) * TILE + row3]);
    \\    const uint sh[8] = {28u, 24u, 20u, 16u, 12u, 8u, 4u, 0u};
    \\    const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\    for (uint p = 0u; p < 4u; p++) {
    \\      const uint2 cw = uint2(uint(merged >> sh[p * 2u]), uint(merged >> sh[p * 2u + 1u])) & uint2(0xffffu);
    \\      const float2 w = exl3_decode2(cw);
    \\      acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);
    \\      acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);
    \\    }
    \\  }
    \\  } else {
    \\    uint w0[4];
    \\    uint w1[4];
    \\    uint shv[4];
    \\    uint frv[4];
    \\    for (uint p = 0u; p < 4u; p++) {
    \\      const exl3_win wv = exl3_pair_window(lane * 8u + p * 2u, N);
    \\      w0[p] = wv.i0;
    \\      w1[p] = wv.i1;
    \\      shv[p] = wv.sh;
    \\      frv[p] = wv.fresh;
    \\    }
    \\    for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    \\      const device uint *words = trellis_e + ((size_t)tk * (size_t)OT + ot) * PACKED_W;
    \\      const float in0 = float(prepared[xb + (tk - tk0) * TILE + row0]);
    \\      const float in1 = float(prepared[xb + (tk - tk0) * TILE + row1]);
    \\      const float in2 = float(prepared[xb + (tk - tk0) * TILE + row2]);
    \\      const float in3 = float(prepared[xb + (tk - tk0) * TILE + row3]);
    \\      const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\      for (uint p = 0u; p < 4u; p++) {
    \\        const ulong merged = ((ulong)words[w0[p]] << 32) | (ulong)words[w1[p]];
    \\        const uint funnel = uint(merged >> shv[p]);
    \\        const uint2 cw = uint2((funnel >> frv[p]) & 0xffffu, funnel & 0xffffu);
    \\        const float2 w = exl3_decode2(cw);
    \\        acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);
    \\        acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);
    \\      }
    \\    }
    \\  }
    \\  float clo = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    \\  float chi = (acc[4] + acc[5]) + (acc[6] + acc[7]);
    \\  clo += simd_shuffle_xor(clo, 1u);
    \\  chi += simd_shuffle_xor(chi, 1u);
    \\  clo += simd_shuffle_xor(clo, 2u);
    \\  chi += simd_shuffle_xor(chi, 2u);
    \\  if ((lane & 3u) == 0u) {
    \\    partial[sg * 16u + pcol] = clo;
    \\    partial[sg * 16u + pcol + 8u] = chi;
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (lid < 16u) {
    \\    float sum = 0.0f;
    \\    for (uint g = 0u; g < SGS; g++) {
    \\      sum += partial[g * 16u + lid];
    \\    }
    \\    y[(size_t)(slot * uint(NSPLIT) + split) * (size_t)(ODIM) + ot * TILE + lid] = sum;
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
;

const MID_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint eid = uint(slots[slot]);
    \\const uint base = block * 128u;
    \\const size_t xb = (size_t)slot * (size_t)(ODIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(ODIM) + base;
    \\const float sc = 0.08838834764831845f;
    \\float4 v = float4(float(ig[xb + lane]), float(ig[xb + lane + 32u]), float(ig[xb + lane + 64u]), float(ig[xb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\float s0 = v.x + v.y;
    \\float s1 = v.x - v.y;
    \\float s2 = v.z + v.w;
    \\float s3 = v.z - v.w;
    \\const float g0 = (s0 + s2) * sc * float(svhg[sb + lane]);
    \\const float g1 = (s1 + s3) * sc * float(svhg[sb + lane + 32u]);
    \\const float g2 = (s0 - s2) * sc * float(svhg[sb + lane + 64u]);
    \\const float g3 = (s1 - s3) * sc * float(svhg[sb + lane + 96u]);
    \\v = float4(float(iu[xb + lane]), float(iu[xb + lane + 32u]), float(iu[xb + lane + 64u]), float(iu[xb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\s0 = v.x + v.y;
    \\s1 = v.x - v.y;
    \\s2 = v.z + v.w;
    \\s3 = v.z - v.w;
    \\const float u0 = (s0 + s2) * sc * float(svhu[sb + lane]);
    \\const float u1 = (s1 + s3) * sc * float(svhu[sb + lane + 32u]);
    \\const float u2 = (s0 - s2) * sc * float(svhu[sb + lane + 64u]);
    \\const float u3 = (s1 - s3) * sc * float(svhu[sb + lane + 96u]);
    \\const float ysig0 = 1 / (1 + exp(abs(g0)));
    \\const float ysig1 = 1 / (1 + exp(abs(g1)));
    \\const float ysig2 = 1 / (1 + exp(abs(g2)));
    \\const float ysig3 = 1 / (1 + exp(abs(g3)));
    \\const float sig0 = (g0 < 0) ? ysig0 : 1 - ysig0;
    \\const float sig1 = (g1 < 0) ? ysig1 : 1 - ysig1;
    \\const float sig2 = (g2 < 0) ? ysig2 : 1 - ysig2;
    \\const float sig3 = (g3 < 0) ? ysig3 : 1 - ysig3;
    \\const float silu0 = g0 * sig0;
    \\const float silu1 = g1 * sig1;
    \\const float silu2 = g2 * sig2;
    \\const float silu3 = g3 * sig3;
    \\const float h0 = silu0 * u0;
    \\const float h1 = silu1 * u1;
    \\const float h2 = silu2 * u2;
    \\const float h3 = silu3 * u3;
    \\v = float4(h0 * float(suhd[sb + lane]), h1 * float(suhd[sb + lane + 32u]), h2 * float(suhd[sb + lane + 64u]), h3 * float(suhd[sb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\s0 = v.x + v.y;
    \\s1 = v.x - v.y;
    \\s2 = v.z + v.w;
    \\s3 = v.z - v.w;
    \\yd[xb + lane] = half((s0 + s2) * sc);
    \\yd[xb + lane + 32u] = half((s1 + s3) * sc);
    \\yd[xb + lane + 64u] = half((s0 - s2) * sc);
    \\yd[xb + lane + 96u] = half((s1 - s3) * sc);
;

const REDUCE_SOURCE: [:0]const u8 =
    \\threadgroup float vals[uint(TOPK) * 128u];
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint row = uint(threadgroup_position_in_grid.y);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint base = block * 128u;
    \\const uint slot = row * uint(TOPK) + sg;
    \\const uint eid = uint(slots[slot]);
    \\const size_t xb = (size_t)slot * (size_t)(ODIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(ODIM) + base;
    \\const float scv = 0.08838834764831845f;
    \\float4 v = float4(float(inner[xb + lane]), float(inner[xb + lane + 32u]), float(inner[xb + lane + 64u]), float(inner[xb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\vals[sg * 128u + lane] = (s0 + s2) * scv * float(svh[sb + lane]);
    \\vals[sg * 128u + lane + 32u] = (s1 + s3) * scv * float(svh[sb + lane + 32u]);
    \\vals[sg * 128u + lane + 64u] = (s0 - s2) * scv * float(svh[sb + lane + 64u]);
    \\vals[sg * 128u + lane + 96u] = (s1 - s3) * scv * float(svh[sb + lane + 96u]);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (sg == 0u) {
    \\  float a0 = 0.0f;
    \\  float a1 = 0.0f;
    \\  float a2 = 0.0f;
    \\  float a3 = 0.0f;
    \\  for (uint k = 0u; k < uint(TOPK); k++) {
    \\    const float w = float(sc[row * uint(TOPK) + k]);
    \\    a0 += vals[k * 128u + lane] * w;
    \\    a1 += vals[k * 128u + lane + 32u] * w;
    \\    a2 += vals[k * 128u + lane + 64u] * w;
    \\    a3 += vals[k * 128u + lane + 96u] * w;
    \\  }
    \\  const size_t yb = (size_t)row * (size_t)(ODIM) + base;
    \\  y[yb + lane] = T(a0);
    \\  y[yb + lane + 32u] = T(a1);
    \\  y[yb + lane + 64u] = T(a2);
    \\  y[yb + lane + 96u] = T(a3);
    \\}
;

var pair_gemv_kernel: KernelSlots = no_kernels;
var mid_kernel: ?mlx.mlx_fast_metal_kernel = null;
var reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;
var down_fused_kernel: KernelSlots = no_kernels;

fn getNamedKernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8, header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header.ptr, true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = kernel;
    return kernel;
}

fn applyOuts(s: mlx.mlx_stream, kernel: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, cfg: mlx.mlx_fast_metal_kernel_config, n_out: usize) !mlx.mlx_vector_array {
    fused_dispatches += 1;
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    errdefer _ = mlx.mlx_vector_array_free(outputs_vec);
    const host_on = applyUbenchOn();
    const io = std.Io.Threaded.global_single_threaded.io();
    var sw = if (host_on) io_util.Stopwatch.init(io) else undefined;
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (host_on) {
        apply_host_ns += sw.read();
        apply_host_n += 1;
    }
    if (mlx.mlx_vector_array_size(outputs_vec) != n_out) return error.MetalKernelBadOutputCount;
    return outputs_vec;
}

fn pairGemv(s: mlx.mlx_stream, x: mlx.mlx_array, suhg: mlx.mlx_array, suhu: mlx.mlx_array, tg: mlx.mlx_array, tu: mlx.mlx_array, slots: mlx.mlx_array, in_dim: c_int, out_dim: c_int, nslots: c_int, topk: c_int) !struct { mlx.mlx_array, mlx.mlx_array } {
    const tsh = mlx.getShape(tg);
    const ush = mlx.getShape(tu);
    const rate = try packedRate(tsh[tsh.len - 1]);
    if (ush[ush.len - 1] != tsh[tsh.len - 1]) return error.BadExl3Shape;
    const out_tiles = @divExact(out_dim, 16);
    const nsplit: c_int = @intCast(pairSplitCountFor(in_dim));
    const key = PairGemvKey{ .in_dim = in_dim, .out_dim = out_dim, .nslots = nslots, .nsplit = nsplit, .topk = topk, .n = rate.n };
    const cfg = pair_gemv_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots * nsplit, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, out_tiles * 128, nslots, nsplit));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NSPLIT", nsplit));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        pair_gemv_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "x", "suhg", "suhu", "tg", "tu", "slots" };
    const outs = [_][*:0]const u8{ "yg", "yu" };
    const kernel = try codebookKernel(&pair_gemv_kernel, "mlxserve_exl3_pair_gemv", &ins, &outs, PAIR_GEMV_SOURCE);
    const ov = try applyOuts(s, kernel, &.{ x, suhg, suhu, tg, tu, slots }, cfg, 2);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b, ov, 1));
    return .{ a, b };
}

fn midSwigluPrep(s: mlx.mlx_stream, ig: mlx.mlx_array, iu: mlx.mlx_array, svhg: mlx.mlx_array, svhu: mlx.mlx_array, suhd: mlx.mlx_array, slots: mlx.mlx_array, dim: c_int, nslots: c_int) !mlx.mlx_array {
    const key = MidKey{ .dim = dim, .nslots = nslots };
    const cfg = mid_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        const blocks = @divExact(dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", dim));
        mid_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "ig", "iu", "svhg", "svhu", "suhd", "slots" };
    const outs = [_][*:0]const u8{"yd"};
    const kernel = try getNamedKernel(&mid_kernel, "mlxserve_exl3_mid_swiglu", &ins, &outs, MID_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ ig, iu, svhg, svhu, suhd, slots }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

/// One simdgroup per (row, k) slot, so the threadgroup cannot hold more slots
/// than Metal allows threads.
pub const REDUCE_MAX_TOPK: c_int = 32;

fn downFinishReduce(s: mlx.mlx_stream, inner: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, out_dim: c_int, rows: c_int, topk: c_int, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    if (topk < 1 or topk > REDUCE_MAX_TOPK) return error.Exl3TopkUnsupported;
    const key = DecodeReduceKey{ .out_dim = out_dim, .rows = rows, .topk = topk, .dtype = out_dtype };
    const cfg = reduce_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = if (rows == 1) [_]c_int{out_dim} ++ [_]c_int{0} else [_]c_int{ rows, out_dim };
        const ndim: usize = if (rows == 1) 1 else 2;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, ndim, out_dtype));
        const blocks = @divExact(out_dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * topk * blocks, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32 * topk, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", out_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        reduce_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "inner", "svh", "slots", "sc" };
    const outs = [_][*:0]const u8{"y"};
    const kernel = try getNamedKernel(&reduce_kernel, "mlxserve_exl3_down_reduce", &ins, &outs, REDUCE_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ inner, svh, slots, scores }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

fn prepareFromTokens(s: mlx.mlx_stream, x: mlx.mlx_array, suh: mlx.mlx_array, slots: mlx.mlx_array, order: mlx.mlx_array, in_dim: c_int, nslots: c_int, topk: c_int) !mlx.mlx_array {
    const key = PairPrepKey{ .in_dim = in_dim, .nslots = nslots, .topk = topk };
    const cfg = token_prep_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, in_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        const blocks = @divExact(in_dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        token_prep_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "x", "suh", "slots", "order" };
    const outs = [_][*:0]const u8{"y"};
    const kernel = try getNamedKernel(&token_prepare_kernel, "mlxserve_exl3_token_prepare", &ins, &outs, TOKEN_PREPARE_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ x, suh, slots, order }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

fn tokenReduce(s: mlx.mlx_stream, d: mlx.mlx_array, inv: mlx.mlx_array, scores: mlx.mlx_array, out_dim: c_int, rows: c_int, topk: c_int) !mlx.mlx_array {
    const key = ReduceKey{ .out_dim = out_dim, .rows = rows, .topk = topk };
    const cfg = token_reduce_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ rows, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, out_dim, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        token_reduce_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "d", "inv", "sc" };
    const outs = [_][*:0]const u8{"y"};
    const kernel = try getNamedKernel(&token_reduce_kernel, "mlxserve_exl3_token_reduce", &ins, &outs, TOKEN_REDUCE_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ d, inv, scores }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

fn pairPrepareFromTokens(s: mlx.mlx_stream, x: mlx.mlx_array, suhg: mlx.mlx_array, suhu: mlx.mlx_array, slots: mlx.mlx_array, order: mlx.mlx_array, in_dim: c_int, nslots: c_int, topk: c_int) !struct { mlx.mlx_array, mlx.mlx_array } {
    const key = PairPrepKey{ .in_dim = in_dim, .nslots = nslots, .topk = topk };
    const cfg = token_pair_prep_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, in_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        const blocks = @divExact(in_dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        token_pair_prep_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "x", "suhg", "suhu", "slots", "order" };
    const outs = [_][*:0]const u8{ "yg", "yu" };
    const kernel = try getNamedKernel(&token_pair_prepare_kernel, "mlxserve_exl3_token_pair_prepare", &ins, &outs, TOKEN_PAIR_PREPARE_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ x, suhg, suhu, slots, order }, cfg, 2);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b, ov, 1));
    return .{ a, b };
}

fn scatterSorted(s: mlx.mlx_stream, x: mlx.mlx_array, order: mlx.mlx_array, dim: c_int, nslots: c_int) !mlx.mlx_array {
    const key = ScatterKey{ .dim = dim, .nslots = nslots };
    const cfg = token_scatter_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, dim, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "DIM", dim));
        token_scatter_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "x", "order" };
    const outs = [_][*:0]const u8{"y"};
    const kernel = try getNamedKernel(&token_scatter_kernel, "mlxserve_exl3_token_scatter", &ins, &outs, TOKEN_SCATTER_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ x, order }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

pub fn moeSwigluFused(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
    out_dtype: mlx.mlx_dtype,
) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const ssh = mlx.getShape(slots);
    const tsh = mlx.getShape(gate_t);
    const nslots = ssh[0];
    const hidden: c_int = if (xsh.len == 1) xsh[0] else xsh[xsh.len - 1];
    const rows: c_int = if (xsh.len == 1) 1 else xsh[0];
    const topk = @divExact(nslots, rows);
    const inter = tsh[2] * 16;
    const inners = try pairGemv(s, x, gate_suh, up_suh, gate_t, up_t, slots, hidden, inter, nslots, topk);
    defer _ = mlx.mlx_array_free(inners[0]);
    defer _ = mlx.mlx_array_free(inners[1]);
    try ubenchEval(inners[0], "pair_gemv");
    if (exl3UbenchOn()) try mlx.check(mlx.mlx_array_eval(inners[1]));
    const maxabs = swigluMaxabsOn() and swiglu_maxabs_dumped < 96;
    if (maxabs) {
        try dumpAbsMax(s, inners[0], "ig");
        try dumpAbsMax(s, inners[1], "iu");
    }
    const down_inner = try downGemvFusedMid(s, inners[0], inners[1], down_t, gate_svh, up_svh, down_suh, slots, inter, hidden, nslots);
    defer _ = mlx.mlx_array_free(down_inner);
    try ubenchEval(down_inner, "down_gemv");
    // The SwiGLU product and the down inner plane are the f16 stores that can
    // saturate: a non-finite here is the mid overflow, not a decode fault.
    if (maxabs) {
        try dumpAbsMax(s, down_inner, "down_inner");
        swiglu_maxabs_dumped += 1;
    }
    const out = try downFinishReduce(s, down_inner, down_svh, slots, scores, hidden, rows, topk, out_dtype);
    errdefer _ = mlx.mlx_array_free(out);
    try ubenchEval(out, "reduce");
    if (applyUbenchOn()) {
        apply_host_layers += 1;
        if (apply_host_layers == 48) {
            if (apply_host_dumps < 8) {
                const ms = @as(f64, @floatFromInt(apply_host_ns)) / 1e6;
                const n: f64 = @floatFromInt(@max(apply_host_n, 1));
                log.info("[exl3-apply] host {d:.3} ms n={d} us/apply={d:.1}\n", .{
                    ms,
                    apply_host_n,
                    (ms * 1e3) / n,
                });
                apply_host_dumps += 1;
            }
            apply_host_ns = 0;
            apply_host_n = 0;
            apply_host_layers = 0;
            if (apply_host_dumps >= 8) apply_ubench_env = false;
        }
    }
    return out;
}

pub fn moeSwigluIndexed(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
) !mlx.mlx_array {
    const g = try projectIndexed(s, x, gate_t, gate_suh, gate_svh, slots);
    defer _ = mlx.mlx_array_free(g);
    const u = try projectIndexed(s, x, up_t, up_suh, up_svh, slots);
    defer _ = mlx.mlx_array_free(u);
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, g, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, g, sig, s));
    var h = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_multiply(&h, silu, u, s));
    const d = try projectIndexed(s, h, down_t, down_suh, down_svh, slots);
    defer _ = mlx.mlx_array_free(d);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_astype(&sc, scores, .float16, s));
    var sc2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc2);
    try mlx.check(mlx.mlx_expand_dims(&sc2, sc, -1, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, d, sc2, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_sum_axis(&out, weighted, 0, false, s));
    return out;
}

fn repeatRows(s: mlx.mlx_stream, x: mlx.mlx_array, rows: c_int, topk: c_int) !mlx.mlx_array {
    var ar = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ar);
    try mlx.check(mlx.mlx_arange(&ar, 0, @floatFromInt(rows), 1, .int32, s));
    var col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(col);
    try mlx.check(mlx.mlx_reshape(&col, ar, &[_]c_int{ rows, 1 }, 2, s));
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    const shape = [_]c_int{ rows, topk };
    try mlx.check(mlx.mlx_broadcast_to(&wide, col, &shape, 2, s));
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_reshape(&idx, wide, &[_]c_int{rows * topk}, 1, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_take_axis(&out, x, idx, 0, s));
    return out;
}

fn projectSorted(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots, 0, s));
    var sorted_slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_slots);
    try mlx.check(mlx.mlx_take_axis(&sorted_slots, slots, order, 0, s));
    var sorted_x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_x);
    try mlx.check(mlx.mlx_take_axis(&sorted_x, x, order, 0, s));
    const prepared = try prepareIndexed(s, sorted_x, suh, sorted_slots);
    defer _ = mlx.mlx_array_free(prepared);
    const inner = try innerGemmSorted(s, prepared, trellis, sorted_slots);
    defer _ = mlx.mlx_array_free(inner);
    const finished = try finishIndexed(s, inner, svh, sorted_slots);
    defer _ = mlx.mlx_array_free(finished);
    var inv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv);
    try mlx.check(mlx.mlx_argsort_axis(&inv, order, 0, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_take_axis(&out, finished, inv, 0, s));
    return out;
}

fn projectSortedWithRuns(
    s: mlx.mlx_stream,
    x_sorted: mlx.mlx_array,
    trellis: mlx.mlx_array,
    suh: mlx.mlx_array,
    svh: mlx.mlx_array,
    slots_sorted: mlx.mlx_array,
) !mlx.mlx_array {
    const prepared = try prepareIndexed(s, x_sorted, suh, slots_sorted);
    defer _ = mlx.mlx_array_free(prepared);
    const inner = try innerGemmSorted(s, prepared, trellis, slots_sorted);
    defer _ = mlx.mlx_array_free(inner);
    return finishIndexed(s, inner, svh, slots_sorted);
}

pub fn moePrefill(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
    topk: c_int,
) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const rows = xsh[0];
    const hidden = xsh[1];
    const nslots = rows * topk;
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots, 0, s));
    var order_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order_i);
    try mlx.check(mlx.mlx_astype(&order_i, order, .int32, s));
    var sorted_slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_slots);
    try mlx.check(mlx.mlx_take_axis(&sorted_slots, slots, order, 0, s));
    try ubenchEval(sorted_slots, "sort");
    const prep = try pairPrepareFromTokens(s, x, gate_suh, up_suh, sorted_slots, order_i, hidden, nslots, topk);
    defer _ = mlx.mlx_array_free(prep[0]);
    defer _ = mlx.mlx_array_free(prep[1]);
    try ubenchEval(prep[0], "token_prepare");
    if (exl3UbenchOn()) try mlx.check(mlx.mlx_array_eval(prep[1]));
    const win = gemmWindowRows();
    const aligned = gemmWindowAligned();
    const tab = try gemmWindowTable(s, sorted_slots, nslots, win, aligned);
    defer _ = mlx.mlx_array_free(tab.starts);
    defer _ = mlx.mlx_array_free(tab.nlives);
    const g_inner = try innerGemmSortedTable(s, prep[0], gate_t, sorted_slots, win, aligned, tab);
    defer _ = mlx.mlx_array_free(g_inner);
    try ubenchEval(g_inner, "gemm_gate");
    const u_inner = try innerGemmSortedTable(s, prep[1], up_t, sorted_slots, win, aligned, tab);
    defer _ = mlx.mlx_array_free(u_inner);
    try ubenchEval(u_inner, "gemm_up");
    const down_x = try midSwigluPrep(s, g_inner, u_inner, gate_svh, up_svh, down_suh, sorted_slots, mlx.getShape(g_inner)[1], nslots);
    defer _ = mlx.mlx_array_free(down_x);
    try ubenchEval(down_x, "mid");
    const d_inner = try innerGemmSortedTable(s, down_x, down_t, sorted_slots, win, aligned, tab);
    defer _ = mlx.mlx_array_free(d_inner);
    try ubenchEval(d_inner, "gemm_down");
    const d_unsorted = try scatterSorted(s, d_inner, order_i, hidden, nslots);
    defer _ = mlx.mlx_array_free(d_unsorted);
    const out = try downFinishReduce(s, d_unsorted, down_svh, slots, scores, hidden, rows, topk, mlx.mlx_array_dtype(x));
    try ubenchEval(out, "token_reduce");
    return out;
}

pub fn prefillDecodeGatherMm(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_pub: mlx.mlx_array,
    up_pub: mlx.mlx_array,
    down_pub: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
    rows: c_int,
    hidden: c_int,
    inter: c_int,
    topk: c_int,
) !mlx.mlx_array {
    const no_idx = mlx.mlx_array{ .ctx = null };
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_reshape(&x4, x, &[_]c_int{ rows, 1, 1, hidden }, 4, s));
    var g4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g4);
    try mlx.check(mlx.mlx_gather_mm(&g4, x4, gate_pub, no_idx, slots, false, s));
    var up4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up4);
    try mlx.check(mlx.mlx_gather_mm(&up4, x4, up_pub, no_idx, slots, false, s));
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_squeeze(&g, g4, s));
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_squeeze(&u, up4, s));
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, g, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, g, sig, s));
    var h = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_multiply(&h, silu, u, s));
    var h4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h4);
    try mlx.check(mlx.mlx_reshape(&h4, h, &[_]c_int{ rows, topk, 1, inter }, 4, s));
    var d4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d4);
    try mlx.check(mlx.mlx_gather_mm(&d4, h4, down_pub, no_idx, slots, false, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_squeeze(&d, d4, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_astype(&sc, scores, mlx.mlx_array_dtype(d), s));
    var sc3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc3);
    try mlx.check(mlx.mlx_reshape(&sc3, sc, &[_]c_int{ rows, topk, 1 }, 3, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, d, sc3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_sum_axis(&out, weighted, 1, false, s));
    return out;
}

pub fn innerGemvF16(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(trellis);
    if (xsh.len != 1 or tsh.len != 3) return error.BadExl3Shape;
    const in_dim = xsh[0];
    const out_dim = tsh[1] * 16;
    const rate = try packedRate(tsh[2]);
    if (tsh[0] * 16 != in_dim) return error.BadExl3Shape;
    const cfg = try gemvConfig(in_dim, out_dim, rate);
    const inputs_arr = [_]mlx.mlx_array{ x, trellis };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getGemvKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (!gemv_engaged) {
        gemv_engaged = true;
        log.info("[expert-exl3] engaged in={d} out={d}\n", .{ in_dim, out_dim });
    }
    return out;
}

pub fn moeSwigluHost(
    alloc: std.mem.Allocator,
    x: []const f32,
    gate_t: []const u16,
    gate_suh: []const u16,
    gate_svh: []const u16,
    up_t: []const u16,
    up_suh: []const u16,
    up_svh: []const u16,
    down_t: []const u16,
    down_suh: []const u16,
    down_svh: []const u16,
    slots: []const u32,
    weights: []const f32,
    hidden: usize,
    inter: usize,
    packed_n: usize,
    in_tiles_h: usize,
    out_tiles_i: usize,
    dec: exl3.Decode,
) ![]f32 {
    const kbits = exl3.kFromPackedDim(packed_n) orelse return error.BadExl3Shape;
    const topk = slots.len;
    const y = try alloc.alloc(f32, hidden);
    @memset(y, 0);
    const transformed = try alloc.alloc(f32, hidden);
    defer alloc.free(transformed);
    const inner = try alloc.alloc(f32, @max(hidden, inter));
    defer alloc.free(inner);
    const gate_y = try alloc.alloc(f32, inter);
    defer alloc.free(gate_y);
    const up_y = try alloc.alloc(f32, inter);
    defer alloc.free(up_y);
    const h = try alloc.alloc(f32, inter);
    defer alloc.free(h);
    const down_y = try alloc.alloc(f32, hidden);
    defer alloc.free(down_y);
    const tstride_gu = in_tiles_h * out_tiles_i * packed_n;
    const tstride_d = out_tiles_i * in_tiles_h * packed_n;
    for (0..topk) |k| {
        const e = slots[k];
        const g_off = e * tstride_gu;
        const u_off = e * tstride_gu;
        const d_off = e * tstride_d;
        exl3.project(x, gate_t[g_off..][0..tstride_gu], gate_suh[e * hidden ..][0..hidden], gate_svh[e * inter ..][0..inter], hidden, inter, kbits, dec, transformed, inner[0..inter], gate_y);
        exl3.project(x, up_t[u_off..][0..tstride_gu], up_suh[e * hidden ..][0..hidden], up_svh[e * inter ..][0..inter], hidden, inter, kbits, dec, transformed, inner[0..inter], up_y);
        for (0..inter) |i| {
            const g = gate_y[i];
            h[i] = (g / (1.0 + @exp(-g))) * up_y[i];
        }
        exl3.project(h, down_t[d_off..][0..tstride_d], down_suh[e * inter ..][0..inter], down_svh[e * hidden ..][0..hidden], inter, hidden, kbits, dec, inner[0..inter], transformed, down_y);
        const w = weights[k];
        for (0..hidden) |i| y[i] += w * down_y[i];
    }
    return y;
}

test "exl3 packedRate reads n from the last dim and refuses outside K2..K4" {
    const t = std.testing;
    try t.expectEqual(@as(u32, 32), (try packedRate(32)).n);
    try t.expectEqual(@as(u32, 40), (try packedRate(40)).n);
    try t.expectEqual(@as(u32, 44), (try packedRate(44)).n);
    try t.expectEqual(@as(u32, 48), (try packedRate(48)).n);
    try t.expectEqual(@as(u32, 64), (try packedRate(64)).n);
    try t.expectError(error.BadExl3Shape, packedRate(16));
    try t.expectError(error.BadExl3Shape, packedRate(41));
    try t.expectError(error.BadExl3Shape, packedRate(80));
}

/// The fixture is `align(2)` so the u16 view below is a cast the caller has
/// already paid for: a byte-aligned blob is a compile error, not a Debug panic.
fn metalInnerGemvFixture(fixture: []align(2) const u8, rate: exl3.Rate, dec: exl3.Decode) !void {
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t_off: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t_end: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, data[t_off..t_end]));
    var x: [128]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    for (&x) |*v| v.* = rnd.float(f32) * 2 - 1;
    const xf16 = try alloc.alloc(u16, 128);
    for (x, xf16) |v, *b| b.* = exl3.f32ToF16Bits(v);
    const xf = try alloc.alloc(f32, 128);
    for (xf16, xf) |b, *v| v.* = exl3.f16BitsToF32(b);
    const x_arr = mlx.mlx_array_new_data(xf16.ptr, &[_]c_int{128}, 1, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(trellis_bits.ptr, &[_]c_int{ 8, 8, @intCast(rate.halfwords()) }, 3, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const got = try innerGemvF16(s, x_arr, tr_arr);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    var host: [128]f32 = undefined;
    exl3.innerGemv(trellis_bits, xf, 128, 128, rate, dec, &host);
    for (0..128) |i| {
        const bits: u16 = @bitCast(src[i]);
        try t.expectEqual(exl3.f32ToF16Bits(host[i]), bits);
    }
}

test "exl3 K3 Metal inner GEMV matches the host tile decode" {
    try metalInnerGemvFixture(exl3.fixtures.k3, exl3.Rate.fromK(3), .mul1);
}

test "exl3 K2 Metal inner GEMV matches the host tile decode" {
    try metalInnerGemvFixture(exl3.fixtures.k2, exl3.Rate.fromK(2), .mul1);
}

test "exl3 K2.5 TINY Metal inner GEMV matches the host tile decode" {
    try metalInnerGemvFixture(exl3.fixtures.k2p5_tiny, .{ .n = 40 }, .tiny);
}

test "exl3 K3 TINY Metal inner GEMV matches the host tile decode" {
    try metalInnerGemvFixture(exl3.fixtures.k3_tiny, .{ .n = 48 }, .tiny);
}

// A w16 bitstream is a valid w12 bitstream — only the value each window decodes
// to changes — so the existing fixtures are the narrowed packs too, scored
// against the host reference under the same width.
test "exl3 Metal inner GEMV matches the host tile decode at a narrowed codeword window" {
    try metalInnerGemvFixture(exl3.fixtures.k2p5_tiny, .{ .n = 40 }, .{ .codebook = .tiny, .window = .w12 });
    try metalInnerGemvFixture(exl3.fixtures.k2p5_tiny, .{ .n = 40 }, .{ .codebook = .tiny, .window = .w14 });
    try metalInnerGemvFixture(exl3.fixtures.k4, exl3.Rate.fromK(4), .{ .codebook = .mul1, .window = .w12 });
}

fn indexedParityStats(rate: exl3.Rate, in_dim: usize, out_dim: usize, e: usize, topk: usize, seed: u64, dec: exl3.Decode, mutate: Exl3Mutation) !Exl3GemmParityStats {
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const packed_n = rate.halfwords();
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const tile_n = in_tiles * out_tiles * packed_n;
    const stacked = try alloc.alloc(u16, e * tile_n);
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    for (stacked) |*v| v.* = @truncate(rnd.int(u32));
    const xh = try alloc.alloc(u16, topk * in_dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % e);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(topk), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(e), @intCast(in_tiles), @intCast(out_tiles), @intCast(packed_n) }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const got = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    if (mutate == .codeword) stacked[tile_n / 2] ^= 0x40;
    return measureInnerGemmParity(alloc, s, src[0 .. topk * out_dim], xh, slots_h, stacked, in_dim, out_dim, rate, mutatedDecode(dec, mutate));
}

fn indexedParity(rate: exl3.Rate, in_dim: usize, out_dim: usize, e: usize, topk: usize, seed: u64, dec: exl3.Decode) !void {
    try reportGemmParity(try indexedParityStats(rate, in_dim, out_dim, e, topk, seed, dec, .none));
}

test "exl3 K3 cooperative indexed GEMV matches host MUL1 tile decode" {
    try indexedParity(exl3.Rate.fromK(3), 128, 128, 4, 10, 17, .mul1);
}

test "exl3 K2 cooperative indexed GEMV matches host MUL1 tile decode" {
    try indexedParity(exl3.Rate.fromK(2), 128, 128, 4, 10, 19, .mul1);
}

test "exl3 K3 Metal inner GEMV matches host on production shape" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const in_dim: usize = 2560;
    const out_dim: usize = 640;
    const k: u32 = 3;
    const packed_n = exl3.packedHalfwords(k);
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const trellis = try alloc.alloc(u16, in_tiles * out_tiles * packed_n);
    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    for (trellis) |*v| v.* = @truncate(rnd.int(u32));
    const xf16 = try alloc.alloc(u16, in_dim);
    for (xf16) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const x_arr = mlx.mlx_array_new_data(xf16.ptr, &[_]c_int{@intCast(in_dim)}, 1, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(trellis.ptr, &[_]c_int{ @intCast(in_tiles), @intCast(out_tiles), @intCast(packed_n) }, 3, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const got = try innerGemvF16(s, x_arr, tr_arr);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try expectInnerGemmParity(alloc, s, src[0..out_dim], xf16, &.{0}, trellis, in_dim, out_dim, exl3.Rate.fromK(k), .mul1);
}

test "exl3 K3 cooperative indexed GEMV matches host MUL1 on production shape" {
    try indexedParity(exl3.Rate.fromK(3), 2560, 640, 4, 10, 23, .mul1);
}

test "exl3 K2 cooperative indexed GEMV matches host MUL1 on production shape" {
    try indexedParity(exl3.Rate.fromK(2), 2560, 640, 4, 10, 29, .mul1);
}

test "exl3 K4 Metal inner GEMV matches the host tile decode" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const inner_meta = parsed.value.object.get("inner").?.object;
    const t_off: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t_end: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const i_off: usize = @intCast(inner_meta.get("data_offsets").?.array.items[0].integer);
    const i_end: usize = @intCast(inner_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t_off..t_end]);
    const inner_bits = std.mem.bytesAsSlice(u16, data[i_off..i_end]);
    var x: [128]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    for (&x) |*v| v.* = rnd.float(f32) * 2 - 1;
    const xf16 = try alloc.alloc(u16, 128);
    for (x, xf16) |v, *b| b.* = exl3.f32ToF16Bits(v);
    const x_arr = mlx.mlx_array_new_data(xf16.ptr, &[_]c_int{128}, 1, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(trellis_bits.ptr, &[_]c_int{ 8, 8, 64 }, 3, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const got = try innerGemvF16(s, x_arr, tr_arr);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    var host: [128]f32 = undefined;
    @memset(&host, 0);
    for (0..128) |o| {
        var acc: f32 = 0;
        for (0..128) |i| acc += exl3.f16BitsToF32(xf16[i]) * exl3.f16BitsToF32(inner_bits[i * 128 + o]);
        host[o] = exl3.f16BitsToF32(exl3.f32ToF16Bits(acc));
    }
    for (0..128) |i| {
        const bits: u16 = @bitCast(src[i]);
        try t.expectEqual(exl3.f32ToF16Bits(host[i]), bits);
    }
}

test "exl3 verify group union unique vs assignment count" {
    const t = std.testing;
    var eids: [20]u32 = undefined;
    var i: usize = 0;
    while (i < 10) : (i += 1) eids[i] = @intCast(i);
    i = 0;
    while (i < 10) : (i += 1) eids[10 + i] = @intCast(100 + i);
    try t.expectEqual(@as(u32, 20), unionUnique(&eids));
    i = 0;
    while (i < 10) : (i += 1) eids[10 + i] = @intCast(i);
    try t.expectEqual(@as(u32, 10), unionUnique(&eids));
    i = 0;
    while (i < 10) : (i += 1) eids[10 + i] = @intCast(7 + i);
    try t.expectEqual(@as(u32, 17), unionUnique(&eids));
    var counts: [512]u32 = @splat(0);
    const u = unionMultiplicity(&eids, &counts);
    try t.expectEqual(@as(u32, 17), u);
    try t.expectEqual(@as(u32, 1), counts[0]);
    try t.expectEqual(@as(u32, 2), counts[7]);
    try t.expectEqual(@as(u32, 2), counts[9]);
    try t.expectEqual(@as(u32, 1), counts[16]);
}

test "exl3 buildRuns groups sorted expert ids" {
    const t = std.testing;
    const ids = [_]u32{ 3, 3, 3, 7, 7, 1 };
    const runs = try buildRuns(t.allocator, &ids);
    defer t.allocator.free(runs.start);
    defer t.allocator.free(runs.len);
    defer t.allocator.free(runs.eid);
    try t.expectEqual(@as(u32, 3), runs.n);
    try t.expectEqual(@as(u32, 0), runs.start[0]);
    try t.expectEqual(@as(u32, 3), runs.len[0]);
    try t.expectEqual(@as(u32, 3), runs.eid[0]);
    try t.expectEqual(@as(u32, 3), runs.start[1]);
    try t.expectEqual(@as(u32, 2), runs.len[1]);
    try t.expectEqual(@as(u32, 7), runs.eid[1]);
    try t.expectEqual(@as(u32, 5), runs.start[2]);
    try t.expectEqual(@as(u32, 1), runs.len[2]);
    try t.expectEqual(@as(u32, 1), runs.eid[2]);
}

test "exl3 row count is the leading dims, not the activation width" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 1), rowsOfShape(&[_]c_int{2560}));
    try t.expectEqual(@as(usize, 2), rowsOfShape(&[_]c_int{ 2, 2560 }));
    try t.expectEqual(@as(usize, 6), rowsOfShape(&[_]c_int{ 2, 3, 2560 }));
    try t.expect(!usesPrefillArm(rowsOfShape(&[_]c_int{ 16, 1, 2560 })));
    try t.expect(usesPrefillArm(rowsOfShape(&[_]c_int{ 17, 1, 2560 })));
    try t.expect(!usesPrefillArm(rowsOfShape(&[_]c_int{ 4, 2560 })));
}

test "exl3 prefill arm is used only above 16 rows" {
    const t = std.testing;
    try t.expect(!usesPrefillArm(1));
    try t.expect(!usesPrefillArm(2));
    try t.expect(!usesPrefillArm(16));
    try t.expect(usesPrefillArm(17));
    try t.expect(usesPrefillArm(512));
}

test "exl3 512-row prefill: decode-to-f16 gather_mm vs rows kernel" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const suh_meta = parsed.value.object.get("suh").?.object;
    const svh_meta = parsed.value.object.get("svh").?.object;
    const pub_meta = parsed.value.object.get("public").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(svh_meta.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(svh_meta.get("data_offsets").?.array.items[1].integer);
    const p0: usize = @intCast(pub_meta.get("data_offsets").?.array.items[0].integer);
    const p1: usize = @intCast(pub_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const pub_bits = std.mem.bytesAsSlice(u16, data[p0..p1]);
    const E: c_int = 512;
    const topk: c_int = 10;
    const dim: c_int = 128;
    const stacked_t = try alloc.alloc(u16, @intCast(E * 8 * 8 * 64));
    const stacked_suh = try alloc.alloc(u16, @intCast(E * dim));
    const stacked_svh = try alloc.alloc(u16, @intCast(E * dim));
    const stacked_pub = try alloc.alloc(u16, @intCast(E * dim * dim));
    var e_i: c_int = 0;
    while (e_i < E) : (e_i += 1) {
        const tb: usize = @intCast(e_i);
        @memcpy(stacked_t[tb * trellis_bits.len ..][0..trellis_bits.len], trellis_bits);
        @memcpy(stacked_suh[tb * suh_bits.len ..][0..suh_bits.len], suh_bits);
        @memcpy(stacked_svh[tb * svh_bits.len ..][0..svh_bits.len], svh_bits);
        @memcpy(stacked_pub[tb * pub_bits.len ..][0..pub_bits.len], pub_bits);
    }
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ E, 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(stacked_svh.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const pub_a = mlx.mlx_array_new_data(stacked_pub.ptr, &[_]c_int{ E, dim, dim }, 3, .float16);
    defer _ = mlx.mlx_array_free(pub_a);
    var w_oi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_oi);
    try mlx.check(mlx.mlx_transpose_axes(&w_oi, pub_a, &[_]c_int{ 0, 2, 1 }, 3, s));
    var w_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_c);
    try mlx.check(mlx.mlx_contiguous(&w_c, w_oi, false, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, w_c, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var wsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsc);
    var wbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    for ([2]c_int{ 512, 2048 }) |R| {
        const xh = try alloc.alloc(u16, @intCast(R * dim));
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
        const slots_h = try alloc.alloc(u32, @intCast(R * topk));
        for (slots_h) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
        var max_e: u32 = 0;
        for (slots_h) |v| if (v > max_e) {
            max_e = v;
        };
        try t.expect(max_e >= 400);
        const scores_h = try alloc.alloc(f32, @intCast(R * topk));
        for (scores_h) |*v| v.* = 0.5;
        const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ R, dim }, 2, .float16);
        defer _ = mlx.mlx_array_free(x_arr);
        const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{R * topk}, 1, .uint32);
        defer _ = mlx.mlx_array_free(slots);
        const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{R * topk}, 1, .float32);
        defer _ = mlx.mlx_array_free(scores);
        const warm = try moePrefill(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, topk);
        try mlx.check(mlx.mlx_array_eval(warm));
        _ = mlx.mlx_array_free(warm);
        var t_rows = io_util.Stopwatch.init(t.io);
        var it: usize = 0;
        while (it < 8) : (it += 1) {
            const rows_out = try moePrefill(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, topk);
            try mlx.check(mlx.mlx_array_eval(rows_out));
            _ = mlx.mlx_array_free(rows_out);
        }
        const rows_ns = t_rows.read() / 8;
        const n_tok: c_int = R * topk;
        const xr = try repeatRows(s, x_arr, R, topk);
        defer _ = mlx.mlx_array_free(xr);
        var xrep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xrep);
        try mlx.check(mlx.mlx_reshape(&xrep, xr, &[_]c_int{ n_tok, 1, dim }, 3, s));
        var slots_i = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(slots_i);
        try mlx.check(mlx.mlx_astype(&slots_i, slots, .int32, s));
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try mlx.check(mlx.mlx_argsort_axis(&order, slots_i, 0, s));
        var sorted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted);
        try mlx.check(mlx.mlx_take_axis(&sorted, slots_i, order, 0, s));
        const no_idx = mlx.mlx_array{ .ctx = null };
        var qmm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qmm);
        try mlx.check(mlx.mlx_gather_qmm(&qmm, xrep, wq, wsc, wbi, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
        try mlx.check(mlx.mlx_array_eval(qmm));
        var t_q = io_util.Stopwatch.init(t.io);
        it = 0;
        while (it < 8) : (it += 1) {
            var qmm2 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_gather_qmm(&qmm2, xrep, wq, wsc, wbi, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
            try mlx.check(mlx.mlx_array_eval(qmm2));
            _ = mlx.mlx_array_free(qmm2);
        }
        const qmm_ns = t_q.read() / 8;
        const ratio_x100: u64 = if (qmm_ns == 0) 0 else (rows_ns * 100) / (qmm_ns * 3);
        benchPrint("exl3 C={d} E=512 H=128 topk=10: sorted-gemm-layer {d} us  affine-gather_qmm-one-proj {d} us  ratio-vs-3x-qmm {d}/100\n", .{
            R,
            rows_ns / 1000,
            qmm_ns / 1000,
            ratio_x100,
        });
        try t.expect(qmm_ns > 0);
    }
}

test "exl3 512-row production-shape sorted gemm vs affine gather_qmm" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: c_int = 512;
    const topk: c_int = 10;
    const H: c_int = 2560;
    const I: c_int = 640;
    const tr_n: usize = @intCast(E * (H / 16) * (I / 16) * 64);
    const tr_g = try alloc.alloc(u16, tr_n);
    const tr_d = try alloc.alloc(u16, @intCast(E * (I / 16) * (H / 16) * 64));
    const suh_g = try alloc.alloc(u16, @intCast(E * H));
    const svh_g = try alloc.alloc(u16, @intCast(E * I));
    const suh_d = try alloc.alloc(u16, @intCast(E * I));
    const svh_d = try alloc.alloc(u16, @intCast(E * H));
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    for (tr_g) |*v| v.* = @truncate(rnd.int(u32));
    for (tr_d) |*v| v.* = @truncate(rnd.int(u32));
    for (suh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (suh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    const trg = mlx.mlx_array_new_data(tr_g.ptr, &[_]c_int{ E, H / 16, I / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trg);
    const trd = mlx.mlx_array_new_data(tr_d.ptr, &[_]c_int{ E, I / 16, H / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trd);
    const sugh = mlx.mlx_array_new_data(suh_g.ptr, &[_]c_int{ E, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(sugh);
    const svgi = mlx.mlx_array_new_data(svh_g.ptr, &[_]c_int{ E, I }, 2, .float16);
    defer _ = mlx.mlx_array_free(svgi);
    const sudi = mlx.mlx_array_new_data(suh_d.ptr, &[_]c_int{ E, I }, 2, .float16);
    defer _ = mlx.mlx_array_free(sudi);
    const svdh = mlx.mlx_array_new_data(svh_d.ptr, &[_]c_int{ E, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(svdh);
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_random_normal(&dense, &[_]c_int{ E, I, H }, 3, .float16, 0, 1, .{ .ctx = null }, s));
    var w_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_c);
    try mlx.check(mlx.mlx_contiguous(&w_c, dense, false, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, w_c, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var wsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsc);
    var wbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));
    for ([2]c_int{ 512, 2048 }) |R| {
        const xh = try alloc.alloc(u16, @intCast(R * H));
        const slots_h = try alloc.alloc(u32, @intCast(R * topk));
        const scores_h = try alloc.alloc(f32, @intCast(R * topk));
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
        for (slots_h) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
        var max_e: u32 = 0;
        for (slots_h) |v| if (v > max_e) {
            max_e = v;
        };
        try t.expect(max_e >= 400);
        for (scores_h) |*v| v.* = 0.5;
        const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ R, H }, 2, .float16);
        defer _ = mlx.mlx_array_free(x_arr);
        const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{R * topk}, 1, .uint32);
        defer _ = mlx.mlx_array_free(slots);
        const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{R * topk}, 1, .float32);
        defer _ = mlx.mlx_array_free(scores);
        const warm = try moePrefill(s, x_arr, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots, scores, topk);
        try mlx.check(mlx.mlx_array_eval(warm));
        _ = mlx.mlx_array_free(warm);
        var t_g = io_util.Stopwatch.init(t.io);
        var it: usize = 0;
        while (it < 3) : (it += 1) {
            const out = try moePrefill(s, x_arr, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots, scores, topk);
            try mlx.check(mlx.mlx_array_eval(out));
            _ = mlx.mlx_array_free(out);
        }
        const gemm_ns = t_g.read() / 3;
        const xr = try repeatRows(s, x_arr, R, topk);
        defer _ = mlx.mlx_array_free(xr);
        var xrep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xrep);
        try mlx.check(mlx.mlx_reshape(&xrep, xr, &[_]c_int{ R * topk, 1, H }, 3, s));
        var slots_i = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(slots_i);
        try mlx.check(mlx.mlx_astype(&slots_i, slots, .int32, s));
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try mlx.check(mlx.mlx_argsort_axis(&order, slots_i, 0, s));
        var sorted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted);
        try mlx.check(mlx.mlx_take_axis(&sorted, slots_i, order, 0, s));
        const no_idx = mlx.mlx_array{ .ctx = null };
        var q0 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q0);
        try mlx.check(mlx.mlx_gather_qmm(&q0, xrep, wq, wsc, wbi, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
        try mlx.check(mlx.mlx_array_eval(q0));
        var t_q = io_util.Stopwatch.init(t.io);
        it = 0;
        while (it < 3) : (it += 1) {
            var q = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_gather_qmm(&q, xrep, wq, wsc, wbi, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
            try mlx.check(mlx.mlx_array_eval(q));
            _ = mlx.mlx_array_free(q);
        }
        const qmm_ns = t_q.read() / 3;
        const ratio_x100: u64 = if (qmm_ns == 0) 0 else (gemm_ns * 100) / (qmm_ns * 3);
        benchPrint("exl3 C={d} E=512 H=2560 I=640 topk=10: sorted-gemm-layer {d} us  affine-gather_qmm-one-proj {d} us  ratio-vs-3x-qmm {d}/100\n", .{
            R,
            gemm_ns / 1000,
            qmm_ns / 1000,
            ratio_x100,
        });
        try t.expect(qmm_ns > 0);
    }
}

test "exl3 K4 cooperative indexed GEMV matches host MUL1 tile decode" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t_off: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t_end: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t_off..t_end]);
    const E: usize = 4;
    const dim: usize = 128;
    const topk: usize = 10;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(17);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, topk * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(topk), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const got = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try expectInnerGemmParity(alloc, s, src[0 .. topk * dim], xh, slots_h, stacked, dim, dim, exl3.Rate.fromK(4), .mul1);
}

/// Two renderings of the same chain agree to `bar` in relative RMS. The chains
/// differ in where they round to f16, so the bar is an envelope, never bytes.
fn expectRelRms(got: []const f16, want: []const f16, bar: f64) !void {
    var ss: f64 = 0;
    var ref: f64 = 0;
    for (got, want) |g, w| {
        const a: f64 = @floatCast(g);
        const b: f64 = @floatCast(w);
        if (!std.math.isFinite(a) or !std.math.isFinite(b)) return error.TestExpectedEqual;
        ss += (a - b) * (a - b);
        ref += b * b;
    }
    const rel = @sqrt(ss / @max(ref, 1e-20));
    if (rel < bar) return;
    std.debug.print("exl3 rel_rms {d:.6} over bar {d:.6}\n", .{ rel, bar });
    return error.TestExpectedEqual;
}

/// What a GEMM/GEMV arm's output may differ from the exact dot product of the
/// same decoded weights.
///
/// A trellis dot product cancels: the result lands about four orders below
/// sum|w_i x_i|, so a bar written against the RESULT measures noise and passes
/// or fails on the seed. Every bar here is relative to the SUMMANDS, which is
/// the scale the arithmetic runs at.
const Exl3GemmParity = struct {
    /// f16 unit roundoff. An arm rounds its result to f16 exactly once and
    /// |result| <= sum|w_i x_i|, so that store costs at most U_F16 of the sum.
    const U_F16: f64 = 0x1p-11;
    /// f32 unit roundoff. Every arm accumulates its products in f32 planes, so
    /// a `depth`-long accumulation costs at most depth * U_F32 of the sum.
    const U_F32: f64 = 0x1p-24;
    /// How much of the composite's RMS error the arm may carry.
    const RMS_FACTOR: f64 = 3.0;

    /// Ceiling on one element's error as a fraction of sum|w_i x_i|.
    fn elemCeiling(depth: usize) f64 {
        return U_F16 + @as(f64, @floatFromInt(depth)) * U_F32;
    }
};

/// A defect injected between the arm and its reference, so the bar is shown to
/// FAIL on a wrong arm and not only to pass on a right one. `.none` is the
/// real test; the others make the arm's decode disagree with the reference's
/// exactly as a miscoded kernel would.
const Exl3Mutation = enum {
    none,
    /// The arm reads the codeword window the pack names, the reference reads
    /// another — the same defect as a kernel built for the wrong mask.
    window,
    /// One bit of one trellis halfword, which moves the weights whose sliding
    /// window covers it.
    codeword,
};

fn mutatedDecode(dec: exl3.Decode, mutate: Exl3Mutation) exl3.Decode {
    if (mutate != .window) return dec;
    return .{ .codebook = dec.codebook, .window = if (dec.window == .w16) .w15 else .w16 };
}

/// Why a GEMM parity check failed, or null when it passed. Pure, so the
/// decision is unit-testable without a GPU.
const Exl3GemmParityFail = enum { nonfinite, gross_element, systematic };

fn exl3GemmParityVerdict(
    finite: bool,
    kern_max: f64,
    rms_kern: f64,
    rms_comp: f64,
    ceiling: f64,
) ?Exl3GemmParityFail {
    if (!finite) return .nonfinite;
    if (kern_max > ceiling) return .gross_element;
    // A zero composite RMS means f16 represents every truth exactly on this
    // data; the arm then owes the same, which is the strictest reading.
    if (rms_kern > Exl3GemmParity.RMS_FACTOR * rms_comp) return .systematic;
    return null;
}

const Exl3GemmParityStats = struct {
    n: usize = 0,
    finite: bool = true,
    kern_max: f64 = 0,
    comp_max: f64 = 0,
    rms_kern: f64 = 0,
    rms_comp: f64 = 0,
    ceiling: f64 = 0,
};

/// Every expert's trellis decoded into a dense `[E, in_dim, out_dim]` f16
/// bank — the one decode both the truth and the composite read.
fn dequantStacked(
    alloc: std.mem.Allocator,
    stacked: []const u16,
    in_dim: usize,
    out_dim: usize,
    rate: exl3.Rate,
    dec: exl3.Decode,
) ![]u16 {
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const packed_n = rate.halfwords();
    const tile_n = in_tiles * out_tiles * packed_n;
    const w = try alloc.alloc(u16, (stacked.len / tile_n) * in_dim * out_dim);
    var tile_w: [exl3.TILE_VALUES]u16 = undefined;
    for (0..stacked.len / tile_n) |e| {
        const trellis = stacked[e * tile_n ..][0..tile_n];
        for (0..in_tiles) |tk| {
            for (0..out_tiles) |tn| {
                exl3.decodeTile(trellis[(tk * out_tiles + tn) * packed_n ..][0..packed_n], rate, dec, &tile_w);
                for (0..16) |r| {
                    const row = w[e * in_dim * out_dim + (tk * 16 + r) * out_dim ..];
                    for (0..16) |c| row[tn * 16 + c] = tile_w[r * 16 + c];
                }
            }
        }
    }
    return w;
}

/// Score one arm's output and mlx's own f16 matmul over the same decoded
/// weights, both against the exact dot product of those weights. Never
/// kernel-vs-kernel: the composite supplies a SCALE for the aggregate, never
/// a per-element answer.
fn measureInnerGemmParity(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    got: []const f16,
    xh: []const u16,
    eids: []const u32,
    stacked: []const u16,
    in_dim: usize,
    out_dim: usize,
    rate: exl3.Rate,
    dec: exl3.Decode,
) !Exl3GemmParityStats {
    const w = try dequantStacked(alloc, stacked, in_dim, out_dim, rate, dec);
    return measureInnerGemmParityOn(alloc, s, got, xh, eids, w, in_dim, out_dim);
}

/// The same verdict against a weight bank the caller already holds — a
/// reference decode, where the bar must not be our own decoder's output.
fn measureInnerGemmParityOn(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    got: []const f16,
    xh: []const u16,
    eids: []const u32,
    w: []const u16,
    in_dim: usize,
    out_dim: usize,
) !Exl3GemmParityStats {
    const rows = eids.len;
    const w_arr = mlx.mlx_array_new_data(w.ptr, &[_]c_int{ @intCast(w.len / (in_dim * out_dim)), @intCast(in_dim), @intCast(out_dim) }, 3, .float16);
    defer _ = mlx.mlx_array_free(w_arr);
    const eid_arr = mlx.mlx_array_new_data(eids.ptr, &[_]c_int{@intCast(rows)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_arr);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), 1, @intCast(in_dim) }, 3, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    var w_sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_sel);
    try mlx.check(mlx.mlx_take_axis(&w_sel, w_arr, eid_arr, 0, s));
    var comp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(comp);
    try mlx.check(mlx.mlx_matmul(&comp, x_arr, w_sel, s));
    var comp_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(comp_c);
    try mlx.check(mlx.mlx_contiguous(&comp_c, comp, false, s));
    try mlx.check(mlx.mlx_array_eval(comp_c));
    const comp_h = mlx.mlx_array_data_float16(comp_c) orelse return error.F16Unreadable;

    const truth = try alloc.alloc(f64, out_dim);
    const amag = try alloc.alloc(f64, out_dim);
    var st = Exl3GemmParityStats{ .n = rows * out_dim, .ceiling = Exl3GemmParity.elemCeiling(in_dim) };
    var se_k: f64 = 0;
    var se_c: f64 = 0;
    for (0..rows) |r| {
        @memset(truth, 0);
        @memset(amag, 0);
        const wb = w[@as(usize, eids[r]) * in_dim * out_dim ..];
        for (0..in_dim) |k| {
            // An f16 product is exact in f64, so this sum IS the dot product.
            const xv: f64 = exl3.f16BitsToF32(xh[r * in_dim + k]);
            const wrow = wb[k * out_dim ..][0..out_dim];
            for (0..out_dim) |o| {
                const p = xv * @as(f64, exl3.f16BitsToF32(wrow[o]));
                truth[o] += p;
                amag[o] += @abs(p);
            }
        }
        for (0..out_dim) |o| {
            const g: f64 = @floatCast(got[r * out_dim + o]);
            const c: f64 = @floatCast(comp_h[r * out_dim + o]);
            if (!std.math.isFinite(g) or !std.math.isFinite(c) or !std.math.isFinite(truth[o])) {
                st.finite = false;
                return st;
            }
            const denom = if (amag[o] > 0) amag[o] else 1.0;
            const ek = @abs(g - truth[o]) / denom;
            const ec = @abs(c - truth[o]) / denom;
            st.kern_max = @max(st.kern_max, ek);
            st.comp_max = @max(st.comp_max, ec);
            se_k += ek * ek;
            se_c += ec * ec;
        }
    }
    const n: f64 = @floatFromInt(rows * out_dim);
    st.rms_kern = @sqrt(se_k / n);
    st.rms_comp = @sqrt(se_c / n);
    return st;
}

/// The verdict on measured stats, with one failure line naming both sides and
/// the bars.
fn reportGemmParity(st: Exl3GemmParityStats) !void {
    if (exl3GemmParityVerdict(st.finite, st.kern_max, st.rms_kern, st.rms_comp, st.ceiling)) |why| {
        std.debug.print(
            "exl3 GEMM parity FAIL ({s}): n={d} kernel max={d:.7} rms={d:.7}; composite max={d:.7} rms={d:.7}; " ++
                "bars: max<={d:.7} rms<={d:.1}x\n",
            .{ @tagName(why), st.n, st.kern_max, st.rms_kern, st.comp_max, st.rms_comp, st.ceiling, Exl3GemmParity.RMS_FACTOR },
        );
        return error.TestExpectedApproxEq;
    }
}

fn expectInnerGemmParity(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    got: []const f16,
    xh: []const u16,
    eids: []const u32,
    stacked: []const u16,
    in_dim: usize,
    out_dim: usize,
    rate: exl3.Rate,
    dec: exl3.Decode,
) !void {
    try reportGemmParity(try measureInnerGemmParity(alloc, s, got, xh, eids, stacked, in_dim, out_dim, rate, dec));
}

test "exl3GemmParityVerdict: cancellation noise passes, a wrong weight or a systematic drift does not" {
    const ceiling = Exl3GemmParity.elemCeiling(128);
    // The arm's own rounding: a worst element inside the f16 store's share of
    // the bar, an RMS that matches the composite's.
    try std.testing.expect(exl3GemmParityVerdict(true, 2.19e-4, 2.65e-5, 2.65e-5, ceiling) == null);
    // One decoded weight wrong moves a single element far past the ceiling
    // while the RMS barely stirs: the element bar is the one that sees it.
    try std.testing.expectEqual(
        Exl3GemmParityFail.gross_element,
        exl3GemmParityVerdict(true, 8.0e-3, 2.7e-5, 2.65e-5, ceiling).?,
    );
    // Noisier everywhere without one gross element: the RMS ratio sees it.
    try std.testing.expectEqual(
        Exl3GemmParityFail.systematic,
        exl3GemmParityVerdict(true, 4.0e-4, 1.0e-4, 2.65e-5, ceiling).?,
    );
    try std.testing.expectEqual(
        Exl3GemmParityFail.nonfinite,
        exl3GemmParityVerdict(false, 0, 0, 0, ceiling).?,
    );
    // An arm better than the composite is never a failure.
    try std.testing.expect(exl3GemmParityVerdict(true, 1.0e-5, 1.0e-6, 2.65e-5, ceiling) == null);
    // The ceiling follows the accumulation depth.
    try std.testing.expect(Exl3GemmParity.elemCeiling(2560) > ceiling);
}

test "exl3 K4 cooperative indexed GEMV matches host MUL1 on production shape" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 4;
    const topk: usize = 10;
    const in_dim: usize = 2560;
    const out_dim: usize = 640;
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const tile_n = in_tiles * out_tiles * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    for (stacked) |*v| v.* = @truncate(rnd.int(u32));
    const xh = try alloc.alloc(u16, topk * in_dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(topk), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const got = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try expectInnerGemmParity(alloc, s, src[0 .. topk * out_dim], xh, slots_h, stacked, in_dim, out_dim, exl3.Rate.fromK(4), .mul1);
}

test "exl3 K4 cooperative indexed GEMV runs at production shape" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 16;
    const topk: usize = 10;
    const in_dim: usize = 2560;
    const out_dim: usize = 640;
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const tile_n = in_tiles * out_tiles * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    var prng = std.Random.DefaultPrng.init(29);
    const rnd = prng.random();
    for (stacked) |*v| v.* = @truncate(rnd.int(u32));
    const xh = try alloc.alloc(u16, topk * in_dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    const slots_h = try alloc.alloc(u32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(topk), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const warm_new = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
    try mlx.check(mlx.mlx_array_eval(warm_new));
    _ = mlx.mlx_array_free(warm_new);
    var new_ns: u64 = 0;
    var it: usize = 0;
    while (it < 10) : (it += 1) {
        var sw = io_util.Stopwatch.init(t.io);
        const b = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
        try mlx.check(mlx.mlx_array_eval(b));
        new_ns += sw.read();
        _ = mlx.mlx_array_free(b);
    }
    new_ns /= 10;
    const packed3 = exl3.packedHalfwords(3);
    const tile3 = in_tiles * out_tiles * packed3;
    const stacked3 = try alloc.alloc(u16, E * tile3);
    for (stacked3) |*v| v.* = @truncate(rnd.int(u32));
    const tr3 = mlx.mlx_array_new_data(stacked3.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), @intCast(packed3) }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr3);
    const warm3 = try indexedGemvCoopF16(s, x_arr, tr3, slots);
    try mlx.check(mlx.mlx_array_eval(warm3));
    _ = mlx.mlx_array_free(warm3);
    var k3_ns: u64 = 0;
    var k4_ns: u64 = 0;
    it = 0;
    while (it < 10) : (it += 1) {
        var sw4 = io_util.Stopwatch.init(t.io);
        const b4 = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
        try mlx.check(mlx.mlx_array_eval(b4));
        k4_ns += sw4.read();
        _ = mlx.mlx_array_free(b4);
        var sw3 = io_util.Stopwatch.init(t.io);
        const b3 = try indexedGemvCoopF16(s, x_arr, tr3, slots);
        try mlx.check(mlx.mlx_array_eval(b3));
        k3_ns += sw3.read();
        _ = mlx.mlx_array_free(b3);
    }
    k3_ns /= 10;
    k4_ns /= 10;
    benchPrint("exl3 indexed GEMV H=2560 I=640 topk=10: K4 {d} us K3 {d} us\n", .{
        k4_ns / 1000,
        k3_ns / 1000,
    });
    try t.expect(new_ns > 0);
    try t.expect(k4_ns > 0);
    benchPrint("[exl3-k3-timing] k3 {d} us k4 {d} us ratio {d:.3}\n", .{ k3_ns / 1000, k4_ns / 1000, @as(f64, @floatFromInt(k3_ns)) / @as(f64, @floatFromInt(@max(k4_ns, 1))) });
    try t.expect(k3_ns < k4_ns * 3);
}

fn layerUbench(dec: exl3.Decode) !void {
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer {
        ubench_mute = false;
    }
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 16;
    const topk: usize = 10;
    const in_dim: usize = 2560;
    const out_dim: usize = 640;
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const g_n = in_tiles * out_tiles * 64;
    const d_n = out_tiles * in_tiles * 64;
    const tr_g = try alloc.alloc(u16, E * g_n);
    const tr_d = try alloc.alloc(u16, E * d_n);
    const suh_g = try alloc.alloc(u16, E * in_dim);
    const svh_g = try alloc.alloc(u16, E * out_dim);
    const suh_d = try alloc.alloc(u16, E * out_dim);
    const svh_d = try alloc.alloc(u16, E * in_dim);
    var prng = std.Random.DefaultPrng.init(71);
    const rnd = prng.random();
    for (tr_g) |*v| v.* = @truncate(rnd.int(u32));
    for (tr_d) |*v| v.* = @truncate(rnd.int(u32));
    for (suh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (suh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    const trg = mlx.mlx_array_new_data(tr_g.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trg);
    const trd = mlx.mlx_array_new_data(tr_d.ptr, &[_]c_int{ @intCast(E), @intCast(out_tiles), @intCast(in_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trd);
    const sugh = mlx.mlx_array_new_data(suh_g.ptr, &[_]c_int{ @intCast(E), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sugh);
    const svgi = mlx.mlx_array_new_data(svh_g.ptr, &[_]c_int{ @intCast(E), @intCast(out_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svgi);
    const sudi = mlx.mlx_array_new_data(suh_d.ptr, &[_]c_int{ @intCast(E), @intCast(out_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sudi);
    const svdh = mlx.mlx_array_new_data(svh_d.ptr, &[_]c_int{ @intCast(E), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svdh);
    const x1h = try alloc.alloc(u16, in_dim);
    for (x1h) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    const sl1 = try alloc.alloc(u32, topk);
    const sc1 = try alloc.alloc(f32, topk);
    for (sl1, 0..) |*v, i| v.* = @intCast(i % E);
    for (sc1) |*v| v.* = 0.1;
    const x1 = mlx.mlx_array_new_data(x1h.ptr, &[_]c_int{@intCast(in_dim)}, 1, .float16);
    defer _ = mlx.mlx_array_free(x1);
    const slots1 = mlx.mlx_array_new_data(sl1.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots1);
    const scores1 = mlx.mlx_array_new_data(sc1.ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores1);
    ubench_mute = true;
    const warm1 = try moeSwigluFused(s, x1, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots1, scores1, .float16);
    try mlx.check(mlx.mlx_array_eval(warm1));
    _ = mlx.mlx_array_free(warm1);
    ubench_mute = false;
    benchPrint("exl3-ubench codebook={s} rows=1\n", .{@tagName(dec.codebook)});
    const y1 = try moeSwigluFused(s, x1, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots1, scores1, .float16);
    try mlx.check(mlx.mlx_array_eval(y1));
    _ = mlx.mlx_array_free(y1);
    const R: usize = 512;
    const xnh = try alloc.alloc(u16, R * in_dim);
    for (xnh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    const sln = try alloc.alloc(u32, R * topk);
    const scn = try alloc.alloc(f32, R * topk);
    for (sln, 0..) |*v, i| v.* = @intCast(i % E);
    for (scn) |*v| v.* = 0.1;
    const xn = mlx.mlx_array_new_data(xnh.ptr, &[_]c_int{ @intCast(R), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(xn);
    const slotsn = mlx.mlx_array_new_data(sln.ptr, &[_]c_int{@intCast(R * topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slotsn);
    const scoresn = mlx.mlx_array_new_data(scn.ptr, &[_]c_int{@intCast(R * topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scoresn);
    ubench_mute = true;
    const warmn = try moePrefill(s, xn, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slotsn, scoresn, @intCast(topk));
    try mlx.check(mlx.mlx_array_eval(warmn));
    _ = mlx.mlx_array_free(warmn);
    ubench_mute = false;
    benchPrint("exl3-ubench codebook={s} rows=512\n", .{@tagName(dec.codebook)});
    const yn = try moePrefill(s, xn, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slotsn, scoresn, @intCast(topk));
    try mlx.check(mlx.mlx_array_eval(yn));
    _ = mlx.mlx_array_free(yn);
    for ([_]usize{ 2, 3 }) |Rsmall| {
        const xsh = try alloc.alloc(u16, Rsmall * in_dim);
        for (xsh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
        const sls = try alloc.alloc(u32, Rsmall * topk);
        const scs = try alloc.alloc(f32, Rsmall * topk);
        for (sls, 0..) |*v, i| v.* = @intCast(i % E);
        for (scs) |*v| v.* = 0.1;
        const xs = mlx.mlx_array_new_data(xsh.ptr, &[_]c_int{ @intCast(Rsmall), @intCast(in_dim) }, 2, .float16);
        defer _ = mlx.mlx_array_free(xs);
        const slotss = mlx.mlx_array_new_data(sls.ptr, &[_]c_int{@intCast(Rsmall * topk)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(slotss);
        const scoress = mlx.mlx_array_new_data(scs.ptr, &[_]c_int{@intCast(Rsmall * topk)}, 1, .float32);
        defer _ = mlx.mlx_array_free(scoress);
        ubench_mute = true;
        const warms = try moeSwigluFused(s, xs, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slotss, scoress, .float16);
        try mlx.check(mlx.mlx_array_eval(warms));
        _ = mlx.mlx_array_free(warms);
        ubench_mute = false;
        benchPrint("exl3-ubench codebook={s} rows={d}\n", .{ @tagName(dec.codebook), Rsmall });
        const ys = try moeSwigluFused(s, xs, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slotss, scoress, .float16);
        try mlx.check(mlx.mlx_array_eval(ys));
        _ = mlx.mlx_array_free(ys);
    }
}

test "exl3 layer ubench production shape rows=1 and 512 per codebook" {
    // With the ubench on, the two codebooks interleave over several rounds so
    // GPU clock ramp lands on both; read medians per kernel, never one shot.
    const rounds: usize = if (exl3UbenchOn()) 5 else 1;
    for (0..rounds) |_| {
        for ([_]exl3.Decode{ .mul1, .tiny }) |dec| try layerUbench(dec);
    }
}

test "exl3 fused decode chain matches indexed SwiGLU on one row" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const suh_meta = parsed.value.object.get("suh").?.object;
    const svh_meta = parsed.value.object.get("svh").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(svh_meta.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(svh_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: usize = 4;
    const dim: usize = 128;
    const topk: usize = 4;
    const tile_n = 8 * 8 * 64;
    const stacked_t = try alloc.alloc(u16, E * tile_n);
    const stacked_suh = try alloc.alloc(u16, E * dim);
    const stacked_svh = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(stacked_t[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(stacked_suh[e * dim ..][0..dim], suh_bits);
        @memcpy(stacked_svh[e * dim ..][0..dim], svh_bits);
    }
    var prng = std.Random.DefaultPrng.init(41);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, topk);
    const scores_h = try alloc.alloc(f32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    for (scores_h) |*v| v.* = rnd.float(f32);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{@intCast(dim)}, 1, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(stacked_svh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    resetFusedDispatchCount();
    pair_splits_force = 1;
    const fused = try moeSwigluFused(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, .float16);
    defer _ = mlx.mlx_array_free(fused);
    const n_disp = fusedDispatchCount();
    const old = try moeSwigluIndexed(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores);
    defer _ = mlx.mlx_array_free(old);
    var c_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c_f);
    var c_o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c_o);
    try mlx.check(mlx.mlx_contiguous(&c_f, fused, false, s));
    try mlx.check(mlx.mlx_contiguous(&c_o, old, false, s));
    try mlx.check(mlx.mlx_array_eval(c_f));
    try mlx.check(mlx.mlx_array_eval(c_o));
    const sf = mlx.mlx_array_data_float16(c_f) orelse return error.F16Unreadable;
    const so = mlx.mlx_array_data_float16(c_o) orelse return error.F16Unreadable;
    try expectRelRms(sf[0..dim], so[0..dim], 0.01);
    // The decode chain is three dispatches: pair GEMV (prepare inlined), fused mid+down, finish reduce.
    try t.expectEqual(@as(u32, 3), n_disp);
    const fused_bf = try moeSwigluFused(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, .bfloat16);
    defer _ = mlx.mlx_array_free(fused_bf);
    try t.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(fused_bf));
    pair_splits_force = 2;
    defer {
        pair_splits_force = null;
    }
    const fused2 = try moeSwigluFused(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, .float16);
    defer _ = mlx.mlx_array_free(fused2);
    var c2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c2);
    try mlx.check(mlx.mlx_contiguous(&c2, fused2, false, s));
    try mlx.check(mlx.mlx_array_eval(c2));
    const s2 = mlx.mlx_array_data_float16(c2) orelse return error.F16Unreadable;
    try expectRelRms(s2[0..dim], sf[0..dim], 0.01);
}

test "exl3 fused decode chain rows match N solo calls" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const suh_meta = parsed.value.object.get("suh").?.object;
    const svh_meta = parsed.value.object.get("svh").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(svh_meta.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(svh_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: usize = 4;
    const dim: usize = 128;
    const topk: usize = 4;
    const rows: usize = 8;
    const tile_n = 8 * 8 * 64;
    const stacked_t = try alloc.alloc(u16, E * tile_n);
    const stacked_suh = try alloc.alloc(u16, E * dim);
    const stacked_svh = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(stacked_t[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(stacked_suh[e * dim ..][0..dim], suh_bits);
        @memcpy(stacked_svh[e * dim ..][0..dim], svh_bits);
    }
    var prng = std.Random.DefaultPrng.init(43);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, rows * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, rows * topk);
    const scores_h = try alloc.alloc(f32, rows * topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    for (scores_h) |*v| v.* = rnd.float(f32);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(stacked_svh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    pair_splits_force = 1;
    defer {
        pair_splits_force = null;
    }
    const fused = try moeSwigluFused(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, .float16);
    defer _ = mlx.mlx_array_free(fused);
    var c_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c_f);
    try mlx.check(mlx.mlx_contiguous(&c_f, fused, false, s));
    try mlx.check(mlx.mlx_array_eval(c_f));
    const sf = mlx.mlx_array_data_float16(c_f) orelse return error.F16Unreadable;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const x1 = mlx.mlx_array_new_data(xh[r * dim ..][0..dim].ptr, &[_]c_int{@intCast(dim)}, 1, .float16);
        defer _ = mlx.mlx_array_free(x1);
        const sl = mlx.mlx_array_new_data(slots_h[r * topk ..][0..topk].ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(sl);
        const sc = mlx.mlx_array_new_data(scores_h[r * topk ..][0..topk].ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
        defer _ = mlx.mlx_array_free(sc);
        const solo = try moeSwigluIndexed(s, x1, tr, suh, svh, tr, suh, svh, tr, suh, svh, sl, sc);
        defer _ = mlx.mlx_array_free(solo);
        var c_s = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c_s);
        try mlx.check(mlx.mlx_contiguous(&c_s, solo, false, s));
        try mlx.check(mlx.mlx_array_eval(c_s));
        const ss = mlx.mlx_array_data_float16(c_s) orelse return error.F16Unreadable;
        try expectRelRms(sf[r * dim ..][0..dim], ss[0..dim], 0.01);
    }
}

test "exl3 pair split clamps to one where it would cut a Hadamard block" {
    const t = std.testing;
    setPairSplitsForTest(4);
    defer setPairSplitsForTest(null);
    try t.expectEqual(@as(u32, 4), pairSplitCountFor(2048));
    try t.expectEqual(@as(u32, 1), pairSplitCountFor(1280));
    try t.expectEqual(@as(u32, 1), pairSplitCountFor(128));
}

test "exl3 split-2 decode chain matches split-1 on a k range the GEMV stages itself" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 4;
    const topk: usize = 2;
    const dim: usize = 256;
    const inter: usize = 128;
    const gt = try alloc.alloc(u16, E * (dim / 16) * (inter / 16) * 64);
    const dt = try alloc.alloc(u16, E * (inter / 16) * (dim / 16) * 64);
    var prng = std.Random.DefaultPrng.init(9);
    const rnd = prng.random();
    for (gt) |*v| v.* = @truncate(rnd.int(u32));
    for (dt) |*v| v.* = @truncate(rnd.int(u32));
    const suh_in = try alloc.alloc(u16, E * dim);
    const svh_in = try alloc.alloc(u16, E * inter);
    const suh_d = try alloc.alloc(u16, E * inter);
    const svh_d = try alloc.alloc(u16, E * dim);
    for (suh_in) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) + 0.5);
    for (svh_in) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) + 0.5);
    for (suh_d) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) + 0.5);
    for (svh_d) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) + 0.5);
    const gta = mlx.mlx_array_new_data(gt.ptr, &[_]c_int{ E, dim / 16, inter / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(gta);
    const dta = mlx.mlx_array_new_data(dt.ptr, &[_]c_int{ E, inter / 16, dim / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(dta);
    const suh_in_a = mlx.mlx_array_new_data(suh_in.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh_in_a);
    const svh_in_a = mlx.mlx_array_new_data(svh_in.ptr, &[_]c_int{ E, inter }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh_in_a);
    const suh_d_a = mlx.mlx_array_new_data(suh_d.ptr, &[_]c_int{ E, inter }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh_d_a);
    const svh_d_a = mlx.mlx_array_new_data(svh_d.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh_d_a);
    const xh = try alloc.alloc(u16, dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{dim}, 1, .float16);
    defer _ = mlx.mlx_array_free(x);
    const sl = try alloc.alloc(u32, topk);
    for (sl, 0..) |*v, i| v.* = @intCast(i % E);
    const slots = mlx.mlx_array_new_data(sl.ptr, &[_]c_int{topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const sc = try alloc.alloc(f32, topk);
    for (sc) |*v| v.* = 0.5;
    const scores = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{topk}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const solo = blk: {
        setPairSplitsForTest(1);
        defer setPairSplitsForTest(null);
        break :blk try moeSwigluFused(s, x, gta, suh_in_a, svh_in_a, gta, suh_in_a, svh_in_a, dta, suh_d_a, svh_d_a, slots, scores, .float16);
    };
    defer _ = mlx.mlx_array_free(solo);
    const split = blk: {
        setPairSplitsForTest(2);
        defer setPairSplitsForTest(null);
        break :blk try moeSwigluFused(s, x, gta, suh_in_a, svh_in_a, gta, suh_in_a, svh_in_a, dta, suh_d_a, svh_d_a, slots, scores, .float16);
    };
    defer _ = mlx.mlx_array_free(split);
    try mlx.check(mlx.mlx_array_eval(solo));
    try mlx.check(mlx.mlx_array_eval(split));
    const a = mlx.mlx_array_data_float16(solo) orelse return error.F16Unreadable;
    const b = mlx.mlx_array_data_float16(split) orelse return error.F16Unreadable;
    try expectRelRms(b[0..dim], a[0..dim], 0.01);
}

test "exl3 pair GEMV inner planes are f32" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const suh_meta = parsed.value.object.get("suh").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const E: usize = 4;
    const dim: usize = 128;
    const topk: usize = 4;
    const tile_n = 8 * 8 * 64;
    const stacked_t = try alloc.alloc(u16, E * tile_n);
    const stacked_suh = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(stacked_t[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(stacked_suh[e * dim ..][0..dim], suh_bits);
    }
    var prng = std.Random.DefaultPrng.init(47);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{@intCast(dim)}, 1, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    pair_splits_force = 1;
    defer {
        pair_splits_force = null;
    }
    const inners = try pairGemv(s, x_arr, suh, suh, tr, tr, slots, @intCast(dim), @intCast(dim), @intCast(topk), @intCast(topk));
    defer _ = mlx.mlx_array_free(inners[0]);
    defer _ = mlx.mlx_array_free(inners[1]);
    try mlx.check(mlx.mlx_array_eval(inners[0]));
    try t.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(inners[0]));
    try t.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(inners[1]));
}

test "exl3 fused rows at split-2 match N fused solo" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const suh_meta = parsed.value.object.get("suh").?.object;
    const svh_meta = parsed.value.object.get("svh").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(svh_meta.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(svh_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: usize = 4;
    const dim: usize = 128;
    const topk: usize = 4;
    const rows: usize = 8;
    const tile_n = 8 * 8 * 64;
    const stacked_t = try alloc.alloc(u16, E * tile_n);
    const stacked_suh = try alloc.alloc(u16, E * dim);
    const stacked_svh = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(stacked_t[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(stacked_suh[e * dim ..][0..dim], suh_bits);
        @memcpy(stacked_svh[e * dim ..][0..dim], svh_bits);
    }
    var prng = std.Random.DefaultPrng.init(43);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, rows * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, rows * topk);
    const scores_h = try alloc.alloc(f32, rows * topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    for (scores_h) |*v| v.* = rnd.float(f32);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(stacked_svh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    pair_splits_force = 2;
    defer {
        pair_splits_force = null;
    }
    const fused = try moeSwigluFused(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, .float16);
    defer _ = mlx.mlx_array_free(fused);
    var c_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c_f);
    try mlx.check(mlx.mlx_contiguous(&c_f, fused, false, s));
    try mlx.check(mlx.mlx_array_eval(c_f));
    const sf = mlx.mlx_array_data_float16(c_f) orelse return error.F16Unreadable;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const x1 = mlx.mlx_array_new_data(xh[r * dim ..][0..dim].ptr, &[_]c_int{@intCast(dim)}, 1, .float16);
        defer _ = mlx.mlx_array_free(x1);
        const sl = mlx.mlx_array_new_data(slots_h[r * topk ..][0..topk].ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(sl);
        const sc = mlx.mlx_array_new_data(scores_h[r * topk ..][0..topk].ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
        defer _ = mlx.mlx_array_free(sc);
        const solo = try moeSwigluFused(s, x1, tr, suh, svh, tr, suh, svh, tr, suh, svh, sl, sc, .float16);
        defer _ = mlx.mlx_array_free(solo);
        var c_s = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c_s);
        try mlx.check(mlx.mlx_contiguous(&c_s, solo, false, s));
        try mlx.check(mlx.mlx_array_eval(c_s));
        const ss = mlx.mlx_array_data_float16(c_s) orelse return error.F16Unreadable;
        for (0..dim) |i| {
            const a: u16 = @bitCast(sf[r * dim + i]);
            const b: u16 = @bitCast(ss[i]);
            try t.expectEqual(b, a);
        }
    }
}

test "exl3 MTP MoE rows stay on the fused decode arm" {
    const t = std.testing;
    try t.expect(!usesPrefillArm(1));
    try t.expect(!usesPrefillArm(4));
    try t.expect(!usesPrefillArm(16));
    try t.expect(usesPrefillArm(17));
}

fn sortedGemmSmallShape(dec: exl3.Decode) !void {
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 8;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(47);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const eids = [_]u32{ 0, 0, 0, 0, 2, 2, 1, 1 };
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const got = try innerGemmSorted(s, x_arr, tr_arr, eid_a);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try expectInnerGemmParity(alloc, s, src[0 .. n * dim], xh, &eids, stacked, dim, dim, exl3.Rate.fromK(4), dec);
}

test "exl3 sorted GEMM matches host MUL1 on small shape" {
    try sortedGemmSmallShape(.mul1);
}

test "exl3 sorted GEMM matches host TINY on small shape" {
    try sortedGemmSmallShape(.tiny);
}

test "exl3 a NAX GEMM source the Metal toolchain rejects is declined at the probe, not at prefill" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    mlx.installErrorHandler();
    try t.expect(!mlx.errorPending());
    try t.expect(buildNaxGemmKernel("this is not metal;", "", "mlxserve_exl3_probe_bad") == null);
    try t.expect(!mlx.errorPending());
    const real = buildNaxGemmKernel(GEMM_NAX_SOURCE, naxHeader(.mul1, .w16), "mlxserve_exl3_k4_gemm_nax") orelse return error.TestUnexpectedResult;
    _ = mlx.mlx_fast_metal_kernel_free(real);
    try t.expect(!mlx.errorPending());
}

test "exl3 K3 sorted GEMM matches host MUL1 on small shape" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k3;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 8;
    const tile_n = 8 * 8 * 48;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(47);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const eids = [_]u32{ 0, 0, 0, 0, 2, 2, 1, 1 };
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 48 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const got = try innerGemmSorted(s, x_arr, tr_arr, eid_a);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try expectInnerGemmParity(alloc, s, src[0 .. n * dim], xh, &eids, stacked, dim, dim, exl3.Rate.fromK(3), .mul1);
}

test "exl3 K4 sorted GEMM matches host MUL1 when a run half-fills the second block" {
    // 20-row runs put 16 rows in the first destination and 4 in the second, so
    // the second block's activation reads are the predicated ones.
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 40;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(61);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    var eids: [40]u32 = undefined;
    var i: usize = 0;
    while (i < 20) : (i += 1) eids[i] = 0;
    while (i < n) : (i += 1) eids[i] = 2;
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const got = try innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 32);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try expectInnerGemmParity(alloc, s, src[0 .. n * dim], xh, &eids, stacked, dim, dim, exl3.Rate.fromK(4), .mul1);
}

test "exl3 K3 sorted GEMM matches host MUL1 on 20-40 row runs" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    if (!gemmNaxOn()) return error.SkipZigTest;
    const fixture = exl3.fixtures.k3;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 52;
    const tile_n = 8 * 8 * 48;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(53);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    var eids: [52]u32 = undefined;
    var i: usize = 0;
    while (i < 20) : (i += 1) eids[i] = 0;
    while (i < n) : (i += 1) eids[i] = 1;
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 48 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const got = try innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 32);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try expectInnerGemmParity(alloc, s, src[0 .. n * dim], xh, &eids, stacked, dim, dim, exl3.Rate.fromK(3), .mul1);
}

test "exl3 sorted GEMM matches host MUL1 with the NAX arm forced off" {
    // On NAX hardware every other sorted-GEMM test takes the NAX arm (out_dim 128), so this is the SIMD body's only bar.
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    if (!gemmNaxOn()) return error.SkipZigTest;
    _ = setenv("MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK", "1", 1);
    defer _ = unsetenv("MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK");
    try t.expect(!gemmNaxOn());
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    const runs = [_]u32{ 7, 32, 5, 18, 40, 10 };
    var n: usize = 0;
    for (runs) |r| n += r;
    const eids = try alloc.alloc(u32, n);
    {
        var off: usize = 0;
        for (runs, 0..) |r, ei| {
            var j: usize = 0;
            while (j < r) : (j += 1) eids[off + j] = @intCast(ei % E);
            off += r;
        }
    }
    var prng = std.Random.DefaultPrng.init(0x5124);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(eids.ptr, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    for ([_]bool{ true, false }) |aligned| {
        const got = try innerGemmSortedWinAlign(s, x_arr, tr_arr, eid_a, 32, aligned);
        defer _ = mlx.mlx_array_free(got);
        var contig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(contig);
        try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
        try mlx.check(mlx.mlx_array_eval(contig));
        const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
        errdefer std.debug.print("[exl3-simd-arm] aligned={}\n", .{aligned});
        try expectInnerGemmParity(alloc, s, src[0 .. n * dim], xh, eids, stacked, dim, dim, exl3.Rate.fromK(4), .mul1);
    }
}

test "exl3 NAX K3 GEMM within 3x of K4 at C=2048 and 8192" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    if (!gemmNaxOn()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 4;
    const in_dim: usize = 2560;
    const out_dim: usize = 640;
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    var prng = std.Random.DefaultPrng.init(71);
    const rnd = prng.random();
    const contexts = [_]c_int{ 2048, 8192 };
    for (contexts) |C| {
        const n: usize = @intCast(C);
        const xh = try alloc.alloc(u16, n * in_dim);
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
        const eids = try alloc.alloc(u32, n);
        const run = n / E;
        for (eids, 0..) |*v, i| v.* = @intCast(@min(i / run, E - 1));
        const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ C, @intCast(in_dim) }, 2, .float16);
        defer _ = mlx.mlx_array_free(x_arr);
        const eid_a = mlx.mlx_array_new_data(eids.ptr, &[_]c_int{C}, 1, .uint32);
        defer _ = mlx.mlx_array_free(eid_a);
        var k4_ns: u64 = 0;
        var k3_ns: u64 = 0;
        const packed4 = exl3.packedHalfwords(4);
        const packed3 = exl3.packedHalfwords(3);
        const tile4 = in_tiles * out_tiles * packed4;
        const tile3 = in_tiles * out_tiles * packed3;
        const stacked4 = try alloc.alloc(u16, E * tile4);
        const stacked3 = try alloc.alloc(u16, E * tile3);
        for (stacked4) |*v| v.* = @truncate(rnd.int(u32));
        for (stacked3) |*v| v.* = @truncate(rnd.int(u32));
        const tr4 = mlx.mlx_array_new_data(stacked4.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), @intCast(packed4) }, 4, .uint16);
        defer _ = mlx.mlx_array_free(tr4);
        const tr3 = mlx.mlx_array_new_data(stacked3.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), @intCast(packed3) }, 4, .uint16);
        defer _ = mlx.mlx_array_free(tr3);
        var warm_i: usize = 0;
        while (warm_i < 3) : (warm_i += 1) {
            inline for (.{ tr4, tr3 }) |tr| {
                const warm = try innerGemmSorted(s, x_arr, tr, eid_a);
                try mlx.check(mlx.mlx_array_eval(warm));
                _ = mlx.mlx_array_free(warm);
            }
        }
        var it: usize = 0;
        while (it < 8) : (it += 1) {
            var sw4 = io_util.Stopwatch.init(t.io);
            const b4 = try innerGemmSorted(s, x_arr, tr4, eid_a);
            try mlx.check(mlx.mlx_array_eval(b4));
            k4_ns += sw4.read();
            _ = mlx.mlx_array_free(b4);
            var sw3 = io_util.Stopwatch.init(t.io);
            const b3 = try innerGemmSorted(s, x_arr, tr3, eid_a);
            try mlx.check(mlx.mlx_array_eval(b3));
            k3_ns += sw3.read();
            _ = mlx.mlx_array_free(b3);
        }
        k4_ns /= 8;
        k3_ns /= 8;
        benchPrint("exl3 NAX one-proj C={d} H=2560 I=640: K4 {d} us K3 {d} us\n", .{
            C,
            k4_ns / 1000,
            k3_ns / 1000,
        });
        try t.expect(k4_ns > 0);
        benchPrint("[exl3-k3-timing] k3 {d} us k4 {d} us ratio {d:.3}\n", .{ k3_ns / 1000, k4_ns / 1000, @as(f64, @floatFromInt(k3_ns)) / @as(f64, @floatFromInt(@max(k4_ns, 1))) });
        try t.expect(k3_ns < k4_ns * 3);
    }
}

test "exl3 sorted GEMM 16-row windows match 4-row per row" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 32;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(61);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    var eids: [32]u32 = undefined;
    var i: usize = 0;
    while (i < 10) : (i += 1) eids[i] = 0;
    while (i < 13) : (i += 1) eids[i] = 2;
    while (i < 29) : (i += 1) eids[i] = 1;
    while (i < n) : (i += 1) eids[i] = 3;
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const got4 = try innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 4);
    defer _ = mlx.mlx_array_free(got4);
    const got16 = try innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 16);
    defer _ = mlx.mlx_array_free(got16);
    var c4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c4);
    var c16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c16);
    try mlx.check(mlx.mlx_contiguous(&c4, got4, false, s));
    try mlx.check(mlx.mlx_contiguous(&c16, got16, false, s));
    try mlx.check(mlx.mlx_array_eval(c4));
    try mlx.check(mlx.mlx_array_eval(c16));
    const a4 = mlx.mlx_array_data_float16(c4) orelse return error.F16Unreadable;
    const a16 = mlx.mlx_array_data_float16(c16) orelse return error.F16Unreadable;
    for (0..n * dim) |j| {
        const b4: u16 = @bitCast(a4[j]);
        const b16: u16 = @bitCast(a16[j]);
        try t.expectEqual(b4, b16);
    }
}

test "exl3 run-aligned windows match stride per row" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 40;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(101);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    var eids: [40]u32 = undefined;
    var i: usize = 0;
    while (i < 7) : (i += 1) eids[i] = 0;
    while (i < 23) : (i += 1) eids[i] = 1;
    while (i < 27) : (i += 1) eids[i] = 2;
    while (i < n) : (i += 1) eids[i] = 3;
    const st = windowStats(eids[0..], 16, false);
    const al = windowStats(eids[0..], 16, true);
    try t.expect(st.mixed > 0);
    try t.expectEqual(@as(u32, 0), al.mixed);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const wins = [_]c_int{ 16, 32 };
    for (wins) |w| {
        const stride = try innerGemmSortedWinAlign(s, x_arr, tr_arr, eid_a, w, false);
        defer _ = mlx.mlx_array_free(stride);
        const aligned = try innerGemmSortedWinAlign(s, x_arr, tr_arr, eid_a, w, true);
        defer _ = mlx.mlx_array_free(aligned);
        var cs = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cs);
        var ca = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ca);
        try mlx.check(mlx.mlx_contiguous(&cs, stride, false, s));
        try mlx.check(mlx.mlx_contiguous(&ca, aligned, false, s));
        try mlx.check(mlx.mlx_array_eval(cs));
        try mlx.check(mlx.mlx_array_eval(ca));
        const as = mlx.mlx_array_data_float16(cs) orelse return error.F16Unreadable;
        const aa = mlx.mlx_array_data_float16(ca) orelse return error.F16Unreadable;
        for (0..n * dim) |j| {
            const bs: u16 = @bitCast(as[j]);
            const ba: u16 = @bitCast(aa[j]);
            try t.expectEqual(bs, ba);
        }
    }
}

test "exl3 aligned GEMM reuses config across nwin" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 40;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(113);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    var long_runs: [40]u32 = undefined;
    for (&long_runs) |*v| v.* = 1;
    var short_runs: [40]u32 = undefined;
    for (&short_runs, 0..) |*v, i| v.* = @intCast(i % 4);
    try t.expect(windowStats(long_runs[0..], 32, true).nwin < windowStats(short_runs[0..], 32, true).nwin);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_long = mlx.mlx_array_new_data(&long_runs, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_long);
    const eid_short = mlx.mlx_array_new_data(&short_runs, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_short);
    const a_long = try innerGemmSortedWinAlign(s, x_arr, tr_arr, eid_long, 32, true);
    defer _ = mlx.mlx_array_free(a_long);
    try mlx.check(mlx.mlx_array_eval(a_long));
    const stride_short = try innerGemmSortedWinAlign(s, x_arr, tr_arr, eid_short, 32, false);
    defer _ = mlx.mlx_array_free(stride_short);
    const aligned_short = try innerGemmSortedWinAlign(s, x_arr, tr_arr, eid_short, 32, true);
    defer _ = mlx.mlx_array_free(aligned_short);
    var cs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cs);
    var ca = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ca);
    try mlx.check(mlx.mlx_contiguous(&cs, stride_short, false, s));
    try mlx.check(mlx.mlx_contiguous(&ca, aligned_short, false, s));
    try mlx.check(mlx.mlx_array_eval(cs));
    try mlx.check(mlx.mlx_array_eval(ca));
    const as = mlx.mlx_array_data_float16(cs) orelse return error.F16Unreadable;
    const aa = mlx.mlx_array_data_float16(ca) orelse return error.F16Unreadable;
    for (0..n * dim) |j| {
        const bs: u16 = @bitCast(as[j]);
        const ba: u16 = @bitCast(aa[j]);
        try t.expectEqual(bs, ba);
    }
}

test "exl3 sorted GEMM 32-row windows match 16-row per row" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 40;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(97);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    var eids: [40]u32 = undefined;
    var i: usize = 0;
    while (i < 24) : (i += 1) eids[i] = 1;
    while (i < 30) : (i += 1) eids[i] = 0;
    while (i < n) : (i += 1) eids[i] = 2;
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const got16 = try innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 16);
    defer _ = mlx.mlx_array_free(got16);
    const got32 = try innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 32);
    defer _ = mlx.mlx_array_free(got32);
    var c16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c16);
    var c32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c32);
    try mlx.check(mlx.mlx_contiguous(&c16, got16, false, s));
    try mlx.check(mlx.mlx_contiguous(&c32, got32, false, s));
    try mlx.check(mlx.mlx_array_eval(c16));
    try mlx.check(mlx.mlx_array_eval(c32));
    const a16 = mlx.mlx_array_data_float16(c16) orelse return error.F16Unreadable;
    const a32 = mlx.mlx_array_data_float16(c32) orelse return error.F16Unreadable;
    for (0..n * dim) |j| {
        const b16: u16 = @bitCast(a16[j]);
        const b32: u16 = @bitCast(a32[j]);
        try t.expectEqual(b16, b32);
    }
}

test "exl3 window 16 vs 32 production C=2048" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: c_int = 512;
    const R: c_int = 2048;
    const topk: c_int = 10;
    const H: c_int = 2560;
    const I: c_int = 640;
    const tr_n: usize = @intCast(E * (H / 16) * (I / 16) * 64);
    const tr_g = try alloc.alloc(u16, tr_n);
    const xh = try alloc.alloc(u16, @intCast(R * H));
    const slots_h = try alloc.alloc(u32, @intCast(R * topk));
    var prng = std.Random.DefaultPrng.init(99);
    const rnd = prng.random();
    for (tr_g) |*v| v.* = @truncate(rnd.int(u32));
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    for (slots_h) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ R, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{R * topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const trg = mlx.mlx_array_new_data(tr_g.ptr, &[_]c_int{ E, H / 16, I / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trg);
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots, 0, s));
    var sorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted);
    try mlx.check(mlx.mlx_take_axis(&sorted, slots, order, 0, s));
    const xr = try repeatRows(s, x_arr, R, topk);
    defer _ = mlx.mlx_array_free(xr);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_contiguous(&sc, sorted, false, s));
    try mlx.check(mlx.mlx_array_eval(sc));
    const nslots: usize = @intCast(R * topk);
    const ids = try alloc.alloc(u32, nslots);
    const sp = mlx.mlx_array_data_uint32(sc) orelse return error.F16Unreadable;
    @memcpy(ids, sp[0..nslots]);
    const io = std.Io.Threaded.global_single_threaded.io();
    const arms = [_]struct { win: c_int, aligned: bool, name: []const u8 }{
        .{ .win = 16, .aligned = false, .name = "stride-16" },
        .{ .win = 16, .aligned = true, .name = "aligned-16" },
        .{ .win = 32, .aligned = true, .name = "aligned-32" },
        .{ .win = 32, .aligned = false, .name = "stride-32" },
    };
    for (arms) |arm| {
        const st = windowStats(ids, @intCast(arm.win), arm.aligned);
        const warm = try innerGemmSortedWinAlign(s, xr, trg, sorted, arm.win, arm.aligned);
        try mlx.check(mlx.mlx_array_eval(warm));
        _ = mlx.mlx_array_free(warm);
        var sw = io_util.Stopwatch.init(io);
        const got = try innerGemmSortedWinAlign(s, xr, trg, sorted, arm.win, arm.aligned);
        try mlx.check(mlx.mlx_array_eval(got));
        const ns = sw.read();
        _ = mlx.mlx_array_free(got);
        benchPrint("C=2048 {s} {d} us nwin={d} mixed={d} decodes={d}\n", .{
            arm.name, ns / 1000, st.nwin, st.mixed, st.decodes,
        });
        try t.expect(ns > 0);
    }
    const R8: c_int = 8192;
    const xh8 = try alloc.alloc(u16, @intCast(R8 * H));
    const slots8 = try alloc.alloc(u32, @intCast(R8 * topk));
    for (xh8) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    for (slots8) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
    const x8 = mlx.mlx_array_new_data(xh8.ptr, &[_]c_int{ R8, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(x8);
    const sl8 = mlx.mlx_array_new_data(slots8.ptr, &[_]c_int{R8 * topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(sl8);
    var order8 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order8);
    try mlx.check(mlx.mlx_argsort_axis(&order8, sl8, 0, s));
    var sorted8 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted8);
    try mlx.check(mlx.mlx_take_axis(&sorted8, sl8, order8, 0, s));
    const xr8 = try repeatRows(s, x8, R8, topk);
    defer _ = mlx.mlx_array_free(xr8);
    var sc8 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc8);
    try mlx.check(mlx.mlx_contiguous(&sc8, sorted8, false, s));
    try mlx.check(mlx.mlx_array_eval(sc8));
    const n8: usize = @intCast(R8 * topk);
    const ids8 = try alloc.alloc(u32, n8);
    const sp8 = mlx.mlx_array_data_uint32(sc8) orelse return error.F16Unreadable;
    @memcpy(ids8, sp8[0..n8]);
    for (arms) |arm| {
        const st = windowStats(ids8, @intCast(arm.win), arm.aligned);
        const warm = try innerGemmSortedWinAlign(s, xr8, trg, sorted8, arm.win, arm.aligned);
        try mlx.check(mlx.mlx_array_eval(warm));
        _ = mlx.mlx_array_free(warm);
        var sw = io_util.Stopwatch.init(io);
        const got = try innerGemmSortedWinAlign(s, xr8, trg, sorted8, arm.win, arm.aligned);
        try mlx.check(mlx.mlx_array_eval(got));
        const ns = sw.read();
        _ = mlx.mlx_array_free(got);
        benchPrint("C=8192 {s} {d} us nwin={d} mixed={d} decodes={d}\n", .{
            arm.name, ns / 1000, st.nwin, st.mixed, st.decodes,
        });
        try t.expect(ns > 0);
    }
}

test "exl3 moePrefill matches staged sorted chain" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const suh_meta = parsed.value.object.get("suh").?.object;
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const svh_meta = parsed.value.object.get("svh").?.object;
    const v0: usize = @intCast(svh_meta.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(svh_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: c_int = 4;
    const R: c_int = 8;
    const topk: c_int = 2;
    const dim: c_int = 128;
    const stacked_t = try alloc.alloc(u16, @intCast(E * 8 * 8 * 64));
    const stacked_suh = try alloc.alloc(u16, @intCast(E * dim));
    const stacked_svh = try alloc.alloc(u16, @intCast(E * dim));
    var e_i: c_int = 0;
    while (e_i < E) : (e_i += 1) {
        const tb: usize = @intCast(e_i);
        @memcpy(stacked_t[tb * trellis_bits.len ..][0..trellis_bits.len], trellis_bits);
        @memcpy(stacked_suh[tb * suh_bits.len ..][0..suh_bits.len], suh_bits);
        @memcpy(stacked_svh[tb * svh_bits.len ..][0..svh_bits.len], svh_bits);
    }
    var prng = std.Random.DefaultPrng.init(71);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, @intCast(R * dim));
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, @intCast(R * topk));
    for (slots_h) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
    const scores_h = try alloc.alloc(f32, @intCast(R * topk));
    for (scores_h) |*v| v.* = 0.25 + rnd.float(f32) * 0.5;
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ R, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{R * topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{R * topk}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ E, 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(stacked_svh.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const got = try moePrefill(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, topk);
    defer _ = mlx.mlx_array_free(got);
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots, 0, s));
    var order_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order_u);
    try mlx.check(mlx.mlx_astype(&order_u, order, .uint32, s));
    var sorted_slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_slots);
    try mlx.check(mlx.mlx_take_axis(&sorted_slots, slots, order, 0, s));
    const g_prep = try prepareFromTokens(s, x_arr, suh, sorted_slots, order_u, dim, R * topk, topk);
    defer _ = mlx.mlx_array_free(g_prep);
    const u_prep = try prepareFromTokens(s, x_arr, suh, sorted_slots, order_u, dim, R * topk, topk);
    defer _ = mlx.mlx_array_free(u_prep);
    const g_inner = try innerGemmSorted(s, g_prep, tr, sorted_slots);
    defer _ = mlx.mlx_array_free(g_inner);
    const u_inner = try innerGemmSorted(s, u_prep, tr, sorted_slots);
    defer _ = mlx.mlx_array_free(u_inner);
    const g = try finishIndexed(s, g_inner, svh, sorted_slots);
    defer _ = mlx.mlx_array_free(g);
    const u = try finishIndexed(s, u_inner, svh, sorted_slots);
    defer _ = mlx.mlx_array_free(u);
    // The arm holds the SwiGLU product wide, so the staged chain must too:
    // an f16 `silu(g) * u` is the store this bar exists to keep out.
    var g32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g32);
    try mlx.check(mlx.mlx_astype(&g32, g, .float32, s));
    var u32a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u32a);
    try mlx.check(mlx.mlx_astype(&u32a, u, .float32, s));
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, g32, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, g32, sig, s));
    var h32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h32);
    try mlx.check(mlx.mlx_multiply(&h32, silu, u32a, s));
    var h = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_astype(&h, h32, .float16, s));
    const d_sorted = try projectSortedWithRuns(s, h, tr, suh, svh, sorted_slots);
    defer _ = mlx.mlx_array_free(d_sorted);
    var inv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv);
    try mlx.check(mlx.mlx_argsort_axis(&inv, order, 0, s));
    var inv_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv_u);
    try mlx.check(mlx.mlx_astype(&inv_u, inv, .uint32, s));
    const ref = try tokenReduce(s, d_sorted, inv_u, scores, dim, R, topk);
    defer _ = mlx.mlx_array_free(ref);
    var cg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cg);
    var cr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cr);
    try mlx.check(mlx.mlx_contiguous(&cg, got, false, s));
    try mlx.check(mlx.mlx_contiguous(&cr, ref, false, s));
    try mlx.check(mlx.mlx_array_eval(cg));
    try mlx.check(mlx.mlx_array_eval(cr));
    const ag = mlx.mlx_array_data_float16(cg) orelse return error.F16Unreadable;
    const ar = mlx.mlx_array_data_float16(cr) orelse return error.F16Unreadable;
    const n: usize = @intCast(R * dim);
    // The arm carries the whole SwiGLU in registers and rounds once; the staged
    // chain lands every stage in f16. An ULP bar would measure that, not the
    // chain, so the bar is an envelope.
    try expectRelRms(ag[0..n], ar[0..n], 0.005);
}

test "exl3 512-row E=512 topk=10 layer within 2x affine" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: c_int = 512;
    const R: c_int = 512;
    const topk: c_int = 10;
    const H: c_int = 2560;
    const I: c_int = 640;
    const tr_g_n: usize = @intCast(E * (H / 16) * (I / 16) * 64);
    const tr_d_n: usize = @intCast(E * (I / 16) * (H / 16) * 64);
    const tr_g = try alloc.alloc(u16, tr_g_n);
    const tr_d = try alloc.alloc(u16, tr_d_n);
    const suh_g = try alloc.alloc(u16, @intCast(E * H));
    const svh_g = try alloc.alloc(u16, @intCast(E * I));
    const suh_d = try alloc.alloc(u16, @intCast(E * I));
    const svh_d = try alloc.alloc(u16, @intCast(E * H));
    const xh = try alloc.alloc(u16, @intCast(R * H));
    const slots_h = try alloc.alloc(u32, @intCast(R * topk));
    const scores_h = try alloc.alloc(f32, @intCast(R * topk));
    var prng = std.Random.DefaultPrng.init(53);
    const rnd = prng.random();
    for (tr_g) |*v| v.* = @truncate(rnd.int(u32));
    for (tr_d) |*v| v.* = @truncate(rnd.int(u32));
    for (suh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (suh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    for (slots_h) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
    for (scores_h) |*v| v.* = 0.5;
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ R, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{R * topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{R * topk}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const trg = mlx.mlx_array_new_data(tr_g.ptr, &[_]c_int{ E, H / 16, I / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trg);
    const trd = mlx.mlx_array_new_data(tr_d.ptr, &[_]c_int{ E, I / 16, H / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trd);
    const sugh = mlx.mlx_array_new_data(suh_g.ptr, &[_]c_int{ E, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(sugh);
    const svgi = mlx.mlx_array_new_data(svh_g.ptr, &[_]c_int{ E, I }, 2, .float16);
    defer _ = mlx.mlx_array_free(svgi);
    const sudi = mlx.mlx_array_new_data(suh_d.ptr, &[_]c_int{ E, I }, 2, .float16);
    defer _ = mlx.mlx_array_free(sudi);
    const svdh = mlx.mlx_array_new_data(svh_d.ptr, &[_]c_int{ E, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(svdh);
    const warm = try moePrefill(s, x_arr, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots, scores, topk);
    try mlx.check(mlx.mlx_array_eval(warm));
    _ = mlx.mlx_array_free(warm);
    var t_g = io_util.Stopwatch.init(t.io);
    var it: usize = 0;
    while (it < 3) : (it += 1) {
        const out = try moePrefill(s, x_arr, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots, scores, topk);
        try mlx.check(mlx.mlx_array_eval(out));
        _ = mlx.mlx_array_free(out);
    }
    const gemm_ns = t_g.read() / 3;
    var dense_g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense_g);
    try mlx.check(mlx.mlx_random_normal(&dense_g, &[_]c_int{ E, I, H }, 3, .float16, 0, 1, .{ .ctx = null }, s));
    var w_cg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_cg);
    try mlx.check(mlx.mlx_contiguous(&w_cg, dense_g, false, s));
    var triple_g = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple_g);
    try mlx.check(mlx.mlx_quantize(&triple_g, w_cg, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
    var wqg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wqg);
    var wscg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wscg);
    var wbig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbig);
    try mlx.check(mlx.mlx_vector_array_get(&wqg, triple_g, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wscg, triple_g, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbig, triple_g, 2));
    var dense_d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense_d);
    try mlx.check(mlx.mlx_random_normal(&dense_d, &[_]c_int{ E, H, I }, 3, .float16, 0, 1, .{ .ctx = null }, s));
    var w_cd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_cd);
    try mlx.check(mlx.mlx_contiguous(&w_cd, dense_d, false, s));
    var triple_d = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple_d);
    try mlx.check(mlx.mlx_quantize(&triple_d, w_cd, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
    var wqd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wqd);
    var wscd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wscd);
    var wbid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbid);
    try mlx.check(mlx.mlx_vector_array_get(&wqd, triple_d, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wscd, triple_d, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbid, triple_d, 2));
    const xr = try repeatRows(s, x_arr, R, topk);
    defer _ = mlx.mlx_array_free(xr);
    var xrep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xrep);
    try mlx.check(mlx.mlx_reshape(&xrep, xr, &[_]c_int{ R * topk, 1, H }, 3, s));
    var xdi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xdi);
    try mlx.check(mlx.mlx_random_normal(&xdi, &[_]c_int{ R * topk, 1, I }, 3, .float16, 0, 1, .{ .ctx = null }, s));
    var slots_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slots_i);
    try mlx.check(mlx.mlx_astype(&slots_i, slots, .int32, s));
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots_i, 0, s));
    var sorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted);
    try mlx.check(mlx.mlx_take_axis(&sorted, slots_i, order, 0, s));
    const no_idx = mlx.mlx_array{ .ctx = null };
    var q0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q0);
    try mlx.check(mlx.mlx_gather_qmm(&q0, xrep, wqg, wscg, wbig, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
    try mlx.check(mlx.mlx_array_eval(q0));
    var t_q = io_util.Stopwatch.init(t.io);
    it = 0;
    while (it < 3) : (it += 1) {
        var qg = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_gather_qmm(&qg, xrep, wqg, wscg, wbig, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
        try mlx.check(mlx.mlx_array_eval(qg));
        _ = mlx.mlx_array_free(qg);
        var qu = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_gather_qmm(&qu, xrep, wqg, wscg, wbig, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
        try mlx.check(mlx.mlx_array_eval(qu));
        _ = mlx.mlx_array_free(qu);
        var qd = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_gather_qmm(&qd, xdi, wqd, wscd, wbid, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
        try mlx.check(mlx.mlx_array_eval(qd));
        _ = mlx.mlx_array_free(qd);
    }
    const affine_ns = t_q.read() / 3;
    const ratio_x100: u64 = if (affine_ns == 0) 0 else (gemm_ns * 100) / affine_ns;
    benchPrint("exl3 512-row E=512 H=2560 I=640 topk=10: layer {d} us  affine-3x-gather_qmm {d} us  ratio {d}/100\n", .{
        gemm_ns / 1000,
        affine_ns / 1000,
        ratio_x100,
    });
    try t.expect(ratio_x100 <= 200);
}

test "exl3 fused decode chain matches the indexed chain across the top-k range" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const tm = parsed.value.object.get("trellis").?.object;
    const sm = parsed.value.object.get("suh").?.object;
    const vm = parsed.value.object.get("svh").?.object;
    const t0: usize = @intCast(tm.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(tm.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(sm.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(sm.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(vm.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(vm.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: usize = 7;
    const dim: usize = 128;
    const tile_n = 8 * 8 * 64;
    const st = try alloc.alloc(u16, E * tile_n);
    const su = try alloc.alloc(u16, E * dim);
    const sv = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(st[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(su[e * dim ..][0..dim], suh_bits);
        @memcpy(sv[e * dim ..][0..dim], svh_bits);
    }
    const tr = mlx.mlx_array_new_data(st.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(su.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    var prng = std.Random.DefaultPrng.init(131);
    const rnd = prng.random();
    // The reduce bank is per (row, k) slot: a top-k past its width silently read
    // another slot's partial. Bar is 10x the f16 floor these shapes agree at.
    for ([_]usize{ 8, 16, 17, 20, 32 }) |topk| {
        const xh = try alloc.alloc(u16, dim);
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
        const sl = try alloc.alloc(u32, topk);
        const sc = try alloc.alloc(f32, topk);
        for (sl) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
        for (sc) |*v| v.* = rnd.float(f32);
        const xa = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{@intCast(dim)}, 1, .float16);
        defer _ = mlx.mlx_array_free(xa);
        const sa = mlx.mlx_array_new_data(sl.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(sa);
        const ca = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
        defer _ = mlx.mlx_array_free(ca);
        const fused = try moeSwigluFused(s, xa, tr, suh, svh, tr, suh, svh, tr, suh, svh, sa, ca, .float16);
        defer _ = mlx.mlx_array_free(fused);
        const ref = try moeSwigluIndexed(s, xa, tr, suh, svh, tr, suh, svh, tr, suh, svh, sa, ca);
        defer _ = mlx.mlx_array_free(ref);
        var cf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cf);
        var cr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cr);
        try mlx.check(mlx.mlx_contiguous(&cf, fused, false, s));
        try mlx.check(mlx.mlx_contiguous(&cr, ref, false, s));
        try mlx.check(mlx.mlx_array_eval(cf));
        try mlx.check(mlx.mlx_array_eval(cr));
        const af = mlx.mlx_array_data_float16(cf) orelse return error.F16Unreadable;
        const ar = mlx.mlx_array_data_float16(cr) orelse return error.F16Unreadable;
        var ss: f64 = 0;
        var refsq: f64 = 0;
        for (0..dim) |j| {
            const a = exl3.f16BitsToF32(@bitCast(af[j]));
            const b = exl3.f16BitsToF32(@bitCast(ar[j]));
            ss += @as(f64, a - b) * @as(f64, a - b);
            refsq += @as(f64, b) * @as(f64, b);
        }
        const rel = @sqrt(ss / @max(refsq, 1e-20));
        if (!(rel < 0.01)) {
            std.debug.print("exl3 topk={d} rel_rms={d:.6}\n", .{ topk, rel });
            return error.TestExpectedEqual;
        }
    }
}

test "exl3 prefill arm matches the fused decode arm across row counts and top-k" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const tm = parsed.value.object.get("trellis").?.object;
    const sm = parsed.value.object.get("suh").?.object;
    const vm = parsed.value.object.get("svh").?.object;
    const t0: usize = @intCast(tm.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(tm.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(sm.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(sm.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(vm.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(vm.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: usize = 7;
    const dim: usize = 128;
    const tile_n = 8 * 8 * 64;
    const st = try alloc.alloc(u16, E * tile_n);
    const su = try alloc.alloc(u16, E * dim);
    const sv = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(st[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(su[e * dim ..][0..dim], suh_bits);
        @memcpy(sv[e * dim ..][0..dim], svh_bits);
    }
    const tr = mlx.mlx_array_new_data(st.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(su.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    var prng = std.Random.DefaultPrng.init(179);
    const rnd = prng.random();
    // Row counts either side of the 16-row window, a tail that does not fill a
    // window, and a run that spans one: the two arms must answer the same rows.
    for ([_][2]usize{ .{ 1, 1 }, .{ 3, 2 }, .{ 5, 7 }, .{ 16, 10 }, .{ 17, 3 }, .{ 31, 5 }, .{ 33, 1 }, .{ 64, 6 } }) |c| {
        const rows = c[0];
        const topk = c[1];
        const xh = try alloc.alloc(u16, rows * dim);
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
        const sl = try alloc.alloc(u32, rows * topk);
        const sc = try alloc.alloc(f32, rows * topk);
        for (sl) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
        for (sc) |*v| v.* = rnd.float(f32);
        const xa = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), @intCast(dim) }, 2, .float16);
        defer _ = mlx.mlx_array_free(xa);
        const sa = mlx.mlx_array_new_data(sl.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(sa);
        const ca = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .float32);
        defer _ = mlx.mlx_array_free(ca);
        const dec = try moeSwigluFused(s, xa, tr, suh, svh, tr, suh, svh, tr, suh, svh, sa, ca, .float16);
        defer _ = mlx.mlx_array_free(dec);
        const pre = try moePrefill(s, xa, tr, suh, svh, tr, suh, svh, tr, suh, svh, sa, ca, @intCast(topk));
        defer _ = mlx.mlx_array_free(pre);
        var cd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cd);
        var cp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cp);
        try mlx.check(mlx.mlx_contiguous(&cd, dec, false, s));
        try mlx.check(mlx.mlx_contiguous(&cp, pre, false, s));
        try mlx.check(mlx.mlx_array_eval(cd));
        try mlx.check(mlx.mlx_array_eval(cp));
        const ad = mlx.mlx_array_data_float16(cd) orelse return error.F16Unreadable;
        const ap = mlx.mlx_array_data_float16(cp) orelse return error.F16Unreadable;
        try expectRelRms(ad[0 .. rows * dim], ap[0 .. rows * dim], 0.01);
    }
}

fn fusedChainMatchesHost(fixture: []const u8, rate: exl3.Rate, dec: exl3.Decode) !void {
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const tm = parsed.value.object.get("trellis").?.object;
    const sm = parsed.value.object.get("suh").?.object;
    const vm = parsed.value.object.get("svh").?.object;
    const t0: usize = @intCast(tm.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(tm.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(sm.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(sm.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(vm.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(vm.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: usize = 4;
    const dim: usize = 128;
    const topk: usize = 4;
    const tile_n = 8 * 8 * rate.halfwords();
    const st = try alloc.alloc(u16, E * tile_n);
    const su = try alloc.alloc(u16, E * dim);
    const sv = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(st[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(su[e * dim ..][0..dim], suh_bits);
        @memcpy(sv[e * dim ..][0..dim], svh_bits);
    }
    var prng = std.Random.DefaultPrng.init(211);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, dim);
    const xf = try alloc.alloc(f32, dim);
    for (xh, xf) |*b, *v| {
        b.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
        v.* = exl3.f16BitsToF32(b.*);
    }
    const sl = try alloc.alloc(u32, topk);
    const sc = try alloc.alloc(f32, topk);
    for (sl, 0..) |*v, i| v.* = @intCast(i % E);
    for (sc) |*v| v.* = rnd.float(f32);
    const xa = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{@intCast(dim)}, 1, .float16);
    defer _ = mlx.mlx_array_free(xa);
    const tr = mlx.mlx_array_new_data(st.ptr, &[_]c_int{ @intCast(E), 8, 8, @intCast(rate.halfwords()) }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(su.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const sa = mlx.mlx_array_new_data(sl.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(sa);
    const ca = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(ca);
    const fused = try moeSwigluFused(s, xa, tr, suh, svh, tr, suh, svh, tr, suh, svh, sa, ca, .float16);
    defer _ = mlx.mlx_array_free(fused);
    var cf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cf);
    try mlx.check(mlx.mlx_contiguous(&cf, fused, false, s));
    try mlx.check(mlx.mlx_array_eval(cf));
    const af = mlx.mlx_array_data_float16(cf) orelse return error.F16Unreadable;
    const want = try moeSwigluHost(alloc, xf, st, su, sv, st, su, sv, st, su, sv, sl, sc, dim, dim, rate.halfwords(), 8, 8, dec);
    var ss: f64 = 0;
    var ref: f64 = 0;
    for (0..dim) |i| {
        const a: f64 = @floatCast(af[i]);
        const b: f64 = want[i];
        ss += (a - b) * (a - b);
        ref += b * b;
    }
    const rel = @sqrt(ss / @max(ref, 1e-20));
    if (!(rel < 0.005)) {
        std.debug.print("exl3 host oracle rel_rms={d:.6}\n", .{rel});
        return error.TestExpectedEqual;
    }
}

test "exl3 fused decode chain matches the host SwiGLU reference" {
    try fusedChainMatchesHost(exl3.fixtures.k4, exl3.Rate.fromK(4), .mul1);
}

test "exl3 fused decode chain matches the host SwiGLU reference under TINY" {
    try fusedChainMatchesHost(exl3.fixtures.k4, exl3.Rate.fromK(4), .tiny);
}

test "exl3 fused decode chain matches the host SwiGLU reference at K2.5 TINY" {
    try fusedChainMatchesHost(exl3.fixtures.k2p5_tiny, .{ .n = 40 }, .tiny);
}

test "exl3 fused decode chain matches the host SwiGLU reference at a narrowed codeword window" {
    try fusedChainMatchesHost(exl3.fixtures.k2p5_tiny, .{ .n = 40 }, .{ .codebook = .tiny, .window = .w12 });
    try fusedChainMatchesHost(exl3.fixtures.k2p5_tiny_w12, .{ .n = 40 }, .{ .codebook = .tiny, .window = .w12 });
    try fusedChainMatchesHost(exl3.fixtures.k4, exl3.Rate.fromK(4), .{ .codebook = .mul1, .window = .w14 });
}

test "exl3 fused decode chain matches the host SwiGLU reference below window 12" {
    try fusedChainMatchesHost(exl3.fixtures.k2p5_tiny, .{ .n = 40 }, .{ .codebook = .tiny, .window = .w10 });
}

test "exl3 cooperative indexed GEMV matches host TINY tile decode at K4 K3 K2" {
    for (0..PARITY_SEEDS) |i| {
        try indexedParity(exl3.Rate.fromK(4), 128, 128, 4, 10, 23 + i, .tiny);
        try indexedParity(exl3.Rate.fromK(3), 128, 128, 4, 10, 1201 + i, .tiny);
        try indexedParity(exl3.Rate.fromK(2), 128, 128, 4, 10, 1301 + i, .tiny);
    }
}

test "exl3 cooperative indexed GEMV matches the host tile decode at a fractional rate" {
    for (0..PARITY_SEEDS) |i| {
        try indexedParity(.{ .n = 40 }, 128, 128, 4, 10, 1401 + i, .tiny);
        try indexedParity(.{ .n = 44 }, 128, 128, 4, 10, 1501 + i, .tiny);
    }
    try indexedParity(.{ .n = 40 }, 2560, 640, 4, 10, 43, .mul1);
    try indexedParity(.{ .n = 44 }, 2560, 640, 4, 10, 47, .mul1);
}

/// The w12 fixture through the Metal indexed GEMV, scored against PonyExl3's
/// own reference decode (the fixture's `inner`) rather than against our host
/// decoder — the one bar that certifies the narrowed-window convention on the
/// GPU end to end.
fn indexedGemvMatchesFixtureInner(fixture: []const u8, rate: exl3.Rate, dec: exl3.Decode) !void {
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, fixture[8 .. 8 + header_len], .{});
    defer parsed.deinit();
    const data = fixture[8 + header_len ..];
    const span = struct {
        fn of(root: std.json.Value, name: []const u8, blob: []const u8) []align(1) const u16 {
            const off = root.object.get(name).?.object.get("data_offsets").?.array.items;
            const a: usize = @intCast(off[0].integer);
            const b: usize = @intCast(off[1].integer);
            return std.mem.bytesAsSlice(u16, blob[a..b]);
        }
    };
    const trellis_bits = span.of(parsed.value, "trellis", data);
    const inner_bits = span.of(parsed.value, "inner", data);

    const E: usize = 3;
    const dim: usize = 128;
    const rows: usize = 6;
    const tile_n = 8 * 8 * rate.halfwords();
    const stacked = try alloc.alloc(u16, E * tile_n);
    const w = try alloc.alloc(u16, E * dim * dim);
    for (0..E) |e| {
        @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(w[e * dim * dim ..][0 .. dim * dim], inner_bits);
    }
    var prng = std.Random.DefaultPrng.init(307);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, rows * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const eids = try alloc.alloc(u32, rows);
    for (eids, 0..) |*v, i| v.* = @intCast(i % E);

    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, @intCast(rate.halfwords()) }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const slots = mlx.mlx_array_new_data(eids.ptr, &[_]c_int{@intCast(rows)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const got = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try reportGemmParity(try measureInnerGemmParityOn(alloc, s, src[0 .. rows * dim], xh, eids, w, dim, dim));
}

test "exl3 indexed GEMV decodes the w12 fixture to PonyExl3's own inner weights" {
    try indexedGemvMatchesFixtureInner(exl3.fixtures.k2p5_tiny_w12, .{ .n = 40 }, .{ .codebook = .tiny, .window = .w12 });
}

test "exl3 cooperative indexed GEMV matches the host tile decode at a narrowed codeword window" {
    for (0..PARITY_SEEDS) |i| {
        try indexedParity(.{ .n = 40 }, 128, 128, 4, 10, 1601 + i, .{ .codebook = .tiny, .window = .w12 });
        // K4 takes the packed fast branch, which decodes through the same helper.
        try indexedParity(exl3.Rate.fromK(4), 128, 128, 4, 10, 1701 + i, .{ .codebook = .mul1, .window = .w12 });
        try indexedParity(.{ .n = 40 }, 128, 128, 4, 10, 1801 + i, .{ .codebook = .tiny, .window = .w14 });
    }
}

test "exl3 cooperative indexed GEMV matches the host tile decode below window 12" {
    for (0..PARITY_SEEDS) |i| {
        try indexedParity(.{ .n = 40 }, 128, 128, 4, 10, 1901 + i, .{ .codebook = .tiny, .window = .w10 });
    }
}

/// One sorted-GEMM arm (NAX where the shape and silicon allow, else the SIMD
/// body) against the host tile decode, at whatever rate the trellis names.
fn sortedGemmParityStats(rate: exl3.Rate, dec: exl3.Decode, win: c_int, seed: u64, mutate: Exl3Mutation) !Exl3GemmParityStats {
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 4;
    const dim: usize = 128;
    const tiles = dim / 16;
    const packed_n = rate.halfwords();
    const tile_n = tiles * tiles * packed_n;
    const stacked = try alloc.alloc(u16, E * tile_n);
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    for (stacked) |*v| v.* = @truncate(rnd.int(u32));
    const runs = [_]u32{ 7, 32, 5, 18, 40, 10 };
    var rows: usize = 0;
    for (runs) |r| rows += r;
    const eids = try alloc.alloc(u32, rows);
    {
        var off: usize = 0;
        for (runs, 0..) |r, ei| {
            var j: usize = 0;
            while (j < r) : (j += 1) eids[off + j] = @intCast(ei % E);
            off += r;
        }
    }
    const xh = try alloc.alloc(u16, rows * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), @intCast(tiles), @intCast(tiles), @intCast(packed_n) }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(eids.ptr, &[_]c_int{@intCast(rows)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const got = try innerGemmSortedWin(s, x_arr, tr_arr, eid_a, win);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    if (mutate == .codeword) stacked[tile_n / 2] ^= 0x40;
    return measureInnerGemmParity(alloc, s, src[0 .. rows * dim], xh, eids, stacked, dim, dim, rate, mutatedDecode(dec, mutate));
}

fn sortedGemmParity(rate: exl3.Rate, dec: exl3.Decode, win: c_int, seed: u64) !void {
    try reportGemmParity(try sortedGemmParityStats(rate, dec, win, seed, .none));
}

/// Seeds are a sample, not a choice: the bar holds for any trellis, so every
/// case sweeps a fixed run of them rather than one that happened to pass.
const PARITY_SEEDS: usize = 8;

test "exl3 sorted GEMM matches the host tile decode at a fractional rate" {
    for (0..PARITY_SEEDS) |i| {
        try sortedGemmParity(.{ .n = 40 }, .tiny, 32, 101 + i);
        try sortedGemmParity(.{ .n = 44 }, .mul1, 32, 201 + i);
        try sortedGemmParity(exl3.Rate.fromK(4), .mul1, 16, 301 + i);
    }
}

test "exl3 sorted GEMM matches the host tile decode at a fractional rate with the NAX arm off" {
    const t = std.testing;
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    if (!gemmNaxOn()) return error.SkipZigTest;
    _ = setenv("MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK", "1", 1);
    defer _ = unsetenv("MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK");
    try t.expect(!gemmNaxOn());
    for (0..PARITY_SEEDS) |i| {
        try sortedGemmParity(.{ .n = 40 }, .tiny, 32, 401 + i);
        try sortedGemmParity(.{ .n = 44 }, .mul1, 16, 501 + i);
    }
}

test "exl3 sorted GEMM matches the host tile decode at a narrowed codeword window" {
    for (0..PARITY_SEEDS) |i| {
        try sortedGemmParity(.{ .n = 40 }, .{ .codebook = .tiny, .window = .w12 }, 32, 601 + i);
        try sortedGemmParity(exl3.Rate.fromK(4), .{ .codebook = .mul1, .window = .w12 }, 16, 701 + i);
        try sortedGemmParity(.{ .n = 40 }, .{ .codebook = .tiny, .window = .w14 }, 16, 801 + i);
    }
}

test "exl3 the GEMM parity bar convicts a wrong window and a wrong codeword" {
    // The bar has to fail on a wrong arm, not only pass on a right one: run
    // the real kernel and give the reference a decode the arm did not use.
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    for ([_]Exl3Mutation{ .window, .codeword }) |m| {
        for ([_]exl3.Decode{ .tiny, .{ .codebook = .mul1, .window = .w12 } }) |dec| {
            const st = try sortedGemmParityStats(.{ .n = 40 }, dec, 16, 901, m);
            try std.testing.expect(exl3GemmParityVerdict(st.finite, st.kern_max, st.rms_kern, st.rms_comp, st.ceiling) != null);
        }
        // Again at the production reduction width, where the ceiling is widest.
        const wide = try indexedParityStats(exl3.Rate.fromK(4), 2560, 640, 4, 10, 907, .mul1, m);
        try std.testing.expect(exl3GemmParityVerdict(wide.finite, wide.kern_max, wide.rms_kern, wide.rms_comp, wide.ceiling) != null);
    }
}

test "exl3 sorted GEMM matches the host tile decode at a narrowed window with the NAX arm off" {
    const t = std.testing;
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    if (!gemmNaxOn()) return error.SkipZigTest;
    _ = setenv("MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK", "1", 1);
    defer _ = unsetenv("MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK");
    try t.expect(!gemmNaxOn());
    for (0..PARITY_SEEDS) |i| {
        try sortedGemmParity(.{ .n = 40 }, .{ .codebook = .tiny, .window = .w12 }, 32, 1001 + i);
        try sortedGemmParity(exl3.Rate.fromK(4), .{ .codebook = .mul1, .window = .w14 }, 16, 1101 + i);
    }
}

test "exl3 decode and prefill arms agree with the indexed chain at production geometry" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 4;
    const H: usize = 2560;
    const I: usize = 640;
    const topk: usize = 4;
    const gu_n = (H / 16) * (I / 16) * 64;
    const d_n = (I / 16) * (H / 16) * 64;
    const tr_gu = try alloc.alloc(u16, E * gu_n);
    const tr_d = try alloc.alloc(u16, E * d_n);
    const suh_h = try alloc.alloc(u16, E * H);
    const svh_i = try alloc.alloc(u16, E * I);
    const suh_i = try alloc.alloc(u16, E * I);
    const svh_h = try alloc.alloc(u16, E * H);
    var prng = std.Random.DefaultPrng.init(233);
    const rnd = prng.random();
    for (tr_gu) |*v| v.* = @truncate(rnd.int(u32));
    for (tr_d) |*v| v.* = @truncate(rnd.int(u32));
    for (suh_h) |*v| v.* = exl3.f32ToF16Bits(0.5 + rnd.float(f32));
    for (svh_i) |*v| v.* = exl3.f32ToF16Bits(0.5 + rnd.float(f32));
    for (suh_i) |*v| v.* = exl3.f32ToF16Bits(0.5 + rnd.float(f32));
    for (svh_h) |*v| v.* = exl3.f32ToF16Bits(0.5 + rnd.float(f32));
    const tg = mlx.mlx_array_new_data(tr_gu.ptr, &[_]c_int{ @intCast(E), @intCast(H / 16), @intCast(I / 16), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tg);
    const td = mlx.mlx_array_new_data(tr_d.ptr, &[_]c_int{ @intCast(E), @intCast(I / 16), @intCast(H / 16), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(td);
    const sgh = mlx.mlx_array_new_data(suh_h.ptr, &[_]c_int{ @intCast(E), @intCast(H) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sgh);
    const vgi = mlx.mlx_array_new_data(svh_i.ptr, &[_]c_int{ @intCast(E), @intCast(I) }, 2, .float16);
    defer _ = mlx.mlx_array_free(vgi);
    const sdi = mlx.mlx_array_new_data(suh_i.ptr, &[_]c_int{ @intCast(E), @intCast(I) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sdi);
    const vdh = mlx.mlx_array_new_data(svh_h.ptr, &[_]c_int{ @intCast(E), @intCast(H) }, 2, .float16);
    defer _ = mlx.mlx_array_free(vdh);
    const sl1 = try alloc.alloc(u32, topk);
    const sc1 = try alloc.alloc(f32, topk);
    for (sl1, 0..) |*v, i| v.* = @intCast(i % E);
    for (sc1) |*v| v.* = 0.2 + rnd.float(f32) * 0.3;
    const x1h = try alloc.alloc(u16, H);
    for (x1h) |*v| v.* = exl3.f32ToF16Bits((rnd.float(f32) * 2 - 1) * 0.05);
    const x1 = mlx.mlx_array_new_data(x1h.ptr, &[_]c_int{@intCast(H)}, 1, .float16);
    defer _ = mlx.mlx_array_free(x1);
    const s1 = mlx.mlx_array_new_data(sl1.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(s1);
    const c1 = mlx.mlx_array_new_data(sc1.ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(c1);
    const fused1 = try moeSwigluFused(s, x1, tg, sgh, vgi, tg, sgh, vgi, td, sdi, vdh, s1, c1, .float16);
    defer _ = mlx.mlx_array_free(fused1);
    const ref1 = try moeSwigluIndexed(s, x1, tg, sgh, vgi, tg, sgh, vgi, td, sdi, vdh, s1, c1);
    defer _ = mlx.mlx_array_free(ref1);
    var cf1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cf1);
    var cr1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cr1);
    try mlx.check(mlx.mlx_contiguous(&cf1, fused1, false, s));
    try mlx.check(mlx.mlx_contiguous(&cr1, ref1, false, s));
    try mlx.check(mlx.mlx_array_eval(cf1));
    try mlx.check(mlx.mlx_array_eval(cr1));
    const af1 = mlx.mlx_array_data_float16(cf1) orelse return error.F16Unreadable;
    const ar1 = mlx.mlx_array_data_float16(cr1) orelse return error.F16Unreadable;
    try expectRelRms(af1[0..H], ar1[0..H], 0.01);
    const rows: usize = 20;
    const xnh = try alloc.alloc(u16, rows * H);
    for (xnh) |*v| v.* = exl3.f32ToF16Bits((rnd.float(f32) * 2 - 1) * 0.05);
    const sln = try alloc.alloc(u32, rows * topk);
    const scn = try alloc.alloc(f32, rows * topk);
    for (sln) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
    for (scn) |*v| v.* = 0.2 + rnd.float(f32) * 0.3;
    const xn = mlx.mlx_array_new_data(xnh.ptr, &[_]c_int{ @intCast(rows), @intCast(H) }, 2, .float16);
    defer _ = mlx.mlx_array_free(xn);
    const sn = mlx.mlx_array_new_data(sln.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(sn);
    const cn = mlx.mlx_array_new_data(scn.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(cn);
    const dec = try moeSwigluFused(s, xn, tg, sgh, vgi, tg, sgh, vgi, td, sdi, vdh, sn, cn, .float16);
    defer _ = mlx.mlx_array_free(dec);
    const pre = try moePrefill(s, xn, tg, sgh, vgi, tg, sgh, vgi, td, sdi, vdh, sn, cn, @intCast(topk));
    defer _ = mlx.mlx_array_free(pre);
    var cd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cd);
    var cp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cp);
    try mlx.check(mlx.mlx_contiguous(&cd, dec, false, s));
    try mlx.check(mlx.mlx_contiguous(&cp, pre, false, s));
    try mlx.check(mlx.mlx_array_eval(cd));
    try mlx.check(mlx.mlx_array_eval(cp));
    const ad = mlx.mlx_array_data_float16(cd) orelse return error.F16Unreadable;
    const ap = mlx.mlx_array_data_float16(cp) orelse return error.F16Unreadable;
    try expectRelRms(ad[0 .. rows * H], ap[0 .. rows * H], 0.01);
}

/// One MiMo-V2.6-Flash MoE layer's shape and routing on the GPU arms: hidden
/// != inter, every expert holding its OWN bank, and a routing that leaves some
/// experts unrouted while giving others more rows than one GEMM window holds.
const MimoMoeCase = struct {
    e: usize,
    hidden: usize,
    inter: usize,
    topk: usize,
    rows: usize,
    rate: exl3.Rate,
    dec: exl3.Decode,
    seed: u64,
    /// |suh_h|, |svh_i|, |suh_i|, |svh_h|. The default is the synthetic 0.9 the
    /// parity cases use; the served pack's own magnitudes are `MIMO_BANKS`.
    banks: [4]f32 = @splat(0.9),
    x_scale: f32 = 0.05,
    /// Raw `[E]`-sliced gate/up/down trellis then suh_h/svh_i/suh_i/svh_h, as
    /// the shards store them. Set, the fixture carries the PACK's own bytes.
    real_blob: ?[*:0]const u8 = null,
};

/// The served w12 pack's own scale magnitudes: |suh| is ~0.01 and |svh| ~1,
/// so the residual's size reaches the arm's f16 planes through the GEMMs.
const MIMO_BANKS = [4]f32{ 0.0126, 1.03, 0.0083, 1.009 };

fn readAll(fd: c_int, dst: []u8) !void {
    var got: usize = 0;
    while (got < dst.len) {
        const n = std.c.read(fd, dst.ptr + got, dst.len - got);
        if (n <= 0) return error.RealBlobShort;
        got += @intCast(n);
    }
}

const MimoMoeFixture = struct {
    arrays: [10]mlx.mlx_array,
    gate_t: []u16,
    up_t: []u16,
    down_t: []u16,
    suh_h: []u16,
    svh_i: []u16,
    suh_i: []u16,
    svh_h: []u16,
    xf: []f32,
    slots: []u32,
    scores: []f32,

    fn deinit(self: *MimoMoeFixture) void {
        for (self.arrays) |a| _ = mlx.mlx_array_free(a);
    }
};

/// Real routing: every row takes `topk` DISTINCT experts from a skewed draw, so
/// a few experts carry runs longer than a window and many carry none.
fn mimoRouting(alloc: std.mem.Allocator, rows: usize, topk: usize, e: usize, rnd: std.Random) ![]u32 {
    const out = try alloc.alloc(u32, rows * topk);
    const hot = @max(topk, e / 8);
    for (0..rows) |r| {
        const row = out[r * topk ..][0..topk];
        var k: usize = 0;
        while (k < topk) {
            const pick: u32 = if (rnd.float(f32) < 0.75)
                rnd.uintLessThan(u32, @intCast(hot))
            else
                rnd.uintLessThan(u32, @intCast(e));
            if (std.mem.indexOfScalar(u32, row[0..k], pick) != null) continue;
            row[k] = pick;
            k += 1;
        }
    }
    return out;
}

fn mimoMoeFixture(alloc: std.mem.Allocator, c: MimoMoeCase) !MimoMoeFixture {
    const n = c.rate.halfwords();
    const ith = c.hidden / 16;
    const iti = c.inter / 16;
    const gu_tile = ith * iti * n;
    var prng = std.Random.DefaultPrng.init(c.seed);
    const rnd = prng.random();
    const gate_t = try alloc.alloc(u16, c.e * gu_tile);
    const up_t = try alloc.alloc(u16, c.e * gu_tile);
    const down_t = try alloc.alloc(u16, c.e * gu_tile);
    for ([_][]u16{ gate_t, up_t, down_t }) |bank| {
        for (bank) |*v| v.* = @truncate(rnd.int(u32));
    }
    var real_fd: ?c_int = null;
    if (c.real_blob) |path| {
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.RealBlobOpen;
        real_fd = fd;
        for ([_][]u16{ gate_t, up_t, down_t }) |bank| try readAll(fd, std.mem.sliceAsBytes(bank));
    }
    const suh_h = try alloc.alloc(u16, c.e * c.hidden);
    const svh_i = try alloc.alloc(u16, c.e * c.inter);
    const suh_i = try alloc.alloc(u16, c.e * c.inter);
    const svh_h = try alloc.alloc(u16, c.e * c.hidden);
    for ([_][]u16{ suh_h, svh_i, suh_i, svh_h }, c.banks) |bank, mag| {
        for (bank) |*v| v.* = exl3.f32ToF16Bits(if (rnd.boolean()) mag else -mag);
    }
    if (real_fd) |fd| {
        for ([_][]u16{ suh_h, svh_i, suh_i, svh_h }) |bank| try readAll(fd, std.mem.sliceAsBytes(bank));
        _ = std.c.close(fd);
    }
    const xh = try alloc.alloc(u16, c.rows * c.hidden);
    const xf = try alloc.alloc(f32, c.rows * c.hidden);
    for (xh, xf) |*b, *v| {
        b.* = exl3.f32ToF16Bits((rnd.float(f32) * 2 - 1) * c.x_scale);
        v.* = exl3.f16BitsToF32(b.*);
    }
    const slots = try mimoRouting(alloc, c.rows, c.topk, c.e, rnd);
    const scores = try alloc.alloc(f32, c.rows * c.topk);
    for (scores) |*v| v.* = 0.05 + rnd.float(f32) * 0.3;
    const ci = struct {
        fn i(v: usize) c_int {
            return @intCast(v);
        }
    }.i;
    return .{
        .arrays = .{
            mlx.mlx_array_new_data(gate_t.ptr, &[_]c_int{ ci(c.e), ci(ith), ci(iti), ci(n) }, 4, .uint16),
            mlx.mlx_array_new_data(up_t.ptr, &[_]c_int{ ci(c.e), ci(ith), ci(iti), ci(n) }, 4, .uint16),
            mlx.mlx_array_new_data(down_t.ptr, &[_]c_int{ ci(c.e), ci(iti), ci(ith), ci(n) }, 4, .uint16),
            mlx.mlx_array_new_data(suh_h.ptr, &[_]c_int{ ci(c.e), ci(c.hidden) }, 2, .float16),
            mlx.mlx_array_new_data(svh_i.ptr, &[_]c_int{ ci(c.e), ci(c.inter) }, 2, .float16),
            mlx.mlx_array_new_data(suh_i.ptr, &[_]c_int{ ci(c.e), ci(c.inter) }, 2, .float16),
            mlx.mlx_array_new_data(svh_h.ptr, &[_]c_int{ ci(c.e), ci(c.hidden) }, 2, .float16),
            mlx.mlx_array_new_data(slots.ptr, &[_]c_int{ci(c.rows * c.topk)}, 1, .uint32),
            mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ ci(c.rows), ci(c.hidden) }, 2, .float16),
            mlx.mlx_array_new_data(scores.ptr, &[_]c_int{ci(c.rows * c.topk)}, 1, .float32),
        },
        .gate_t = gate_t,
        .up_t = up_t,
        .down_t = down_t,
        .suh_h = suh_h,
        .svh_i = svh_i,
        .suh_i = suh_i,
        .svh_h = svh_h,
        .xf = xf,
        .slots = slots,
        .scores = scores,
    };
}

fn mimoPrefillArm(s: mlx.mlx_stream, f: *const MimoMoeFixture, topk: usize) !mlx.mlx_array {
    const a = f.arrays;
    return moePrefill(s, a[8], a[0], a[3], a[4], a[1], a[3], a[4], a[2], a[5], a[6], a[7], a[9], @intCast(topk));
}

fn mimoDecodeArm(s: mlx.mlx_stream, f: *const MimoMoeFixture) !mlx.mlx_array {
    const a = f.arrays;
    return moeSwigluFused(s, a[8], a[0], a[3], a[4], a[1], a[3], a[4], a[2], a[5], a[6], a[7], a[9], .float16);
}

fn evalF16(s: mlx.mlx_stream, a: mlx.mlx_array, out: *mlx.mlx_array) ![*c]const f16 {
    try mlx.check(mlx.mlx_contiguous(out, a, false, s));
    try mlx.check(mlx.mlx_array_eval(out.*));
    return mlx.mlx_array_data_float16(out.*) orelse error.F16Unreadable;
}

/// The prefill arm against the host tile decode of the SAME routing: the only
/// oracle here that shares no kernel with what it scores.
fn mimoPrefillMatchesHost(c: MimoMoeCase) !void {
    setDecodeParams(c.dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var f = try mimoMoeFixture(alloc, c);
    defer f.deinit();
    const pre = try mimoPrefillArm(s, &f, c.topk);
    defer _ = mlx.mlx_array_free(pre);
    var cp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cp);
    const got = try evalF16(s, pre, &cp);
    const want = try alloc.alloc(f16, c.rows * c.hidden);
    for (0..c.rows) |r| {
        const y = try moeSwigluHost(
            alloc,
            f.xf[r * c.hidden ..][0..c.hidden],
            f.gate_t,
            f.suh_h,
            f.svh_i,
            f.up_t,
            f.suh_h,
            f.svh_i,
            f.down_t,
            f.suh_i,
            f.svh_h,
            f.slots[r * c.topk ..][0..c.topk],
            f.scores[r * c.topk ..][0..c.topk],
            c.hidden,
            c.inter,
            c.rate.halfwords(),
            c.hidden / 16,
            c.inter / 16,
            c.dec,
        );
        for (y, 0..) |v, i| want[r * c.hidden + i] = @floatCast(v);
    }
    try expectRelRms(got[0 .. c.rows * c.hidden], want, 0.02);
}

/// The two arms on the same rows. The decode chain is what MiMo answers
/// correctly live, so it is the reference at widths the host oracle cannot reach.
fn mimoArmsAgree(c: MimoMoeCase) !void {
    setDecodeParams(c.dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var f = try mimoMoeFixture(arena.allocator(), c);
    defer f.deinit();
    const pre = try mimoPrefillArm(s, &f, c.topk);
    defer _ = mlx.mlx_array_free(pre);
    const dec = try mimoDecodeArm(s, &f);
    defer _ = mlx.mlx_array_free(dec);
    var cp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cp);
    var cd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cd);
    const ap = try evalF16(s, pre, &cp);
    const ad = try evalF16(s, dec, &cd);
    try expectRelRms(ap[0 .. c.rows * c.hidden], ad[0 .. c.rows * c.hidden], 0.02);
}

/// The SwiGLU every EXL3 arm approximates, carried end to end in f32: the
/// stored weights are f16 but nothing between them is. `moeSwigluHost` mirrors
/// the kernels' own f16 stores, so it cannot say whether one of them saturated.
const Exl3F32Peaks = struct { g: f64 = 0, u: f64 = 0, mid: f64 = 0, down_inner: f64 = 0 };

fn exl3SwigluF32(
    alloc: std.mem.Allocator,
    x: []const f32,
    f: *const MimoMoeFixture,
    c: MimoMoeCase,
    slots: []const u32,
    scores: []const f32,
    peaks: *Exl3F32Peaks,
    out: []f32,
) !void {
    const n = c.rate.halfwords();
    const gu_tile = (c.hidden / 16) * (c.inter / 16) * n;
    const t_h = try alloc.alloc(f32, c.hidden);
    const t_i = try alloc.alloc(f32, c.inter);
    const gy = try alloc.alloc(f32, c.inter);
    const uy = try alloc.alloc(f32, c.inter);
    const dy = try alloc.alloc(f32, c.hidden);
    @memset(out, 0);
    const proj = struct {
        fn run(src: []const f32, tr: []const u16, suh: []const u16, svh: []const u16, in_f: usize, out_f: usize, rate: exl3.Rate, dec: exl3.Decode, scratch: []f32, dst: []f32) void {
            for (src, suh, scratch) |v, sb, *d| d.* = v * exl3.f16BitsToF32(sb);
            var b: usize = 0;
            while (b < in_f) : (b += exl3.HAD_DIM) {
                var vec: [exl3.HAD_DIM]f32 = scratch[b..][0..exl3.HAD_DIM].*;
                exl3.hadamard128(&vec);
                @memcpy(scratch[b..][0..exl3.HAD_DIM], &vec);
            }
            @memset(dst, 0);
            var tile: [exl3.TILE_VALUES]u16 = undefined;
            const ot = out_f / 16;
            for (0..in_f / 16) |tk| {
                for (0..ot) |tn| {
                    exl3.decodeTile(tr[(tk * ot + tn) * rate.halfwords() ..][0..rate.halfwords()], rate, dec, &tile);
                    for (0..16) |r| {
                        const xv = scratch[tk * 16 + r];
                        for (0..16) |cc| dst[tn * 16 + cc] += xv * exl3.f16BitsToF32(tile[r * 16 + cc]);
                    }
                }
            }
            var ob: usize = 0;
            while (ob < out_f) : (ob += exl3.HAD_DIM) {
                var vec: [exl3.HAD_DIM]f32 = dst[ob..][0..exl3.HAD_DIM].*;
                exl3.hadamard128(&vec);
                @memcpy(dst[ob..][0..exl3.HAD_DIM], &vec);
            }
            for (dst, svh) |*v, sb| v.* *= exl3.f16BitsToF32(sb);
        }
    }.run;
    for (slots, scores) |e, w| {
        const go = e * gu_tile;
        proj(x, f.gate_t[go..][0..gu_tile], f.suh_h[e * c.hidden ..][0..c.hidden], f.svh_i[e * c.inter ..][0..c.inter], c.hidden, c.inter, c.rate, c.dec, t_h, gy);
        proj(x, f.up_t[go..][0..gu_tile], f.suh_h[e * c.hidden ..][0..c.hidden], f.svh_i[e * c.inter ..][0..c.inter], c.hidden, c.inter, c.rate, c.dec, t_h, uy);
        for (gy, uy) |*g, u| {
            peaks.g = @max(peaks.g, @abs(@as(f64, g.*)));
            peaks.u = @max(peaks.u, @abs(@as(f64, u)));
            g.* = (g.* / (1.0 + @exp(-g.*))) * u;
            peaks.mid = @max(peaks.mid, @abs(@as(f64, g.*)));
        }
        proj(gy, f.down_t[e * gu_tile ..][0..gu_tile], f.suh_i[e * c.inter ..][0..c.inter], f.svh_h[e * c.hidden ..][0..c.hidden], c.inter, c.hidden, c.rate, c.dec, t_i, dy);
        for (dy) |v| peaks.down_inner = @max(peaks.down_inner, @abs(@as(f64, v)));
        for (out, dy) |*o, v| o.* += w * v;
    }
}

/// Both arms against the f32 SwiGLU, at whatever magnitude the case carries.
fn mimoArmMatchesF32(c: MimoMoeCase) !void {
    setDecodeParams(c.dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var f = try mimoMoeFixture(alloc, c);
    defer f.deinit();
    const y = if (usesPrefillArm(c.rows))
        try mimoPrefillArm(s, &f, c.topk)
    else
        try mimoDecodeArm(s, &f);
    defer _ = mlx.mlx_array_free(y);
    var cy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cy);
    const got = try evalF16(s, y, &cy);
    const want = try alloc.alloc(f32, c.hidden);
    var peaks: Exl3F32Peaks = .{};
    var ss: f64 = 0;
    var ref: f64 = 0;
    for (0..c.rows) |r| {
        try exl3SwigluF32(alloc, f.xf[r * c.hidden ..][0..c.hidden], &f, c, f.slots[r * c.topk ..][0..c.topk], f.scores[r * c.topk ..][0..c.topk], &peaks, want);
        for (want, 0..) |w, i| {
            const a: f64 = @floatCast(got[r * c.hidden + i]);
            if (!std.math.isFinite(a)) {
                std.debug.print("exl3 arm went non-finite at row {d}: |silu(g)*u| peaks at {d:.0}\n", .{ r, peaks.mid });
                return error.TestExpectedEqual;
            }
            ss += (a - w) * (a - w);
            ref += @as(f64, w) * @as(f64, w);
        }
    }
    const rel = @sqrt(ss / @max(ref, 1e-20));
    if (rel < 0.01) return;
    std.debug.print("exl3 vs f32 SwiGLU rel_rms {d:.6}; peaks g={d:.0} u={d:.0} mid={d:.0}\n", .{ rel, peaks.g, peaks.u, peaks.mid });
    return error.TestExpectedEqual;
}

// The pack's own scale magnitudes with a residual the size the served model
// carries put `silu(gate) * up` past 65504 while every input, weight and
// output stays ordinary: an f16 plane there turns a whole routed row into inf.
// Synthetic-magnitude parity cases cannot see it — they never leave f16 range.
// Every other parity case drives the arms with synthetic trellis and a single
// scale magnitude; a real shard's suh spans 0.0006..0.17 inside one vector.
// `REAL_BLOB` names a pack slice (E experts of gate/up/down trellis, then
// suh_h/svh_i/suh_i/svh_h) so the Metal arms are scored on the bytes a pack
// actually ships. Absent, there is nothing to read and the case skips.
test "mimo_v2 EXL3 arms match the f32 SwiGLU on a real pack's own bytes" {
    const blob = std.c.getenv("REAL_BLOB") orelse return error.SkipZigTest;
    for ([_]usize{ 1, 33 }) |rows| {
        for ([_]f32{ 0.3, 1.0, 3.0 }) |xs| {
            try mimoArmMatchesF32(.{
                .e = 8,
                .hidden = 4096,
                .inter = 2048,
                .topk = 8,
                .rows = rows,
                .rate = .{ .n = 40 },
                .dec = .{ .codebook = .tiny, .window = .w12 },
                .seed = 4242,
                .x_scale = xs,
                .real_blob = blob,
            });
        }
    }
}

test "mimo_v2 EXL3 arms stay finite where the SwiGLU product passes the f16 ceiling" {
    for ([_]usize{ 4, 33 }) |rows| {
        try mimoArmMatchesF32(.{
            .e = 8,
            .hidden = 512,
            .inter = 256,
            .topk = 4,
            .rows = rows,
            .rate = .{ .n = 40 },
            .dec = .{ .codebook = .tiny, .window = .w12 },
            .seed = 9001 + rows,
            .banks = MIMO_BANKS,
            .x_scale = 512,
        });
    }
}

test "mimo_v2 EXL3 prefill rows match the host SwiGLU oracle past one GEMM window" {
    for ([_]usize{ 33, 40, 64, 128 }) |rows| {
        try mimoPrefillMatchesHost(.{
            .e = 64,
            .hidden = 256,
            .inter = 128,
            .topk = 8,
            .rows = rows,
            .rate = .{ .n = 40 },
            .dec = .{ .codebook = .tiny, .window = .w12 },
            .seed = 1301 + rows,
        });
    }
}

test "mimo_v2 EXL3 prefill rows match the fused decode arm at E=256 top-8" {
    for ([_]usize{ 33, 64, 128, 512 }) |rows| {
        try mimoArmsAgree(.{
            .e = 256,
            .hidden = 256,
            .inter = 128,
            .topk = 8,
            .rows = rows,
            .rate = .{ .n = 40 },
            .dec = .{ .codebook = .tiny, .window = .w12 },
            .seed = 1401 + rows,
        });
    }
}

test "mimo_v2 EXL3 prefill rows match the fused decode arm at the served hidden and inter" {
    for ([_]usize{ 33, 64 }) |rows| {
        try mimoArmsAgree(.{
            .e = 16,
            .hidden = 4096,
            .inter = 2048,
            .topk = 8,
            .rows = rows,
            .rate = .{ .n = 40 },
            .dec = .{ .codebook = .tiny, .window = .w12 },
            .seed = 1501 + rows,
        });
    }
}

/// The window geometry the production code resolves, through the levers it
/// reads: the cached answer is dropped so the env is what decides.
fn withGemmWindow(win: ?[*:0]const u8, aligned: bool, c: MimoMoeCase) !void {
    const prev_win = gemm_win_cached;
    const prev_align = gemm_align_cached;
    defer {
        gemm_win_cached = prev_win;
        gemm_align_cached = prev_align;
        _ = unsetenv("MLX_SERVE_EXL3_GEMM_WIN");
        _ = unsetenv("MLX_SERVE_EXL3_WIN_ALIGN");
    }
    gemm_win_cached = null;
    gemm_align_cached = null;
    if (win) |w| _ = setenv("MLX_SERVE_EXL3_GEMM_WIN", w, 1) else _ = unsetenv("MLX_SERVE_EXL3_GEMM_WIN");
    _ = setenv("MLX_SERVE_EXL3_WIN_ALIGN", if (aligned) "1" else "0", 1);
    try mimoArmsAgree(c);
}

test "mimo_v2 EXL3 prefill rows match the fused decode arm on every GEMM arm" {
    const base = MimoMoeCase{
        .e = 256,
        .hidden = 256,
        .inter = 128,
        .topk = 8,
        .rows = 55,
        .rate = .{ .n = 40 },
        .dec = .{ .codebook = .tiny, .window = .w12 },
        .seed = 1601,
    };
    for ([_]bool{ true, false }) |nax_off| {
        if (nax_off) {
            if (!gemmNaxOn()) continue;
            _ = setenv("MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK", "1", 1);
        }
        defer if (nax_off) {
            _ = unsetenv("MLX_SERVE_FORCE_GPU_FAMILY_FALLBACK");
        };
        for ([_]?[*:0]const u8{ null, "16" }) |win| {
            for ([_]bool{ true, false }) |aligned| try withGemmWindow(win, aligned, base);
        }
    }
}

// Every MoE layer of a chunk is built before the chunk's ONE evaluation, and
// each layer's routing gives its GEMM a different window count over the one
// cached kernel config, while a tail-bumped pack gives them different RATES:
// the built dispatches must not read each other's.
test "mimo_v2 EXL3 prefill layers built lazily keep their own window count and rate" {
    const c0 = MimoMoeCase{
        .e = 256,
        .hidden = 256,
        .inter = 128,
        .topk = 8,
        .rows = 55,
        .rate = .{ .n = 40 },
        .dec = .{ .codebook = .tiny, .window = .w12 },
        .seed = 0,
    };
    setDecodeParams(c0.dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const L = 48;
    var fs: [L]MimoMoeFixture = undefined;
    var lazy: [L]mlx.mlx_array = undefined;
    for (0..L) |i| {
        var c = c0;
        c.seed = 7000 + i;
        // The tail layers a bumped pack packs wider, built into the same graph.
        if (i + 2 >= L) c.rate = .{ .n = 64 };
        fs[i] = try mimoMoeFixture(alloc, c);
        lazy[i] = try mimoPrefillArm(s, &fs[i], c0.topk);
    }
    defer for (0..L) |i| {
        _ = mlx.mlx_array_free(lazy[i]);
        fs[i].deinit();
    };
    const vec = mlx.mlx_vector_array_new_data(&lazy, L);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
    for (0..L) |i| {
        const dec = try mimoDecodeArm(s, &fs[i]);
        defer _ = mlx.mlx_array_free(dec);
        var cp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cp);
        var cd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cd);
        const ap = try evalF16(s, lazy[i], &cp);
        const ad = try evalF16(s, dec, &cd);
        try expectRelRms(ap[0 .. c0.rows * c0.hidden], ad[0 .. c0.rows * c0.hidden], 0.02);
    }
}

test "exl3 a window wider than the kernel row capacity refuses" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const n: c_int = 40;
    const dim: c_int = 128;
    const E: c_int = 2;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const xh = try alloc.alloc(u16, @intCast(n * dim));
    @memset(xh, 0);
    const tr = try alloc.alloc(u16, @intCast(E * 8 * 8 * 64));
    @memset(tr, 0);
    var eids: [40]u32 = @splat(0);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ n, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(tr.ptr, &[_]c_int{ E, 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{n}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    // 32 rows is what both kernel bodies accumulate; past it their own
    // `n > WIN` guard still admits the window and the extra rows go unwritten.
    try t.expectError(error.BadExl3Shape, innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 33));
    try t.expectError(error.BadExl3Shape, innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 64));
    const ok = try innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 32);
    defer _ = mlx.mlx_array_free(ok);
    try t.expectEqual(@as(c_int, n), mlx.getShape(ok)[0]);
}

test "exl3 the GEMM window selector answers the window it was asked for" {
    const t = std.testing;
    try t.expectEqual(@as(?c_int, 4), resolveGemmWindowRows("4"));
    try t.expectEqual(@as(?c_int, 8), resolveGemmWindowRows("8"));
    try t.expectEqual(@as(?c_int, 16), resolveGemmWindowRows("16"));
    try t.expectEqual(@as(?c_int, 32), resolveGemmWindowRows("32"));
    try t.expectEqual(@as(?c_int, null), resolveGemmWindowRows(null));
    try t.expectEqual(@as(?c_int, null), resolveGemmWindowRows(""));
    try t.expectEqual(@as(?c_int, null), resolveGemmWindowRows("0"));
    try t.expectEqual(@as(?c_int, null), resolveGemmWindowRows("64"));
    try t.expectEqual(@as(?c_int, null), resolveGemmWindowRows("16x"));
}

test "exl3 prepareIndexed refuses a row count that is not the slot count" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const dim: c_int = 128;
    const topk: c_int = 4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const xh = try alloc.alloc(u16, @intCast(dim));
    @memset(xh, 0);
    const suh = try alloc.alloc(u16, @intCast(2 * dim));
    @memset(suh, 0);
    var slots: [4]u32 = @splat(0);
    // One row spelled [1, dim] is the natural shape for a single activation and
    // the kernel reads `topk` rows of it.
    const x1 = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ 1, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(x1);
    const suh_a = mlx.mlx_array_new_data(suh.ptr, &[_]c_int{ 2, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh_a);
    const sl = mlx.mlx_array_new_data(&slots, &[_]c_int{topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(sl);
    try t.expectError(error.BadExl3Shape, prepareIndexed(s, x1, suh_a, sl));
    const x0 = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{dim}, 1, .float16);
    defer _ = mlx.mlx_array_free(x0);
    const ok = try prepareIndexed(s, x0, suh_a, sl);
    defer _ = mlx.mlx_array_free(ok);
    try t.expectEqual(topk, mlx.getShape(ok)[0]);
}

test "exl3 prefill output carries the activation dtype" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: c_int = 2;
    const dim: c_int = 128;
    const rows: c_int = 20;
    const topk: c_int = 1;
    const tr = try alloc.alloc(u16, @intCast(E * 8 * 8 * 64));
    var prng = std.Random.DefaultPrng.init(21);
    const rnd = prng.random();
    for (tr) |*v| v.* = @truncate(rnd.int(u32));
    const suh = try alloc.alloc(u16, @intCast(E * dim));
    for (suh) |*v| v.* = exl3.f32ToF16Bits(1.0);
    const svh = try alloc.alloc(u16, @intCast(E * dim));
    for (svh) |*v| v.* = exl3.f32ToF16Bits(1.0);
    // A bf16 activation is what the qwen4 trunk hands the routed experts; an
    // f16 result both double-rounds and saturates at 65504.
    const xb = try alloc.alloc(u16, @intCast(rows * dim));
    for (xb) |*v| v.* = @truncate(@as(u32, @bitCast(@as(f32, 0.5))) >> 16);
    const slots_h = try alloc.alloc(u32, @intCast(rows * topk));
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % @as(usize, @intCast(E)));
    const scores_h = try alloc.alloc(f32, @intCast(rows * topk));
    for (scores_h) |*v| v.* = 1e8;
    const x_arr = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ rows, dim }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_a = mlx.mlx_array_new_data(tr.ptr, &[_]c_int{ E, 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_a);
    const suh_a = mlx.mlx_array_new_data(suh.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh_a);
    const svh_a = mlx.mlx_array_new_data(svh.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh_a);
    const sl = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{rows * topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(sl);
    const sc = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{rows * topk}, 1, .float32);
    defer _ = mlx.mlx_array_free(sc);
    const out = try moePrefill(s, x_arr, tr_a, suh_a, svh_a, tr_a, suh_a, svh_a, tr_a, suh_a, svh_a, sl, sc, topk);
    defer _ = mlx.mlx_array_free(out);
    try t.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(out));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, out, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    var f32c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32c);
    try mlx.check(mlx.mlx_astype(&f32c, c, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32c));
    const p = mlx.mlx_array_data_float32(f32c) orelse return error.F16Unreadable;
    var finite: usize = 0;
    for (0..@intCast(rows * dim)) |j| {
        if (std.math.isFinite(p[j])) finite += 1;
    }
    try t.expectEqual(@as(usize, @intCast(rows * dim)), finite);
}

test "exl3 the decode reduce folds the scores in f32" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: c_int = 2;
    const out_dim: c_int = 128;
    const rows: c_int = 2;
    const topk: c_int = 2;
    const nslots: usize = @intCast(rows * topk);
    const od: usize = @intCast(out_dim);
    var prng = std.Random.DefaultPrng.init(37);
    const rnd = prng.random();
    const inner_h = try alloc.alloc(u16, nslots * od);
    for (inner_h) |*v| v.* = exl3.f32ToF16Bits((rnd.float(f32) * 2 - 1) * 10);
    // svh puts the per-slot partials past the f16 range while the score fold
    // brings the answer back: an f16 bank saturates here, an f32 one does not.
    const svh_h = try alloc.alloc(u16, @intCast(E * out_dim));
    for (svh_h) |*v| v.* = exl3.f32ToF16Bits(60000.0);
    const slots_h = try alloc.alloc(u32, nslots);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % @as(usize, @intCast(E)));
    const sc_h = try alloc.alloc(f32, nslots);
    for (sc_h) |*v| v.* = 1e-4;
    const inner = mlx.mlx_array_new_data(inner_h.ptr, &[_]c_int{ @intCast(nslots), out_dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(inner);
    const svh = mlx.mlx_array_new_data(svh_h.ptr, &[_]c_int{ E, out_dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(nslots)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(sc_h.ptr, &[_]c_int{@intCast(nslots)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const got = try downFinishReduce(s, inner, svh, slots, scores, out_dim, rows, topk, .float32);
    defer _ = mlx.mlx_array_free(got);
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, got, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const gp = mlx.mlx_array_data_float32(c) orelse return error.F16Unreadable;
    const nrows: usize = @intCast(rows);
    const host = try alloc.alloc(f32, nrows * od);
    @memset(host, 0);
    var vec: [128]f32 = undefined;
    const ntopk: usize = @intCast(topk);
    for (0..nrows) |r| {
        for (0..ntopk) |k| {
            const slot = r * ntopk + k;
            for (0..od) |j| vec[j] = exl3.f16BitsToF32(inner_h[slot * od + j]);
            exl3.hadamard128(&vec);
            const eid: usize = slots_h[slot];
            for (0..od) |j| host[r * od + j] += vec[j] * exl3.f16BitsToF32(svh_h[eid * od + j]) * sc_h[slot];
        }
    }
    var worst: f32 = 0;
    for (host, 0..) |h, j| {
        try t.expect(std.math.isFinite(gp[j]));
        const rel = @abs(gp[j] - h) / @max(@abs(h), 1e-6);
        if (rel > worst) worst = rel;
    }
    if (!(worst < 1e-4)) {
        std.debug.print("exl3 reduce worst rel {d:.8}\n", .{worst});
        return error.TestExpectedEqual;
    }
}

test "exl3 a diagnostic env switch set to nothing is off" {
    const t = std.testing;
    try t.expect(!diagEnvValueOn(null));
    try t.expect(!diagEnvValueOn("0"));
    try t.expect(!diagEnvValueOn(""));
    try t.expect(diagEnvValueOn("1"));
}

test "exl3 the host SwiGLU oracle decodes at the K its packed dim names" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const dim: usize = 128;
    const tiles: usize = dim / 16;
    const E: usize = 2;
    const k: u32 = 3;
    const packed_n = exl3.packedHalfwords(k);
    const stride = tiles * tiles * packed_n;
    // Sized for K4 so a K4 misread stays inside the buffer and shows as a value.
    const room = tiles * tiles * exl3.packedHalfwords(4);
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const banks = try alloc.alloc(u16, 3 * E * room);
    for (banks) |*v| v.* = @truncate(rnd.int(u32));
    const gate_t = banks[0 .. E * room];
    const up_t = banks[E * room .. 2 * E * room];
    const down_t = banks[2 * E * room ..];
    const scales = try alloc.alloc(u16, 6 * E * dim);
    for (scales) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.5 + 0.75);
    const gate_suh = scales[0 .. E * dim];
    const gate_svh = scales[E * dim .. 2 * E * dim];
    const up_suh = scales[2 * E * dim .. 3 * E * dim];
    const up_svh = scales[3 * E * dim .. 4 * E * dim];
    const down_suh = scales[4 * E * dim .. 5 * E * dim];
    const down_svh = scales[5 * E * dim ..];
    const x = try alloc.alloc(f32, dim);
    for (x) |*v| v.* = rnd.float(f32) * 2 - 1;
    const slots = [_]u32{ 1, 0 };
    const weights = [_]f32{ 0.625, 0.375 };
    const got = try moeSwigluHost(
        alloc,
        x,
        gate_t,
        gate_suh,
        gate_svh,
        up_t,
        up_suh,
        up_svh,
        down_t,
        down_suh,
        down_svh,
        &slots,
        &weights,
        dim,
        dim,
        packed_n,
        tiles,
        tiles,
        .mul1,
    );
    const want = try alloc.alloc(f32, dim);
    @memset(want, 0);
    const scratch_a = try alloc.alloc(f32, dim);
    const scratch_b = try alloc.alloc(f32, dim);
    const gate_y = try alloc.alloc(f32, dim);
    const up_y = try alloc.alloc(f32, dim);
    const h = try alloc.alloc(f32, dim);
    const down_y = try alloc.alloc(f32, dim);
    for (slots, weights) |e, w| {
        const off = e * stride;
        exl3.project(x, gate_t[off..][0..stride], gate_suh[e * dim ..][0..dim], gate_svh[e * dim ..][0..dim], dim, dim, exl3.Rate.fromK(k), .mul1, scratch_a, scratch_b, gate_y);
        exl3.project(x, up_t[off..][0..stride], up_suh[e * dim ..][0..dim], up_svh[e * dim ..][0..dim], dim, dim, exl3.Rate.fromK(k), .mul1, scratch_a, scratch_b, up_y);
        for (0..dim) |i| {
            const g = gate_y[i];
            h[i] = (g / (1.0 + @exp(-g))) * up_y[i];
        }
        exl3.project(h, down_t[off..][0..stride], down_suh[e * dim ..][0..dim], down_svh[e * dim ..][0..dim], dim, dim, exl3.Rate.fromK(k), .mul1, scratch_a, scratch_b, down_y);
        for (0..dim) |i| want[i] += w * down_y[i];
    }
    try t.expectEqualSlices(f32, want, got);
}

test "exl3 a novel row count does not compile another sorted GEMM" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const dim: usize = 256;
    const E: usize = 2;
    const tiles = dim / 16;
    const packed_n = exl3.packedHalfwords(4);
    const tile_n = tiles * tiles * packed_n;
    const stacked = try alloc.alloc(u16, E * tile_n);
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    for (stacked) |*v| v.* = @truncate(rnd.int(u32));
    const tr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), @intCast(tiles), @intCast(tiles), @intCast(packed_n) }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    // Row counts the prefill really sees vary per chunk; only the first shape
    // may pay a kernel compile.
    const counts = [_]usize{ 32, 64, 96 };
    var warm_ns: u64 = 0;
    for (counts, 0..) |n, ci| {
        const xh = try alloc.alloc(u16, n * dim);
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.5);
        const eids = try alloc.alloc(u32, n);
        for (eids, 0..) |*v, i| v.* = @intCast((i / 16) % E);
        const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
        defer _ = mlx.mlx_array_free(x_arr);
        const eid_a = mlx.mlx_array_new_data(eids.ptr, &[_]c_int{@intCast(n)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(eid_a);
        var first_ns: u64 = 0;
        for (0..3) |rep| {
            var sw = io_util.Stopwatch.init(t.io);
            const y = try innerGemmSorted(s, x_arr, tr, eid_a);
            try mlx.check(mlx.mlx_array_eval(y));
            const dt = sw.read();
            _ = mlx.mlx_array_free(y);
            if (rep == 0) first_ns = dt else warm_ns = @max(warm_ns, dt);
        }
        if (ci == 0) continue;
        benchPrint("[exl3] novel n={d} first={d} us warm={d} us\n", .{ n, first_ns / 1000, warm_ns / 1000 });
        try t.expect(warm_ns > 0);
        try t.expect(first_ns < warm_ns * 10);
    }
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

// Codebook A/B at production expert shape: ITER forwards per arm built lazily
// and timed as ONE eval, arms alternated over rounds, medians reported.
// Prints only under MLX_SERVE_EXL3_CODEBOOK_AB (a diagnostic, never a test).
test "exl3 codebook A/B at production shape" {
    if (!diagEnvValueOn(std.c.getenv("MLX_SERVE_EXL3_CODEBOOK_AB"))) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 16;
    const topk: usize = 10;
    const in_dim: usize = 2560;
    const out_dim: usize = 640;
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const g_n = in_tiles * out_tiles * 64;
    var prng = std.Random.DefaultPrng.init(73);
    const rnd = prng.random();
    const tr_g = try alloc.alloc(u16, E * g_n);
    const tr_d = try alloc.alloc(u16, E * g_n);
    for (tr_g) |*v| v.* = @truncate(rnd.int(u32));
    for (tr_d) |*v| v.* = @truncate(rnd.int(u32));
    const ones_in = try alloc.alloc(u16, E * in_dim);
    const ones_out = try alloc.alloc(u16, E * out_dim);
    @memset(ones_in, exl3.f32ToF16Bits(1.0));
    @memset(ones_out, exl3.f32ToF16Bits(1.0));
    const trg = mlx.mlx_array_new_data(tr_g.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trg);
    const trd = mlx.mlx_array_new_data(tr_d.ptr, &[_]c_int{ @intCast(E), @intCast(out_tiles), @intCast(in_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trd);
    const su_g = mlx.mlx_array_new_data(ones_in.ptr, &[_]c_int{ @intCast(E), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(su_g);
    const sv_g = mlx.mlx_array_new_data(ones_out.ptr, &[_]c_int{ @intCast(E), @intCast(out_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sv_g);
    const su_d = mlx.mlx_array_new_data(ones_out.ptr, &[_]c_int{ @intCast(E), @intCast(out_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(su_d);
    const sv_d = mlx.mlx_array_new_data(ones_in.ptr, &[_]c_int{ @intCast(E), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sv_d);
    const io = std.Io.Threaded.global_single_threaded.io();
    const arms = [_]exl3.Decode{ .mul1, .tiny };
    const rounds: usize = 7;
    std.debug.print("{s:>5} {s:>10} {s:>10} {s:>9}\n", .{ "rows", "mul1 ms", "tiny ms", "tiny/mul1" });
    for ([_]usize{ 1, 4, 16, 512 }) |R| {
        const iters: usize = if (R >= 64) 3 else 10;
        const xh = try alloc.alloc(u16, R * in_dim);
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
        const sl = try alloc.alloc(u32, R * topk);
        const sc = try alloc.alloc(f32, R * topk);
        for (sl, 0..) |*v, i| v.* = @intCast(i % E);
        @memset(sc, 0.1);
        const x = if (R == 1)
            mlx.mlx_array_new_data(xh.ptr, &[_]c_int{@intCast(in_dim)}, 1, .float16)
        else
            mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(R), @intCast(in_dim) }, 2, .float16);
        defer _ = mlx.mlx_array_free(x);
        const slots = mlx.mlx_array_new_data(sl.ptr, &[_]c_int{@intCast(R * topk)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(slots);
        const scores = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{@intCast(R * topk)}, 1, .float32);
        defer _ = mlx.mlx_array_free(scores);
        var med: [2][rounds]f64 = undefined;
        for (0..rounds) |round| {
            for (arms, 0..) |dec, ai| {
                setDecodeParams(dec);
                const outs = try alloc.alloc(mlx.mlx_array, iters + 1);
                for (outs) |*o| {
                    o.* = if (R > DECODE_ROWS_MAX)
                        try moePrefill(s, x, trg, su_g, sv_g, trg, su_g, sv_g, trd, su_d, sv_d, slots, scores, @intCast(topk))
                    else
                        try moeSwigluFused(s, x, trg, su_g, sv_g, trg, su_g, sv_g, trd, su_d, sv_d, slots, scores, .float16);
                }
                try mlx.check(mlx.mlx_array_eval(outs[0]));
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                for (outs[1..]) |o| _ = mlx.mlx_vector_array_append_value(vec, o);
                var sw = io_util.Stopwatch.init(io);
                try mlx.check(mlx.mlx_eval(vec));
                const ns = sw.read();
                med[ai][round] = @as(f64, @floatFromInt(ns)) / 1e6 / @as(f64, @floatFromInt(iters));
                for (outs) |o| _ = mlx.mlx_array_free(o);
            }
        }
        for (&med) |*m| std.mem.sort(f64, m, {}, std.sort.asc(f64));
        const a = med[0][rounds / 2];
        const b = med[1][rounds / 2];
        std.debug.print("{d:5} {d:10.3} {d:10.3} {d:9.3}\n", .{ R, a, b, b / a });
    }
}

