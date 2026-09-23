// Cooperative 16x32x16 matmul2d fragments for sushi_attn_pd_nax. The fragment layout and the
// MMA wrapper follow MLX's steel/attn/nax.h (Copyright © 2025 Apple Inc., MIT).
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#define SUSHI_UNROLL _Pragma("clang loop unroll(full)")
struct SushiNax {
  // This lane's element (row y + (j/4)*8, col x + j%4) of a 16x16 fragment, j < 8.
  static short2 coord() {
    ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    short qid = lane >> 2;
    return short2(((qid & 2) | (lane & 1)) * 4, (qid & 4) | ((lane >> 1) & 3));
  }
  // C[16x32] (two 16x16 fragments) += A[16x16] * B[16x32] (B given as two 16x16 fragments).
  template <typename CT, typename AT, typename BT, bool TA, bool TB>
  static void mma(thread metal::vec<CT, 8>& c0, thread metal::vec<CT, 8>& c1,
                  thread const metal::vec<AT, 8>& a,
                  thread const metal::vec<BT, 8>& b0, thread const metal::vec<BT, 8>& b1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, TA, TB, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
    auto cb = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
    auto cc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>,
                                                            metal::remove_addrspace_t<decltype(cb)>, CT>();
    SUSHI_UNROLL for (short i = 0; i < 8; i++) ca[i] = a[i];
    SUSHI_UNROLL for (short i = 0; i < 8; i++) { cb[i] = b0[i]; cb[8 + i] = b1[i]; }
    SUSHI_UNROLL for (short i = 0; i < 8; i++) { cc[i] = c0[i]; cc[8 + i] = c1[i]; }
    op.run(ca, cb, cc);
    SUSHI_UNROLL for (short i = 0; i < 8; i++) { c0[i] = cc[i]; c1[i] = cc[8 + i]; }
  }
  // A 16x16 fragment from row-major memory; rows at or past `rows` read zero.
  template <typename T, typename P>
  static void load_rows(thread metal::vec<T, 8>& d, P src, int ld, int rows) {
    short2 c = coord();
    SUSHI_UNROLL for (short i = 0; i < 2; i++) {
      bool ok = (c.y + i * 8) < rows;
      SUSHI_UNROLL for (short j = 0; j < 4; j++) d[i * 4 + j] = ok ? T(src[(c.y + i * 8) * ld + c.x + j]) : T(0);
    }
  }
  template <typename T, typename P>
  static void load(thread metal::vec<T, 8>& d, P src, int ld) {
    short2 c = coord();
    SUSHI_UNROLL for (short i = 0; i < 2; i++) SUSHI_UNROLL for (short j = 0; j < 4; j++)
      d[i * 4 + j] = T(src[(c.y + i * 8) * ld + c.x + j]);
  }
};
