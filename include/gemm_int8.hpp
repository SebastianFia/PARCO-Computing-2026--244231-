// #pragma once

// #include <iostream>
// #include <cstring>
// #include <algorithm>
// #include <cassert>
// #include <omp.h>
// #include <immintrin.h>

// #include "matrix.hpp"
// #include "utils.hpp"

// #pragma once

// #include <iostream>
// #include <cstring>
// #include <algorithm>
// #include <cassert>
// #include <omp.h>
// #include <immintrin.h>

// #include "matrix.hpp"
// #include "utils.hpp"

// /* Int8 tiling sizes for AVX-512 (VNNI).
   
//    We use _mm512_dpbusd_epi32 which consumes 4 bytes of K per instruction.
// */
// constexpr int M_CACHE_TILE_INT8 = 512;
// constexpr int N_CACHE_TILE_INT8 = 1024;
// constexpr int K_CACHE_TILE_INT8 = 1024; 
// constexpr int M_REGISTER_TILE_INT8 = 16; // 16 accumulators (ZMM)
// constexpr int N_REGISTER_TILE_INT8 = 16; // 16 int32s fit in one ZMM
// constexpr int K_PACK_SIZE_INT8 = 4;      // The instruction consumes 4 bytes at once
// constexpr int K_UNROLL_INT8 = 2;         // How many groups of 4-bytes to unroll
// constexpr int PARALLEL_M_THRESHOLD_INT8 = 64; 

// /* Micro kernel for int8 GEMM (AVX-512 VNNI) with Compensation.
   
//    Math: 
//      We want: C += A_s8 * B_s8
//      HW does: dpbusd(A_u8, B_s8) = (A_s8 + 128) * B_s8
//                                  = A_s8 * B_s8 + 128 * B_s8
     
//      Correction: C += dpbusd(...) - (128 * sum(B_s8))
// */
// inline void microkernel_16x16_avx512_int8(
//     const int8_t* __restrict__ A, 
//     const int8_t* __restrict__ B, 
//     int32_t* __restrict__ C, 
//     const int32_t* __restrict__ comp_B, // Precomputed sum of B columns
//     int C_stride, int n_remain, int k_pad,
//     bool first_k_block
// ) {
//     constexpr int MR = M_REGISTER_TILE_INT8;
//     constexpr int NR = N_REGISTER_TILE_INT8;

//     // MR=16 output accumulators (ZMM registers)
//     __m512i C_acc[MR];

//     // 1. Initialize Accumulators
//     if (first_k_block) {
//         #pragma unroll(MR)
//         for (int i = 0; i < MR; ++i) {
//             C_acc[i] = _mm512_setzero_si512();
//         }
//     } else {
//         __mmask16 load_mask = (1U << n_remain) - 1;
//         #pragma unroll(MR)
//         for (int i = 0; i < MR; ++i) {
//             C_acc[i] = _mm512_mask_loadu_epi32(C_acc[i], load_mask, &C[i * C_stride]);
//         }
//     }

//     // 2. Compute Dot Product (A_u8 * B_s8)
//     for (int k = 0; k < k_pad; k += K_PACK_SIZE_INT8 * K_UNROLL_INT8) {
        
//         #pragma unroll(K_UNROLL_INT8)
//         for (int u = 0; u < K_UNROLL_INT8; ++u) {
//             // Load B: 16 columns x 4 rows (64 bytes)
//             __m512i v_b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(B));
//             B += 64; 

//             #pragma unroll(MR)
//             for (int i = 0; i < MR; ++i) {
//                 // Load A: Broadcast 4 bytes (packed as u8)
//                 // A is already transformed (xor 128) in packA
//                 int32_t a_val_scalar = *reinterpret_cast<const int32_t*>(&A[i * 4]);
//                 __m512i v_a = _mm512_set1_epi32(a_val_scalar);

//                 // Accumulate: (A_s8 + 128) * B_s8
//                 C_acc[i] = _mm512_dpbusd_epi32(C_acc[i], v_a, v_b);
//             }
//             A += MR * 4;
//         }
//     }

//     // 3. Apply Compensation: Subtract (128 * sum(B))
//     // We load the precomputed sums for these 16 columns
//     __m512i v_comp = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(comp_B));
    
//     // Shift left by 7 is equivalent to multiplying by 128
//     v_comp = _mm512_slli_epi32(v_comp, 7);

//     #pragma unroll(MR)
//     for (int i = 0; i < MR; ++i) {
//         C_acc[i] = _mm512_sub_epi32(C_acc[i], v_comp);
//     }

//     // 4. Store
//     __mmask16 store_mask = (1U << n_remain) - 1;
//     #pragma unroll(MR)
//     for (int i = 0; i < MR; ++i) {
//         _mm512_mask_storeu_epi32(&C[i * C_stride], store_mask, C_acc[i]);
//     }
// }

// inline void microkernel_cleanup_avx512_int8(
//     const int8_t* __restrict__ A, 
//     const int8_t* __restrict__ B, 
//     int32_t* __restrict__ C, 
//     const int32_t* __restrict__ comp_B,
//     int C_stride, int m_remain, int n_remain, int k_pad,
//     bool first_k_block
// ) {
//     constexpr int MR = M_REGISTER_TILE_INT8;
    
//     __m512i C_acc[MR];
//     __mmask16 mask = (1U << n_remain) - 1;

//     if (first_k_block) {
//         for (int i = 0; i < m_remain; ++i) {
//             C_acc[i] = _mm512_setzero_si512();
//         }
//     } else {
//         for (int i = 0; i < m_remain; ++i) {
//             C_acc[i] = _mm512_mask_loadu_epi32(C_acc[i], mask, &C[i * C_stride]);
//         }
//     }

//     for (int k = 0; k < k_pad; k += 4) {
//         __m512i v_b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(B));
//         B += 64; 

//         for (int i = 0; i < m_remain; ++i) {
//             int32_t a_val_scalar = *reinterpret_cast<const int32_t*>(&A[i * 4]);
//             __m512i v_a = _mm512_set1_epi32(a_val_scalar);
//             C_acc[i] = _mm512_dpbusd_epi32(C_acc[i], v_a, v_b);
//         }
//         A += MR * 4; 
//     }

//     // Compensation
//     __m512i v_comp = _mm512_mask_loadu_epi32(_mm512_setzero_si512(), mask, comp_B);
//     v_comp = _mm512_slli_epi32(v_comp, 7);

//     for (int i = 0; i < m_remain; ++i) {
//         C_acc[i] = _mm512_sub_epi32(C_acc[i], v_comp);
//     }

//     for (int i = 0; i < m_remain; ++i) {
//         _mm512_mask_storeu_epi32(&C[i * C_stride], mask, C_acc[i]);
//     }
// }

// /* Pack A for VNNI with On-the-fly Transformation.
   
//    Logic:
//    We read signed int8 A.
//    We transform it to "offset unsigned" by flipping the MSB (XOR 0x80).
//    This effectively adds 128 to the value.
   
//    Packed Layout: Row-major chunks of K=4.
//    Memory: [Block K=0..3][Row 0..MR]
// */
// inline void packA_avx512_int8(
//     const int8_t *__restrict__ A_tile,
//     int8_t *__restrict__ A_packed,
//     int A_stride,
//     int micropanel_stride,
//     int m_end, int k_end, int k_pad
// ) {
//     constexpr int MR = M_REGISTER_TILE_INT8;
//     int i_register = 0;

//     // Constant for sign flipping (128 repeated 4 times)
//     const int32_t xor_mask = 0x80808080;

//     for (; (i_register + 1) * MR <= m_end; ++i_register) {
//         int8_t* dst_panel = &A_packed[i_register * micropanel_stride];
        
//         int k = 0;
//         for (; k < k_end; k += 4) {
//             for (int i = 0; i < MR; ++i) {
//                 const int8_t* src = &A_tile[(i_register * MR + i) * A_stride + k];
//                 int32_t val;

//                 if (k + 4 <= k_end) {
//                     val = *reinterpret_cast<const int32_t*>(src);
//                 } else {
//                     // Handle edge K
//                     int8_t tmp[4] = {0, 0, 0, 0};
//                     for (int kk = 0; kk < 4; ++kk) {
//                         tmp[kk] = (k + kk < k_end) ? src[kk] : 0;
//                     }
//                     val = *reinterpret_cast<int32_t*>(tmp);
//                 }

//                 // Transform: Flip MSB to convert s8 to u8 range (add 128)
//                 val ^= xor_mask;
                
//                 *reinterpret_cast<int32_t*>(dst_panel) = val;
//                 dst_panel += 4;
//             }
//         }
        
//         // Zero pad K. 
//         // NOTE: In the transformed domain, "Zero" corresponds to -128.
//         // If we want actual zero-padding (value 0), we must store 0 + 128 = 0x80.
//         // Since B is padded with 0, A*B = 0 anyway, so A's pad value doesn't affect result.
//         // We stick to 0x80 (representing 0) for consistency.
//         for (; k < k_pad; k += 4) {
//             for (int i = 0; i < MR; ++i) {
//                 *reinterpret_cast<int32_t*>(dst_panel) = xor_mask; // Padding with "0"
//                 dst_panel += 4;
//             }
//         }
//     }

//     // Cleanup for m_remain
//     int m_remain = m_end - i_register * MR;
//     if (m_remain > 0) {
//         int8_t* dst_panel = &A_packed[i_register * micropanel_stride];
//         int k = 0;
//         for (; k < k_pad; k += 4) {
//             for (int i = 0; i < m_remain; ++i) {
//                 const int8_t* src = &A_tile[(i_register * MR + i) * A_stride + k];
//                 int8_t tmp[4] = {0, 0, 0, 0};
//                 for (int kk = 0; kk < 4; ++kk) {
//                    if (k + kk < k_end) tmp[kk] = src[kk];
//                 }
//                 int32_t val = *reinterpret_cast<int32_t*>(tmp);
//                 val ^= xor_mask;
//                 *reinterpret_cast<int32_t*>(dst_panel) = val;
//                 dst_panel += 4;
//             }
//             for (int i = m_remain; i < MR; ++i) {
//                 *reinterpret_cast<int32_t*>(dst_panel) = xor_mask;
//                 dst_panel += 4;
//             }
//         }
//     }
// }

// /* Pack B with Compensation Calculation.

//    1. Packs B into VNNI format (ZMM blocks of 16 cols x 4 rows).
//    2. Calculates sum of B elements for each column j, for the CURRENT k block.
//       Stores this in comp_B[j].
// */
// inline void packB_avx512_int8(
//     const int8_t *__restrict__ B_tile,
//     int8_t *__restrict__ B_packed,
//     int32_t *__restrict__ comp_B, // Output buffer for column sums
//     int B_stride,
//     int micropanel_stride,
//     int n_end, int k_end, int k_pad
// ) {
//     constexpr int NR = N_REGISTER_TILE_INT8;
//     int j_register = 0;

//     // Zero out compensation buffer first (for safety, though we could set during loop)
//     // We assume comp_B size is at least n_end rounded up to NR
//     memset(comp_B, 0, round_up(n_end, NR) * sizeof(int32_t));

//     for (; (j_register + 1) * NR <= n_end; ++j_register) {
//         int8_t* dst_panel = &B_packed[j_register * micropanel_stride];
//         int32_t* comp_ptr = &comp_B[j_register * NR];
        
//         int k = 0;
//         for (; k < k_end; k += 4) {
//             for (int j = 0; j < NR; ++j) {
//                 int32_t col_sum_accumulator = 0;
//                 for (int kk = 0; kk < 4; ++kk) {
//                     int8_t val = 0;
//                     if (k + kk < k_end) {
//                          val = B_tile[(k + kk) * B_stride + (j_register * NR + j)];
//                     }
//                     dst_panel[j * 4 + kk] = val;
//                     col_sum_accumulator += val;
//                 }
//                 // Accumulate into the compensation buffer for this column
//                 comp_ptr[j] += col_sum_accumulator;
//             }
//             dst_panel += 64; 
//         }
        
//         // Pad K (B padded with real 0s)
//         for (; k < k_pad; k += 4) {
//             memset(dst_panel, 0, 64);
//             dst_panel += 64;
//         }
//     }
    
//     // Cleanup n_remain
//     int n_remain = n_end - j_register * NR;
//     if (n_remain > 0) {
//         int8_t* dst_panel = &B_packed[j_register * micropanel_stride];
//         int32_t* comp_ptr = &comp_B[j_register * NR];

//         int k = 0;
//         for (; k < k_pad; k += 4) {
//             for (int j = 0; j < NR; ++j) {
//                 int32_t col_sum_accumulator = 0;
//                 for (int kk = 0; kk < 4; ++kk) {
//                     int8_t val = 0;
//                     if (j < n_remain && (k + kk < k_end)) {
//                         val = B_tile[(k + kk) * B_stride + (j_register * NR + j)];
//                     }
//                     dst_panel[j * 4 + kk] = val;
//                     col_sum_accumulator += val;
//                 }
//                 if (j < n_remain) comp_ptr[j] += col_sum_accumulator;
//             }
//             dst_panel += 64;
//         }
//     }
// }

// inline void gemm_int8_avx512_tiled_parallel_N(
//     const int8_t* __restrict__ A, 
//     const int8_t* __restrict__ B, 
//     int32_t* __restrict__ C, 
//     const int M, const int N, const int K
// ) {
//     const int num_threads = omp_get_max_threads();
//     const int MR = M_REGISTER_TILE_INT8;
//     const int NR = N_REGISTER_TILE_INT8;
    
//     const int KC = round_up(std::min(K_CACHE_TILE_INT8, K), K_PACK_SIZE_INT8); 
//     const int MC = std::min(M_CACHE_TILE_INT8, M); 
//     const int NC = round_up(clamp(N / num_threads, NR, N_CACHE_TILE_INT8), NR);

//     #pragma omp parallel
//     {
//         const int max_micropanel_stride_A = round_up(KC * MR, ALIGNMENT);
//         const int n_micropanels_A = ceil_division(MC, MR);
//         size_t bytes_A_packed = round_up(n_micropanels_A * max_micropanel_stride_A, ALIGNMENT);
//         auto* A_packed = reinterpret_cast<int8_t*>(aligned_alloc(ALIGNMENT, bytes_A_packed));

//         const int max_micropanel_stride_B = round_up(KC * NR, ALIGNMENT);
//         const int n_micropanels_B = ceil_division(NC, NR);
//         size_t bytes_B_packed = round_up(n_micropanels_B * max_micropanel_stride_B, ALIGNMENT);
//         auto* B_packed = reinterpret_cast<int8_t*>(aligned_alloc(ALIGNMENT, bytes_B_packed));

//         // Allocation for compensation buffer (One integer per column in the cache tile)
//         // Must be 64-byte aligned for vector loads
//         size_t bytes_comp_B = round_up(NC * sizeof(int32_t), ALIGNMENT);
//         auto* comp_B = reinterpret_cast<int32_t*>(aligned_alloc(ALIGNMENT, bytes_comp_B));

//         int j_cache_end = ceil_division(N, NC);

//         #pragma omp for schedule(static)
//         for (int j_cache = 0; j_cache < j_cache_end; ++j_cache) {
//             int n_end = std::min(NC, N - j_cache * NC);

//             for (int k_cache = 0; k_cache * KC < K; ++k_cache) {
//                 const int k_end = std::min(KC, K - k_cache * KC);
//                 const int k_pad = round_up(k_end, K_PACK_SIZE_INT8);

//                 const int micropanel_stride_A = round_up(k_pad * MR, ALIGNMENT);
//                 const int micropanel_stride_B = round_up(k_pad * NR, ALIGNMENT);

//                 // Pack B and calculate Compensation for this K-block
//                 packB_avx512_int8(
//                     &B[(k_cache * KC) * N + (j_cache * NC)], B_packed, comp_B,
//                     N, micropanel_stride_B, n_end, k_end, k_pad
//                 );

//                 int i_cache_end = ceil_division(M, MC);
//                 for (int i_cache = 0; i_cache < i_cache_end; ++i_cache) {
//                     int m_end = std::min(MC, M - i_cache * MC);

//                     packA_avx512_int8(
//                         &A[(i_cache * MC) * K + (k_cache * KC)], A_packed, 
//                         K, micropanel_stride_A, m_end, k_end, k_pad
//                     );
            
//                     int i_register = 0;
//                     for (; (i_register + 1) * MR <= m_end; ++i_register) {
//                         for (int j_register = 0; j_register * NR < n_end; ++j_register) {
//                             int n_remain = std::min(NR, n_end - j_register * NR);
//                             microkernel_16x16_avx512_int8(
//                                 &A_packed[i_register * micropanel_stride_A],
//                                 &B_packed[j_register * micropanel_stride_B],
//                                 &C[(i_cache * MC + i_register * MR) * N + (j_cache * NC + j_register * NR)],
//                                 &comp_B[j_register * NR], // Pass correct offset of compensation buffer
//                                 N, n_remain, k_pad,
//                                 (k_cache == 0)
//                             );
//                         }
//                     }
                    
//                     int m_remain = std::min(MR, m_end - i_register * MR);
//                     if (m_remain != 0) {
//                         for (int j_register = 0; j_register * NR < n_end; ++j_register) {
//                             int n_remain = std::min(NR, n_end - j_register * NR);
//                             microkernel_cleanup_avx512_int8(
//                                 &A_packed[i_register * micropanel_stride_A],
//                                 &B_packed[j_register * micropanel_stride_B],
//                                 &C[(i_cache * MC + i_register * MR) * N + (j_cache * NC + j_register * NR)],
//                                 &comp_B[j_register * NR],
//                                 N, m_remain, n_remain, k_pad,
//                                 (k_cache == 0)
//                             );
//                         }
//                     }
//                 }
//             }
//         }
//         std::free(A_packed);
//         std::free(B_packed);
//         std::free(comp_B);
//     }
// }

// inline void gemm_int8_avx512_tiled_parallel_M(
//     const int8_t* __restrict__ A, 
//     const int8_t* __restrict__ B, 
//     int32_t* __restrict__ C, 
//     const int M, const int N, const int K
// ) {
//     const int num_threads = omp_get_max_threads();
//     const int MR = M_REGISTER_TILE_INT8;
//     const int NR = N_REGISTER_TILE_INT8;

//     const int KC = round_up(std::min(K_CACHE_TILE_INT8, K), K_PACK_SIZE_INT8); 
//     const int MC = clamp(M / num_threads, 1, M_CACHE_TILE_INT8); 
//     const int NC = round_up(std::min(N, N_CACHE_TILE_INT8), NR); 

//     const int max_micropanel_stride_B = round_up(KC * NR, ALIGNMENT);
//     const int n_micropanels_B = ceil_division(NC, NR);
//     size_t bytes_B_packed = round_up(size_t(n_micropanels_B) * max_micropanel_stride_B, ALIGNMENT);
//     auto* B_packed = reinterpret_cast<int8_t*>(aligned_alloc(ALIGNMENT, bytes_B_packed));
    
//     // Alloc comp_B shared for parallel M
//     size_t bytes_comp_B = round_up(NC * sizeof(int32_t), ALIGNMENT);
//     auto* comp_B = reinterpret_cast<int32_t*>(aligned_alloc(ALIGNMENT, bytes_comp_B));

//     #pragma omp parallel
//     {
//         const int max_micropanel_stride_A = round_up(KC * MR, ALIGNMENT);
//         const int n_micropanels_A = ceil_division(MC, MR);
//         size_t bytes_A_packed = round_up(n_micropanels_A * max_micropanel_stride_A, ALIGNMENT);
//         auto* A_packed = reinterpret_cast<int8_t*>(aligned_alloc(ALIGNMENT, bytes_A_packed));

//         for (int j_cache = 0; j_cache * NC < N; ++j_cache) {
//             int n_end = std::min(NC, N - j_cache * NC);

//             for (int k_cache = 0; k_cache * KC < K; ++k_cache) {
//                 const int k_end = std::min(KC, K - k_cache * KC);
//                 const int k_pad = round_up(k_end, K_PACK_SIZE_INT8);

//                 const int micropanel_stride_A = round_up(k_pad * MR, ALIGNMENT);
//                 const int micropanel_stride_B = round_up(k_pad * NR, ALIGNMENT);

//                 #pragma omp single 
//                 packB_avx512_int8(
//                     &B[(k_cache * KC) * N + (j_cache * NC)], B_packed, comp_B,
//                     N, micropanel_stride_B, n_end, k_end, k_pad
//                 );

//                 int i_cache_end = ceil_division(M, MC);

//                 #pragma omp for schedule(static)
//                 for (int i_cache = 0; i_cache < i_cache_end; ++i_cache) {
//                     int m_end = std::min(MC, M - i_cache * MC);

//                     packA_avx512_int8(
//                         &A[(i_cache * MC) * K + (k_cache * KC)], A_packed, 
//                         K, micropanel_stride_A, m_end, k_end, k_pad
//                     );
            
//                     int i_register = 0;
//                     for (; (i_register + 1) * MR <= m_end; ++i_register) {
//                         for (int j_register = 0; j_register * NR < n_end; ++j_register) {
//                             int n_remain = std::min(NR, n_end - j_register * NR);
//                             microkernel_16x16_avx512_int8(
//                                 &A_packed[i_register * micropanel_stride_A],
//                                 &B_packed[j_register * micropanel_stride_B],
//                                 &C[(i_cache * MC + i_register * MR) * N + (j_cache * NC + j_register * NR)],
//                                 &comp_B[j_register * NR],
//                                 N, n_remain, k_pad,
//                                 (k_cache == 0)
//                             );
//                         }
//                     }
                    
//                     int m_remain = std::min(MR, m_end - i_register * MR);
//                     if (m_remain != 0) {
//                         for (int j_register = 0; j_register * NR < n_end; ++j_register) {
//                             int n_remain = std::min(NR, n_end - j_register * NR);
//                             microkernel_cleanup_avx512_int8(
//                                 &A_packed[i_register * micropanel_stride_A],
//                                 &B_packed[j_register * micropanel_stride_B],
//                                 &C[(i_cache * MC + i_register * MR) * N + (j_cache * NC + j_register * NR)],
//                                 &comp_B[j_register * NR],
//                                 N, m_remain, n_remain, k_pad,
//                                 (k_cache == 0)
//                             );
//                         }
//                     }
//                 }
//             }
//         }
//         std::free(A_packed);
//     }
//     std::free(B_packed);
//     std::free(comp_B);
// }

// inline void gemm_int8_tiled(const Matrix& A_base, const Matrix& B_base, Matrix& C_base) {
//     const auto& A = static_cast<const MatrixINT8&>(A_base);
//     const auto& B = static_cast<const MatrixINT8&>(B_base);
//     auto& C = static_cast<MatrixINT32&>(C_base);

//     const int M = A.rows;
//     const int K = A.cols;
//     const int N = B.cols;

//     const auto* A_ptr = reinterpret_cast<const int8_t*>(A.raw_data);
//     const auto* B_ptr = reinterpret_cast<const int8_t*>(B.raw_data);
//     auto* C_ptr = reinterpret_cast<int32_t*>(C.raw_data);

//     if (M > N) {
//         return gemm_int8_avx512_tiled_parallel_M(A_ptr, B_ptr, C_ptr, M, N, K);
//     } else {
//         return gemm_int8_avx512_tiled_parallel_N(A_ptr, B_ptr, C_ptr, M, N, K);
//     }
// }

#pragma once

#include <iostream>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <omp.h>
#include <immintrin.h>

#include "matrix.hpp"
#include "utils.hpp"

constexpr int M_CACHE_TILE_INT8 = 512;
constexpr int N_CACHE_TILE_INT8 = 1024;
constexpr int K_CACHE_TILE_INT8 = 1024; 
constexpr int M_REGISTER_TILE_INT8 = 16; // 16 accumulators (ZMM)
constexpr int N_REGISTER_TILE_INT8 = 16; // 16 int32s fit in one ZMM

// The kernel processes 2 steps of K at once due to madd_epi16 pairing
constexpr int K_PACK_SIZE_INT8 = 2;      
constexpr int K_UNROLL_INT8 = 4;         // Process 4 groups of 2 (8 K total per loop)
constexpr int PARALLEL_M_THRESHOLD_INT8 = 64; 

/* Micro kernel for int8 GEMM (AVX-512 Int16 Upcast).
   
   Logic:
   C += A (int8) * B (int8)
   Internally upcasts to int16 to handle signs correctly without offsets.
*/
inline void microkernel_16x16_avx512_int8_upcast(
    const int8_t* __restrict__ A, 
    const int8_t* __restrict__ B, 
    int32_t* __restrict__ C, 
    int C_stride, int n_remain, int k_pad,
    bool first_k_block
) {
    constexpr int MR = M_REGISTER_TILE_INT8;
    constexpr int NR = N_REGISTER_TILE_INT8; // 16

    // MR=16 output accumulators (ZMM registers)
    __m512i C_acc[MR];

    // 1. Initialize Accumulators
    if (first_k_block) {
        #pragma unroll(MR)
        for (int i = 0; i < MR; ++i) {
            C_acc[i] = _mm512_setzero_si512();
        }
    } else {
        __mmask16 load_mask = (1U << n_remain) - 1;
        #pragma unroll(MR)
        for (int i = 0; i < MR; ++i) {
            C_acc[i] = _mm512_mask_loadu_epi32(C_acc[i], load_mask, &C[i * C_stride]);
        }
    }

    // 2. Compute
    // We process K in chunks of (K_PACK * K_UNROLL) = 8
    for (int k = 0; k < k_pad; k += K_PACK_SIZE_INT8 * K_UNROLL_INT8) {
        
        #pragma unroll(K_UNROLL_INT8)
        for (int u = 0; u < K_UNROLL_INT8; ++u) {
            
            /* Load B: 
               We need 16 columns. We process 2 rows (K, K+1).
               Packed B layout has interleaved K: [B(k,0), B(k+1,0), B(k,1), B(k+1,1)...]
               Total 16 cols * 2 rows = 32 bytes (256 bits).
            */
            __m256i b_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B));
            B += 32;

            // Upcast B to int16 (Sign Extend) -> fills 512-bit register
            __m512i v_b = _mm512_cvtepi8_epi16(b_raw);

            #pragma unroll(MR)
            for (int i = 0; i < MR; ++i) {
                // Load A: We need A[i, k] and A[i, k+1]
                int8_t val_k0 = A[i * 2 + 0];
                int8_t val_k1 = A[i * 2 + 1];

                // Upcast to int16 scalar
                int16_t val_k0_16 = static_cast<int16_t>(val_k0);
                int16_t val_k1_16 = static_cast<int16_t>(val_k1);

                // Combine into one int32 for broadcast: [val_k1_16 | val_k0_16]
                // This puts k0 in the lower 16 bits and k1 in the upper 16 bits of each lane
                int32_t packed_a = (static_cast<uint16_t>(val_k1_16) << 16) | static_cast<uint16_t>(val_k0_16);
                __m512i v_a = _mm512_set1_epi32(packed_a);

                // Multiply-Add:
                // madd_epi16 computes: (a_low * b_low) + (a_high * b_high) for each 32-bit lane.
                // i.e., A[k]*B[k] + A[k+1]*B[k+1]
                __m512i prod = _mm512_madd_epi16(v_b, v_a);

                // Accumulate 32-bit results
                C_acc[i] = _mm512_add_epi32(C_acc[i], prod);
            }
            // Move A pointer: MR rows * 2 columns
            A += MR * 2;
        }
    }

    // 3. Store
    __mmask16 store_mask = (1U << n_remain) - 1;
    #pragma unroll(MR)
    for (int i = 0; i < MR; ++i) {
        _mm512_mask_storeu_epi32(&C[i * C_stride], store_mask, C_acc[i]);
    }
}

inline void microkernel_cleanup_avx512_int8_upcast(
    const int8_t* __restrict__ A, 
    const int8_t* __restrict__ B, 
    int32_t* __restrict__ C, 
    int C_stride, int m_remain, int n_remain, int k_pad,
    bool first_k_block
) {
    constexpr int MR = M_REGISTER_TILE_INT8;
    
    __m512i C_acc[MR];
    __mmask16 mask = (1U << n_remain) - 1;

    if (first_k_block) {
        for (int i = 0; i < m_remain; ++i) {
            C_acc[i] = _mm512_setzero_si512();
        }
    } else {
        for (int i = 0; i < m_remain; ++i) {
            C_acc[i] = _mm512_mask_loadu_epi32(C_acc[i], mask, &C[i * C_stride]);
        }
    }

    // Process K in chunks of 2 (K_PACK_SIZE)
    for (int k = 0; k < k_pad; k += 2) {
        __m256i b_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B));
        B += 32;
        __m512i v_b = _mm512_cvtepi8_epi16(b_raw);

        for (int i = 0; i < m_remain; ++i) {
            int8_t val_k0 = A[i * 2 + 0];
            int8_t val_k1 = A[i * 2 + 1];

            int16_t val_k0_16 = static_cast<int16_t>(val_k0);
            int16_t val_k1_16 = static_cast<int16_t>(val_k1);

            int32_t packed_a = (static_cast<uint16_t>(val_k1_16) << 16) | static_cast<uint16_t>(val_k0_16);
            __m512i v_a = _mm512_set1_epi32(packed_a);

            __m512i prod = _mm512_madd_epi16(v_b, v_a);
            C_acc[i] = _mm512_add_epi32(C_acc[i], prod);
        }
        A += MR * 2; 
    }

    for (int i = 0; i < m_remain; ++i) {
        _mm512_mask_storeu_epi32(&C[i * C_stride], mask, C_acc[i]);
    }
}

/* Pack A:
   Simple packing. We process K in pairs of 2.
   Layout: [Block K=0..k_pad][Row 0..MR]
   Inside the inner block, we store [A(row, k), A(row, k+1)].
*/
inline void packA_avx512_int8(
    const int8_t *__restrict__ A_tile,
    int8_t *__restrict__ A_packed,
    int A_stride,
    int micropanel_stride,
    int m_end, int k_end, int k_pad
) {
    constexpr int MR = M_REGISTER_TILE_INT8;
    int i_register = 0;

    for (; (i_register + 1) * MR <= m_end; ++i_register) {
        int8_t* dst_panel = &A_packed[i_register * micropanel_stride];
        
        int k = 0;
        for (; k < k_end; k += 2) {
            for (int i = 0; i < MR; ++i) {
                const int8_t* src = &A_tile[(i_register * MR + i) * A_stride + k];
                
                int8_t a0 = src[0];
                int8_t a1 = 0;
                if (k + 1 < k_end) {
                    a1 = src[1];
                }

                dst_panel[0] = a0;
                dst_panel[1] = a1;
                dst_panel += 2;
            }
        }
        
        // Zero pad K
        for (; k < k_pad; k += 2) {
            for (int i = 0; i < MR; ++i) {
                dst_panel[0] = 0;
                dst_panel[1] = 0;
                dst_panel += 2;
            }
        }
    }

    // Cleanup for m_remain
    int m_remain = m_end - i_register * MR;
    if (m_remain > 0) {
        int8_t* dst_panel = &A_packed[i_register * micropanel_stride];
        int k = 0;
        for (; k < k_pad; k += 2) {
            for (int i = 0; i < m_remain; ++i) {
                const int8_t* src = &A_tile[(i_register * MR + i) * A_stride + k];
                int8_t a0 = (k < k_end) ? src[0] : 0;
                int8_t a1 = (k + 1 < k_end) ? src[1] : 0;
                
                dst_panel[0] = a0;
                dst_panel[1] = a1;
                dst_panel += 2;
            }
            // Pad the rest of the register rows with 0
            for (int i = m_remain; i < MR; ++i) {
                dst_panel[0] = 0;
                dst_panel[1] = 0;
                dst_panel += 2;
            }
        }
    }
}

/* Pack B: Interleaving for madd_epi16.
   
   We pack NR=16 columns. 
   We process K in pairs (k, k+1).
   
   Memory Layout required for microkernel:
   For each K-pair:
     [ B(k,0), B(k+1,0), B(k,1), B(k+1,1), ... , B(k,15), B(k+1,15) ]
   
   Total 32 bytes per K-pair.
*/
inline void packB_avx512_int8(
    const int8_t *__restrict__ B_tile,
    int8_t *__restrict__ B_packed,
    int B_stride,
    int micropanel_stride,
    int n_end, int k_end, int k_pad
) {
    constexpr int NR = N_REGISTER_TILE_INT8;
    int j_register = 0;

    for (; (j_register + 1) * NR <= n_end; ++j_register) {
        int8_t* dst_panel = &B_packed[j_register * micropanel_stride];
        
        int k = 0;
        for (; k < k_end; k += 2) {
            const int8_t* src_k0 = &B_tile[k * B_stride + j_register * NR];
            const int8_t* src_k1 = &B_tile[(k + 1) * B_stride + j_register * NR];
            bool has_k1 = (k + 1 < k_end);

            for (int j = 0; j < NR; ++j) {
                dst_panel[0] = src_k0[j];
                dst_panel[1] = has_k1 ? src_k1[j] : 0;
                dst_panel += 2;
            }
        }
        
        // Pad K
        for (; k < k_pad; k += 2) {
            memset(dst_panel, 0, NR * 2);
            dst_panel += NR * 2;
        }
    }
    
    // Cleanup n_remain
    int n_remain = n_end - j_register * NR;
    if (n_remain > 0) {
        int8_t* dst_panel = &B_packed[j_register * micropanel_stride];
        int k = 0;
        for (; k < k_pad; k += 2) {
            const int8_t* src_k0 = &B_tile[k * B_stride + j_register * NR];
            const int8_t* src_k1 = &B_tile[(k + 1) * B_stride + j_register * NR];
            bool has_k0 = (k < k_end);
            bool has_k1 = (k + 1 < k_end);

            for (int j = 0; j < NR; ++j) {
                if (j < n_remain) {
                    dst_panel[0] = has_k0 ? src_k0[j] : 0;
                    dst_panel[1] = has_k1 ? src_k1[j] : 0;
                } else {
                    dst_panel[0] = 0;
                    dst_panel[1] = 0;
                }
                dst_panel += 2;
            }
        }
    }
}

inline void gemm_int8_avx512_tiled_parallel_N(
    const int8_t* __restrict__ A, 
    const int8_t* __restrict__ B, 
    int32_t* __restrict__ C, 
    const int M, const int N, const int K
) {
    const int num_threads = omp_get_max_threads();
    const int MR = M_REGISTER_TILE_INT8;
    const int NR = N_REGISTER_TILE_INT8;
    
    const int KC = round_up(std::min(K_CACHE_TILE_INT8, K), K_PACK_SIZE_INT8); 
    const int MC = std::min(M_CACHE_TILE_INT8, M); 
    const int NC = round_up(clamp(N / num_threads, NR, N_CACHE_TILE_INT8), NR);

    #pragma omp parallel
    {
        // Allocation: KC * MR * sizeof(int8) (since we pack int8)
        const int max_micropanel_stride_A = round_up(KC * MR, ALIGNMENT);
        const int n_micropanels_A = ceil_division(MC, MR);
        size_t bytes_A_packed = round_up(n_micropanels_A * max_micropanel_stride_A, ALIGNMENT);
        auto* A_packed = reinterpret_cast<int8_t*>(aligned_alloc(ALIGNMENT, bytes_A_packed));

        const int max_micropanel_stride_B = round_up(KC * NR, ALIGNMENT);
        const int n_micropanels_B = ceil_division(NC, NR);
        size_t bytes_B_packed = round_up(n_micropanels_B * max_micropanel_stride_B, ALIGNMENT);
        auto* B_packed = reinterpret_cast<int8_t*>(aligned_alloc(ALIGNMENT, bytes_B_packed));

        int j_cache_end = ceil_division(N, NC);

        #pragma omp for schedule(static)
        for (int j_cache = 0; j_cache < j_cache_end; ++j_cache) {
            int n_end = std::min(NC, N - j_cache * NC);

            for (int k_cache = 0; k_cache * KC < K; ++k_cache) {
                const int k_end = std::min(KC, K - k_cache * KC);
                const int k_pad = round_up(k_end, K_PACK_SIZE_INT8);

                // Note: Stride in bytes is essentially rows * cols for packed buffers
                const int micropanel_stride_A = round_up(k_pad * MR, ALIGNMENT);
                const int micropanel_stride_B = round_up(k_pad * NR, ALIGNMENT);

                // Pack B (No compensation needed)
                packB_avx512_int8(
                    &B[(k_cache * KC) * N + (j_cache * NC)], B_packed,
                    N, micropanel_stride_B, n_end, k_end, k_pad
                );

                int i_cache_end = ceil_division(M, MC);
                for (int i_cache = 0; i_cache < i_cache_end; ++i_cache) {
                    int m_end = std::min(MC, M - i_cache * MC);

                    packA_avx512_int8(
                        &A[(i_cache * MC) * K + (k_cache * KC)], A_packed, 
                        K, micropanel_stride_A, m_end, k_end, k_pad
                    );
            
                    int i_register = 0;
                    for (; (i_register + 1) * MR <= m_end; ++i_register) {
                        for (int j_register = 0; j_register * NR < n_end; ++j_register) {
                            int n_remain = std::min(NR, n_end - j_register * NR);
                            
                            microkernel_16x16_avx512_int8_upcast(
                                &A_packed[i_register * micropanel_stride_A],
                                &B_packed[j_register * micropanel_stride_B],
                                &C[(i_cache * MC + i_register * MR) * N + (j_cache * NC + j_register * NR)],
                                N, n_remain, k_pad,
                                (k_cache == 0)
                            );
                        }
                    }
                    
                    int m_remain = std::min(MR, m_end - i_register * MR);
                    if (m_remain != 0) {
                        for (int j_register = 0; j_register * NR < n_end; ++j_register) {
                            int n_remain = std::min(NR, n_end - j_register * NR);
                            microkernel_cleanup_avx512_int8_upcast(
                                &A_packed[i_register * micropanel_stride_A],
                                &B_packed[j_register * micropanel_stride_B],
                                &C[(i_cache * MC + i_register * MR) * N + (j_cache * NC + j_register * NR)],
                                N, m_remain, n_remain, k_pad,
                                (k_cache == 0)
                            );
                        }
                    }
                }
            }
        }
        std::free(A_packed);
        std::free(B_packed);
    }
}

inline void gemm_int8_avx512_tiled_parallel_M(
    const int8_t* __restrict__ A, 
    const int8_t* __restrict__ B, 
    int32_t* __restrict__ C, 
    const int M, const int N, const int K
) {
    const int num_threads = omp_get_max_threads();
    const int MR = M_REGISTER_TILE_INT8;
    const int NR = N_REGISTER_TILE_INT8;

    const int KC = round_up(std::min(K_CACHE_TILE_INT8, K), K_PACK_SIZE_INT8); 
    const int MC = clamp(M / num_threads, 1, M_CACHE_TILE_INT8); 
    const int NC = round_up(std::min(N, N_CACHE_TILE_INT8), NR); 

    const int max_micropanel_stride_B = round_up(KC * NR, ALIGNMENT);
    const int n_micropanels_B = ceil_division(NC, NR);
    size_t bytes_B_packed = round_up(size_t(n_micropanels_B) * max_micropanel_stride_B, ALIGNMENT);
    auto* B_packed = reinterpret_cast<int8_t*>(aligned_alloc(ALIGNMENT, bytes_B_packed));
    
    #pragma omp parallel
    {
        const int max_micropanel_stride_A = round_up(KC * MR, ALIGNMENT);
        const int n_micropanels_A = ceil_division(MC, MR);
        size_t bytes_A_packed = round_up(n_micropanels_A * max_micropanel_stride_A, ALIGNMENT);
        auto* A_packed = reinterpret_cast<int8_t*>(aligned_alloc(ALIGNMENT, bytes_A_packed));

        for (int j_cache = 0; j_cache * NC < N; ++j_cache) {
            int n_end = std::min(NC, N - j_cache * NC);

            for (int k_cache = 0; k_cache * KC < K; ++k_cache) {
                const int k_end = std::min(KC, K - k_cache * KC);
                const int k_pad = round_up(k_end, K_PACK_SIZE_INT8);

                const int micropanel_stride_A = round_up(k_pad * MR, ALIGNMENT);
                const int micropanel_stride_B = round_up(k_pad * NR, ALIGNMENT);

                #pragma omp single 
                packB_avx512_int8(
                    &B[(k_cache * KC) * N + (j_cache * NC)], B_packed,
                    N, micropanel_stride_B, n_end, k_end, k_pad
                );

                int i_cache_end = ceil_division(M, MC);

                #pragma omp for schedule(static)
                for (int i_cache = 0; i_cache < i_cache_end; ++i_cache) {
                    int m_end = std::min(MC, M - i_cache * MC);

                    packA_avx512_int8(
                        &A[(i_cache * MC) * K + (k_cache * KC)], A_packed, 
                        K, micropanel_stride_A, m_end, k_end, k_pad
                    );
            
                    int i_register = 0;
                    for (; (i_register + 1) * MR <= m_end; ++i_register) {
                        for (int j_register = 0; j_register * NR < n_end; ++j_register) {
                            int n_remain = std::min(NR, n_end - j_register * NR);
                            microkernel_16x16_avx512_int8_upcast(
                                &A_packed[i_register * micropanel_stride_A],
                                &B_packed[j_register * micropanel_stride_B],
                                &C[(i_cache * MC + i_register * MR) * N + (j_cache * NC + j_register * NR)],
                                N, n_remain, k_pad,
                                (k_cache == 0)
                            );
                        }
                    }
                    
                    int m_remain = std::min(MR, m_end - i_register * MR);
                    if (m_remain != 0) {
                        for (int j_register = 0; j_register * NR < n_end; ++j_register) {
                            int n_remain = std::min(NR, n_end - j_register * NR);
                            microkernel_cleanup_avx512_int8_upcast(
                                &A_packed[i_register * micropanel_stride_A],
                                &B_packed[j_register * micropanel_stride_B],
                                &C[(i_cache * MC + i_register * MR) * N + (j_cache * NC + j_register * NR)],
                                N, m_remain, n_remain, k_pad,
                                (k_cache == 0)
                            );
                        }
                    }
                }
            }
        }
        std::free(A_packed);
    }
    std::free(B_packed);
}

inline void gemm_int8_tiled(const Matrix& A_base, const Matrix& B_base, Matrix& C_base) {
    const auto& A = static_cast<const MatrixINT8&>(A_base);
    const auto& B = static_cast<const MatrixINT8&>(B_base);
    auto& C = static_cast<MatrixINT32&>(C_base);

    const int M = A.rows;
    const int K = A.cols;
    const int N = B.cols;

    const auto* A_ptr = reinterpret_cast<const int8_t*>(A.raw_data);
    const auto* B_ptr = reinterpret_cast<const int8_t*>(B.raw_data);
    auto* C_ptr = reinterpret_cast<int32_t*>(C.raw_data);

    if (M > N) {
        return gemm_int8_avx512_tiled_parallel_M(A_ptr, B_ptr, C_ptr, M, N, K);
    } else {
        return gemm_int8_avx512_tiled_parallel_N(A_ptr, B_ptr, C_ptr, M, N, K);
    }
}

inline void gemm_int8_naive(const Matrix& A_base, const Matrix& B_base, Matrix& C_base) {
    const auto& A = static_cast<const MatrixINT8&>(A_base);
    const auto& B = static_cast<const MatrixINT8&>(B_base);
    auto& C = static_cast<MatrixINT32&>(C_base);

    const int8_t* A_ptr = reinterpret_cast<const int8_t*>(A.raw_data);
    const int8_t* B_ptr = reinterpret_cast<const int8_t*>(B.raw_data);
    int32_t* C_ptr = reinterpret_cast<int32_t*>(C.raw_data);

    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            int32_t sum = 0;
            for (int k = 0; k < A.cols; ++k) {
                sum += static_cast<int32_t>(A_ptr[i * A.cols + k]) * static_cast<int32_t>(B_ptr[k * B.cols + j]);
            }
            C_ptr[i * C.cols + j] = sum;
        }
    }
}
