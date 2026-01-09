#pragma once 

#include <iostream>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <omp.h>
#include <immintrin.h>

#include "matrix.hpp"
#include "bfloat16.hpp"
#include "utils.hpp"

/* Bf16 tiling sizes for AVX-512.

    Changes from AVX2:
    - Register tiles increased to 16x16 to exploit ZMM registers (16 floats per reg).
    - Cache tiles tuned slightly for the increased throughput.
*/
constexpr int M_CACHE_TILE_BF16 = 512;
constexpr int N_CACHE_TILE_BF16 = 1024;
constexpr int K_CACHE_TILE_BF16 = 512;
constexpr int M_REGISTER_TILE_BF16 = 16; // AVX-512 ZMM holds 16 floats. We use 16 accumulators.
constexpr int N_REGISTER_TILE_BF16 = 16; // 16 floats in 512bit ZMM register.
constexpr int K_UNROLL_BF16 = 4; // Unroll factor
constexpr int PARALLEL_M_THRESHOLD = 64; 

/* Micro kernel for bf16 GEMM (AVX-512).
    
    We will write to a [MR x n_remain] tile of C.
    
    We assume: 
    - MR == 16, NR == 16.
    - B_packed is padded to multiples of NR (16).
    - n_remain can be 1..16.
*/
inline void microkernel_16x16_avx512_bf16(
    const bfloat16_t* __restrict__ A, 
    const bfloat16_t* __restrict__ B, 
    float* __restrict__ C, 
    int C_stride, int n_remain, int k_pad,
    bool first_k_block
) {
    constexpr int MR = M_REGISTER_TILE_BF16;
    constexpr int NR = N_REGISTER_TILE_BF16;

    static_assert(MR == 16, "microkernel assumes MR == 16");
    static_assert(NR == 16, "microkernel assumes NR == 16");

    // MR=16 output accumulators (ZMM registers)
    __m512 C_acc[MR];

    // Handling Accumulators
    if (first_k_block) {
        #pragma unroll(MR)
        for (int i = 0; i < MR; ++i) {
            C_acc[i] = _mm512_setzero_ps();
        }
    } else {
        // Load partial sums. 
        // We use masking for the load if n_remain < 16 to avoid reading OOB, 
        // though strictly C is usually safe to over-read if padded, but masking is safer/cleaner in AVX512.
        __mmask16 load_mask = (1U << n_remain) - 1;
        
        #pragma unroll(MR)
        for (int i = 0; i < MR; ++i) {
            C_acc[i] = _mm512_mask_loadu_ps(C_acc[i], load_mask, &C[i * C_stride]);
        }
    }

    // Hot loop
    #pragma unroll(K_UNROLL_BF16)
    for (int k = 0; k < k_pad; ++k) {
        // 1. Load B: 16 bf16 values. This is 256 bits (32 bytes).
        // We load it into a YMM (half ZMM) and then expand.
        __m256i v_b_bf16_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&B[k * NR]));
        
        // Convert B from bf16 to fp32 (ZMM)
        // Expand 16x16bit -> 16x32bit integers (zero extended)
        __m512i v_b_extended = _mm512_cvtepu16_epi32(v_b_bf16_256);
        // Shift left by 16 to move bits to high half
        __m512 v_b = _mm512_castsi512_ps(_mm512_slli_epi32(v_b_extended, 16));

        #pragma unroll(MR)
        for (int i = 0; i < MR; ++i) {
            // Load A: A is stored column-major within the micropanel, so A[k*MR + i] is adjacent.
            // We need to broadcast the single scalar A[k, i] to all 16 lanes of the register.
            
            // Broadcast 16-bit value to all 32-bit elements of a ZMM
            // (There isn't a direct set1_epi16 to ZMM that treats them as paired 32-bit slots conveniently for BF16,
            // so we broadcast to 32-bit int, then shift).
            // Actually, cleanest way for a single scalar:
            __m512i v_a_broadcast = _mm512_set1_epi32((int)A[k * MR + i]); 
            // Shift left to position the bf16 in the top of the float
            __m512 v_a = _mm512_castsi512_ps(_mm512_slli_epi32(v_a_broadcast, 16));

            // Fused Multiply Add
            C_acc[i] = _mm512_fmadd_ps(v_a, v_b, C_acc[i]);
        }
    }

    // Write back to C
    // We utilize AVX-512 masking to handle the n_remain case cleanly without a scalar loop.
    __mmask16 store_mask = (1U << n_remain) - 1;

    #pragma unroll(MR)
    for (int i = 0; i < MR; ++i) {
        _mm512_mask_storeu_ps(&C[i * C_stride], store_mask, C_acc[i]);
    }
}

/* Cleanup micro kernel.
    Handles cases where m_remain < MR.
    
    Thanks to AVX-512 masking, this logic is much simpler than AVX2. 
    We just loop up to m_remain and use masks for n_remain.
*/
inline void microkernel_cleanup_avx512_bf16(
    const bfloat16_t* __restrict__ A, 
    const bfloat16_t* __restrict__ B, 
    float* __restrict__ C, 
    int C_stride, int m_remain, int n_remain, int k_pad,
    bool first_k_block
) {
    constexpr int MR = M_REGISTER_TILE_BF16;
    constexpr int NR = N_REGISTER_TILE_BF16;

    // Accumulators (we only use the first m_remain ones)
    __m512 C_acc[MR];
    __mmask16 mask = (1U << n_remain) - 1;

    if (first_k_block) {
        for (int i = 0; i < m_remain; ++i) {
            C_acc[i] = _mm512_setzero_ps();
        }
    } else {
        for (int i = 0; i < m_remain; ++i) {
            C_acc[i] = _mm512_mask_loadu_ps(C_acc[i], mask, &C[i * C_stride]);
        }
    }

    // Hot loop
    #pragma unroll(K_UNROLL_BF16)
    for (int k = 0; k < k_pad; ++k) {
        __m256i v_b_bf16_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&B[k * NR]));
        __m512i v_b_extended = _mm512_cvtepu16_epi32(v_b_bf16_256);
        __m512 v_b = _mm512_castsi512_ps(_mm512_slli_epi32(v_b_extended, 16));

        for (int i = 0; i < m_remain; ++i) {
            __m512i v_a_broadcast = _mm512_set1_epi32((int)A[k * m_remain + i]);
            __m512 v_a = _mm512_castsi512_ps(_mm512_slli_epi32(v_a_broadcast, 16));
            C_acc[i] = _mm512_fmadd_ps(v_a, v_b, C_acc[i]);
        }
    }

    // Store
    for (int i = 0; i < m_remain; ++i) {
        _mm512_mask_storeu_ps(&C[i * C_stride], mask, C_acc[i]);
    }
}

/* Read a 16x16 block of bfloat16_t from src, transpose it, and write to dst.
    
    16x16 block of bf16 = 256 scalars.
    Row size = 16 * 2 bytes = 32 bytes (256 bits).
    Column size = 16 * 2 bytes = 32 bytes (256 bits).

    Since the rows fit exactly into YMM registers (256-bit), we can perform the 
    transpose using 16 YMM registers and standard recursive merge/split logic 
    (depth 4) on __m256i. This avoids needing 32 ZMMs or complex permutations.
*/
inline void transpose_16x16_avx512_bf16(
    const bfloat16_t* __restrict__ src, 
    bfloat16_t* __restrict__ dst, 
    int src_stride, 
    int dst_stride
) {
    // We need 16 registers.
    __m256i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
    __m256i t0, t1, t2, t3, t4, t5, t6, t7; // Temporaries

    // Load 16 rows (32 bytes each)
    r0 = _mm256_loadu_si256((const __m256i*)&src[0 * src_stride]);
    r1 = _mm256_loadu_si256((const __m256i*)&src[1 * src_stride]);
    r2 = _mm256_loadu_si256((const __m256i*)&src[2 * src_stride]);
    r3 = _mm256_loadu_si256((const __m256i*)&src[3 * src_stride]);
    r4 = _mm256_loadu_si256((const __m256i*)&src[4 * src_stride]);
    r5 = _mm256_loadu_si256((const __m256i*)&src[5 * src_stride]);
    r6 = _mm256_loadu_si256((const __m256i*)&src[6 * src_stride]);
    r7 = _mm256_loadu_si256((const __m256i*)&src[7 * src_stride]);
    r8 = _mm256_loadu_si256((const __m256i*)&src[8 * src_stride]);
    r9 = _mm256_loadu_si256((const __m256i*)&src[9 * src_stride]);
    r10 = _mm256_loadu_si256((const __m256i*)&src[10 * src_stride]);
    r11 = _mm256_loadu_si256((const __m256i*)&src[11 * src_stride]);
    r12 = _mm256_loadu_si256((const __m256i*)&src[12 * src_stride]);
    r13 = _mm256_loadu_si256((const __m256i*)&src[13 * src_stride]);
    r14 = _mm256_loadu_si256((const __m256i*)&src[14 * src_stride]);
    r15 = _mm256_loadu_si256((const __m256i*)&src[15 * src_stride]);

    // Stage 1: Unpack 16-bit (1 scalar)
    t0 = _mm256_unpacklo_epi16(r0, r1); t1 = _mm256_unpackhi_epi16(r0, r1);
    t2 = _mm256_unpacklo_epi16(r2, r3); t3 = _mm256_unpackhi_epi16(r2, r3);
    t4 = _mm256_unpacklo_epi16(r4, r5); t5 = _mm256_unpackhi_epi16(r4, r5);
    t6 = _mm256_unpacklo_epi16(r6, r7); t7 = _mm256_unpackhi_epi16(r6, r7);
    r0 = t0; r1 = t1; r2 = t2; r3 = t3; r4 = t4; r5 = t5; r6 = t6; r7 = t7;

    t0 = _mm256_unpacklo_epi16(r8, r9); t1 = _mm256_unpackhi_epi16(r8, r9);
    t2 = _mm256_unpacklo_epi16(r10, r11); t3 = _mm256_unpackhi_epi16(r10, r11);
    t4 = _mm256_unpacklo_epi16(r12, r13); t5 = _mm256_unpackhi_epi16(r12, r13);
    t6 = _mm256_unpacklo_epi16(r14, r15); t7 = _mm256_unpackhi_epi16(r14, r15);
    r8 = t0; r9 = t1; r10 = t2; r11 = t3; r12 = t4; r13 = t5; r14 = t6; r15 = t7;

    // Stage 2: Unpack 32-bit (2 scalars)
    t0 = _mm256_unpacklo_epi32(r0, r2); t1 = _mm256_unpackhi_epi32(r0, r2);
    t2 = _mm256_unpacklo_epi32(r1, r3); t3 = _mm256_unpackhi_epi32(r1, r3);
    t4 = _mm256_unpacklo_epi32(r4, r6); t5 = _mm256_unpackhi_epi32(r4, r6);
    t6 = _mm256_unpacklo_epi32(r5, r7); t7 = _mm256_unpackhi_epi32(r5, r7);
    r0 = t0; r1 = t1; r2 = t2; r3 = t3; r4 = t4; r5 = t5; r6 = t6; r7 = t7;

    t0 = _mm256_unpacklo_epi32(r8, r10); t1 = _mm256_unpackhi_epi32(r8, r10);
    t2 = _mm256_unpacklo_epi32(r9, r11); t3 = _mm256_unpackhi_epi32(r9, r11);
    t4 = _mm256_unpacklo_epi32(r12, r14); t5 = _mm256_unpackhi_epi32(r12, r14);
    t6 = _mm256_unpacklo_epi32(r13, r15); t7 = _mm256_unpackhi_epi32(r13, r15);
    r8 = t0; r9 = t1; r10 = t2; r11 = t3; r12 = t4; r13 = t5; r14 = t6; r15 = t7;

    // Stage 3: Unpack 64-bit (4 scalars)
    t0 = _mm256_unpacklo_epi64(r0, r4); t1 = _mm256_unpackhi_epi64(r0, r4);
    t2 = _mm256_unpacklo_epi64(r1, r5); t3 = _mm256_unpackhi_epi64(r1, r5);
    t4 = _mm256_unpacklo_epi64(r2, r6); t5 = _mm256_unpackhi_epi64(r2, r6);
    t6 = _mm256_unpacklo_epi64(r3, r7); t7 = _mm256_unpackhi_epi64(r3, r7);
    r0 = t0; r1 = t1; r2 = t2; r3 = t3; r4 = t4; r5 = t5; r6 = t6; r7 = t7;

    t0 = _mm256_unpacklo_epi64(r8, r12); t1 = _mm256_unpackhi_epi64(r8, r12);
    t2 = _mm256_unpacklo_epi64(r9, r13); t3 = _mm256_unpackhi_epi64(r9, r13);
    t4 = _mm256_unpacklo_epi64(r10, r14); t5 = _mm256_unpackhi_epi64(r10, r14);
    t6 = _mm256_unpacklo_epi64(r11, r15); t7 = _mm256_unpackhi_epi64(r11, r15);
    r8 = t0; r9 = t1; r10 = t2; r11 = t3; r12 = t4; r13 = t5; r14 = t6; r15 = t7;

    // Stage 4: Unpack 128-bit (8 scalars), we replicate unpacklo/hi with a more general permute instruction
    t0 = _mm256_permute2x128_si256(r0, r8, 0x20);  // Lo 128bits of r0, Lo r8
    t1 = _mm256_permute2x128_si256(r0, r8, 0x31);  // Hi r0, Hi r8
    t2 = _mm256_permute2x128_si256(r1, r9, 0x20);
    t3 = _mm256_permute2x128_si256(r1, r9, 0x31);
    t4 = _mm256_permute2x128_si256(r2, r10, 0x20);
    t5 = _mm256_permute2x128_si256(r2, r10, 0x31);
    t6 = _mm256_permute2x128_si256(r3, r11, 0x20);
    t7 = _mm256_permute2x128_si256(r3, r11, 0x31);
    r0 = t0; r1 = t1; r2 = t2; r3 = t3; r4 = t4; r5 = t5; r6 = t6; r7 = t7;

    t0 = _mm256_permute2x128_si256(r4, r12, 0x20);
    t1 = _mm256_permute2x128_si256(r4, r12, 0x31);
    t2 = _mm256_permute2x128_si256(r5, r13, 0x20);
    t3 = _mm256_permute2x128_si256(r5, r13, 0x31);
    t4 = _mm256_permute2x128_si256(r6, r14, 0x20);
    t5 = _mm256_permute2x128_si256(r6, r14, 0x31);
    t6 = _mm256_permute2x128_si256(r7, r15, 0x20);
    t7 = _mm256_permute2x128_si256(r7, r15, 0x31);
    r8 = t0; r9 = t1; r10 = t2; r11 = t3; r12 = t4; r13 = t5; r14 = t6; r15 = t7;

    // Store
    _mm256_storeu_si256((__m256i*)&dst[0 * dst_stride], r0);
    _mm256_storeu_si256((__m256i*)&dst[1 * dst_stride], r1);
    _mm256_storeu_si256((__m256i*)&dst[2 * dst_stride], r2);
    _mm256_storeu_si256((__m256i*)&dst[3 * dst_stride], r3);
    _mm256_storeu_si256((__m256i*)&dst[4 * dst_stride], r4);
    _mm256_storeu_si256((__m256i*)&dst[5 * dst_stride], r5);
    _mm256_storeu_si256((__m256i*)&dst[6 * dst_stride], r6);
    _mm256_storeu_si256((__m256i*)&dst[7 * dst_stride], r7);
    _mm256_storeu_si256((__m256i*)&dst[8 * dst_stride], r8);
    _mm256_storeu_si256((__m256i*)&dst[9 * dst_stride], r9);
    _mm256_storeu_si256((__m256i*)&dst[10 * dst_stride], r10);
    _mm256_storeu_si256((__m256i*)&dst[11 * dst_stride], r11);
    _mm256_storeu_si256((__m256i*)&dst[12 * dst_stride], r12);
    _mm256_storeu_si256((__m256i*)&dst[13 * dst_stride], r13);
    _mm256_storeu_si256((__m256i*)&dst[14 * dst_stride], r14);
    _mm256_storeu_si256((__m256i*)&dst[15 * dst_stride], r15);
}

/* Pack A (AVX-512)
    Uses 16x16 transpose.
*/
inline void packA_avx512_bf16(
    const bfloat16_t *__restrict__ A_tile,
    bfloat16_t *__restrict__ A_packed,
    int A_stride,
    int micropanel_stride,
    int m_end, int k_end, int k_pad
) {
    constexpr int MR = M_REGISTER_TILE_BF16; // 16
    int i_register = 0;

    // Main packing loop
    for (; (i_register + 1) * MR <= m_end; ++i_register) {
        int k = 0;
        // Transpose 16x16 blocks
        for (; k + MR <= k_end; k += MR) {
            const auto* src = &A_tile[(i_register * MR) * A_stride + k];
            auto* dst = &A_packed[i_register * micropanel_stride + k * MR];
            transpose_16x16_avx512_bf16(src, dst, A_stride, MR);
        }

        // Cleanup k (copy scalar)
        for (; k < k_end; ++k) {
            const auto* src = &A_tile[(i_register * MR) * A_stride + k];
            auto* dst = &A_packed[i_register * micropanel_stride + k * MR];
            // Use AVX loads/stores for the column if possible, but simple copy is safe here
            for (int i = 0; i < MR; ++i) {
                dst[i] = src[i * A_stride];
            }
        }

        // Zero pad k
        for (; k < k_pad; ++k) {
            auto* dst = &A_packed[i_register * micropanel_stride + k * MR];
            // 16 bf16s = 32 bytes = YMM
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), _mm256_setzero_si256());
        }
    }

    // Cleanup i (m_remain < 16)
    int m_remain = m_end - i_register * MR;
    if (m_remain > 0) {
        int k = 0;
        for (; k < k_end; ++k) {
            const auto* src = &A_tile[(i_register * MR) * A_stride + k];
            auto* dst = &A_packed[i_register * micropanel_stride + k * m_remain];
            for (int i = 0; i < m_remain; ++i) {
                dst[i] = src[i * A_stride];
            }
        }
        for (; k < k_pad; ++k) {
            auto* dst = &A_packed[i_register * micropanel_stride + k * m_remain];
            for (int i = 0; i < m_remain; ++i) {
                dst[i] = 0;
            }
        }
    }
}

/* Pack B (AVX-512)
    Uses 256-bit load/store instructions to move 16 bf16s (32 bytes) at once.
*/
inline void packB_avx512_bf16(
    const bfloat16_t *__restrict__ B_tile,
    bfloat16_t *__restrict__ B_packed,
    int B_stride,
    int micropanel_stride,
    int n_end, int k_end, int k_pad
) {
    constexpr int NR = N_REGISTER_TILE_BF16; // 16
    int j_register = 0; 
    
    // Main SIMD loop
    for (; (j_register + 1) * NR <= n_end; ++j_register) {
        int k = 0;
        for (; k < k_end; ++k) {
            const auto* src = &B_tile[k * B_stride + (j_register * NR)];
            auto* dst = &B_packed[j_register * micropanel_stride + k * NR];
            // Copy 16 bf16s (32 bytes) using YMM
            __m256i tmp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
            _mm256_store_si256(reinterpret_cast<__m256i*>(dst), tmp);
        }

        // Pad k
        for (; k < k_pad; ++k) {
            auto* dst = &B_packed[j_register * micropanel_stride + k * NR];
            _mm256_store_si256(reinterpret_cast<__m256i*>(dst), _mm256_setzero_si256());
        }
    }

    // Cleanup j
    int n_remain = n_end - j_register * NR;
    if (n_remain == 0) return; 

    // Zero entire last micropanel for padding safety
    for (int k = 0; k < k_pad; ++k) {
        auto* dst = &B_packed[j_register * micropanel_stride + k * NR];
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst), _mm256_setzero_si256());
    }

    // Copy remaining
    for (int k = 0; k < k_end; ++k) {
        for (int j = 0; j < n_remain; ++j) {
            auto src_val = B_tile[k * B_stride + (j_register * NR + j)];
            B_packed[j_register * micropanel_stride + k * NR + j] = src_val;
        }
    }
}

inline void gemm_bf16_avx512_tiled_parallel_N(
    const bfloat16_t* __restrict__ A, 
    const bfloat16_t* __restrict__ B, 
    float* __restrict__ C, 
    const int M, const int N, const int K
) {
    const int num_threads = omp_get_max_threads();
    const int MR = M_REGISTER_TILE_BF16;
    const int NR = N_REGISTER_TILE_BF16;
    
    // Tuned for ZMM
    const int KC = round_up(std::min(K_CACHE_TILE_BF16, K), K_UNROLL_BF16); 
    const int MC = std::min(M_CACHE_TILE_BF16, M); 
    const int NC = round_up(clamp(N / num_threads, NR, N_CACHE_TILE_BF16), NR);

    #pragma omp parallel
    {
        const int max_micropanel_stride_A = round_up(KC * MR, ALIGNMENT / sizeof(bfloat16_t));
        const int n_micropanels_A = ceil_division(MC, MR);
        size_t bytes_A_packed = round_up(n_micropanels_A * max_micropanel_stride_A * sizeof(bfloat16_t), ALIGNMENT);
        auto* A_packed = reinterpret_cast<bfloat16_t*>(aligned_alloc(ALIGNMENT, bytes_A_packed));

        const int max_micropanel_stride_B = round_up(KC * NR, ALIGNMENT / sizeof(bfloat16_t));
        const int n_micropanels_B = ceil_division(NC, NR);
        size_t bytes_B_packed = round_up(n_micropanels_B * max_micropanel_stride_B * sizeof(bfloat16_t), ALIGNMENT);
        auto* B_packed = reinterpret_cast<bfloat16_t*>(aligned_alloc(ALIGNMENT, bytes_B_packed));

        int j_cache_end = ceil_division(N, NC);

        #pragma omp for schedule(static)
        for (int j_cache = 0; j_cache < j_cache_end; ++j_cache) {
            int n_end = std::min(NC, N - j_cache * NC);

            for (int k_cache = 0; k_cache * KC < K; ++k_cache) {
                const int k_end = std::min(KC, K - k_cache * KC);
                const int k_pad = round_up(k_end, K_UNROLL_BF16);

                const int micropanel_stride_A = round_up(k_pad * MR, ALIGNMENT / sizeof(bfloat16_t));
                const int micropanel_stride_B = round_up(k_pad * NR, ALIGNMENT / sizeof(bfloat16_t));

                packB_avx512_bf16(
                    &B[(k_cache * KC) * N + (j_cache * NC)], B_packed, 
                    N, micropanel_stride_B, n_end, k_end, k_pad
                );

                int i_cache_end = ceil_division(M, MC);
                for (int i_cache = 0; i_cache < i_cache_end; ++i_cache) {
                    int m_end = std::min(MC, M - i_cache * MC);

                    packA_avx512_bf16(
                        &A[(i_cache * MC) * K + (k_cache * KC)], A_packed, 
                        K, micropanel_stride_A, m_end, k_end, k_pad
                    );
            
                    int i_register = 0;
                    for (; (i_register + 1) * MR <= m_end; ++i_register) {
                        for (int j_register = 0; j_register * NR < n_end; ++j_register) {
                            int n_remain = std::min(NR, n_end - j_register * NR);
                            microkernel_16x16_avx512_bf16(
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
                            microkernel_cleanup_avx512_bf16(
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

inline void gemm_bf16_avx512_tiled_parallel_M(
    const bfloat16_t* __restrict__ A, 
    const bfloat16_t* __restrict__ B, 
    float* __restrict__ C, 
    const int M, const int N, const int K
) {
    const int num_threads = omp_get_max_threads();
    const int MR = M_REGISTER_TILE_BF16;
    const int NR = N_REGISTER_TILE_BF16;

    const int KC = round_up(std::min(K_CACHE_TILE_BF16, K), K_UNROLL_BF16); 
    const int MC = clamp(M / num_threads, 1, M_CACHE_TILE_BF16); 
    const int NC = round_up(std::min(N, N_CACHE_TILE_BF16), NR); 

    const int max_micropanel_stride_B = round_up(KC * NR, ALIGNMENT / sizeof(bfloat16_t));
    const int n_micropanels_B = ceil_division(NC, NR);
    size_t bytes_B_packed = round_up(size_t(n_micropanels_B) * max_micropanel_stride_B * sizeof(bfloat16_t), ALIGNMENT);
    auto* B_packed = reinterpret_cast<bfloat16_t*>(aligned_alloc(ALIGNMENT, bytes_B_packed));

    #pragma omp parallel
    {
        const int max_micropanel_stride_A = round_up(KC * MR, ALIGNMENT / sizeof(bfloat16_t));
        const int n_micropanels_A = ceil_division(MC, MR);
        size_t bytes_A_packed = round_up(n_micropanels_A * max_micropanel_stride_A * sizeof(bfloat16_t), ALIGNMENT);
        auto* A_packed = reinterpret_cast<bfloat16_t*>(aligned_alloc(ALIGNMENT, bytes_A_packed));

        for (int j_cache = 0; j_cache * NC < N; ++j_cache) {
            int n_end = std::min(NC, N - j_cache * NC);

            for (int k_cache = 0; k_cache * KC < K; ++k_cache) {
                const int k_end = std::min(KC, K - k_cache * KC);
                const int k_pad = round_up(k_end, K_UNROLL_BF16);

                const int micropanel_stride_A = round_up(k_pad * MR, ALIGNMENT / sizeof(bfloat16_t));
                const int micropanel_stride_B = round_up(k_pad * NR, ALIGNMENT / sizeof(bfloat16_t));

                #pragma omp single 
                packB_avx512_bf16(
                    &B[(k_cache * KC) * N + (j_cache * NC)], B_packed, 
                    N, micropanel_stride_B, n_end, k_end, k_pad
                );

                int i_cache_end = ceil_division(M, MC);

                #pragma omp for schedule(static)
                for (int i_cache = 0; i_cache < i_cache_end; ++i_cache) {
                    int m_end = std::min(MC, M - i_cache * MC);

                    packA_avx512_bf16(
                        &A[(i_cache * MC) * K + (k_cache * KC)], A_packed, 
                        K, micropanel_stride_A, m_end, k_end, k_pad
                    );
            
                    int i_register = 0;
                    for (; (i_register + 1) * MR <= m_end; ++i_register) {
                        for (int j_register = 0; j_register * NR < n_end; ++j_register) {
                            int n_remain = std::min(NR, n_end - j_register * NR);
                            microkernel_16x16_avx512_bf16(
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
                            microkernel_cleanup_avx512_bf16(
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

inline void gemm_bf16_tiled(const Matrix& A_base, const Matrix& B_base, Matrix& C_base) {
    const auto& A = static_cast<const MatrixBF16&>(A_base);
    const auto& B = static_cast<const MatrixBF16&>(B_base);
    auto& C = static_cast<MatrixFP32&>(C_base);

    const int M = A.rows;
    const int K = A.cols;
    const int N = B.cols;

    const auto* A_ptr = reinterpret_cast<const bfloat16_t*>(A.raw_data);
    const auto* B_ptr = reinterpret_cast<const bfloat16_t*>(B.raw_data);
    auto* C_ptr = reinterpret_cast<float*>(C.raw_data);

    if (M > N) {
        return gemm_bf16_avx512_tiled_parallel_M(A_ptr, B_ptr, C_ptr, M, N, K);
    } else {
        return gemm_bf16_avx512_tiled_parallel_N(A_ptr, B_ptr, C_ptr, M, N, K);
    }
}

inline void gemm_bf16_naive(const Matrix& A_base, const Matrix& B_base, Matrix& C_base) {
    const auto& A = static_cast<const MatrixBF16&>(A_base);
    const auto& B = static_cast<const MatrixBF16&>(B_base);
    auto& C = static_cast<MatrixFP32&>(C_base);

    const bfloat16_t* A_ptr = reinterpret_cast<const bfloat16_t*>(A.raw_data);
    const bfloat16_t* B_ptr = reinterpret_cast<const bfloat16_t*>(B.raw_data);
    float* C_ptr = reinterpret_cast<float*>(C.raw_data);

    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; ++k) {
                sum += bf16_to_fp32(A_ptr[i * A.cols + k]) * bf16_to_fp32(B_ptr[k * B.cols + j]);
            }
            C_ptr[i * C.cols + j] = sum;
        }
    }
}

