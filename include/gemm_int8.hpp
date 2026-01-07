#pragma once

#include <immintrin.h>

#include "matrix.hpp"
#include "utils.hpp"

// Int8 GEMM tiling sizes
constexpr int M_CACHE_TILE_INT8 = 768; 
constexpr int N_CACHE_TILE_INT8 = 128;
constexpr int K_CACHE_TILE_INT8 = 256;
constexpr int M_REGISTER_TILE_INT8 = 8;
constexpr int N_REGISTER_TILE_INT8 = 8;

/*  Micro kernel for bf16 GEMM.
    
    We will write to a [MR x n_remain] tile of C.

    We assume the following: 
    - MR == NR == 8.
    - B_packed is expected to be zero padded to a size divisible by NR = 8, to allow seamless avx2 loading from it.
    - The n_remain dimension can be any int from 1 to 8 inclusive, whereas MR=8 is fixed
*/
inline void microkernel_8x8_avx2_int8(
    const int8_t* __restrict__ A, 
    const int8_t* __restrict__ B, 
    int32_t* __restrict__ C, 
    int C_stride, int n_remain, int k_pad,
    bool first_k_block
) {
    constexpr int MR = M_REGISTER_TILE_INT8;
    constexpr int NR = N_REGISTER_TILE_INT8;

    static_assert(MR == 8, "microkernel_8x8_avx2_int8 assumes MR == 8");
    static_assert(NR == 8, "microkernel_8x8_avx2_int8 assumes NR == 8");

    // MR=8 output accumulators, each one containing NR=8 fp32 scalars. We will use only the first m_remain ones.
    __m256i C_acc[MR];
    if (first_k_block) {
        // If we are at the first accumulation iteration, init to zero the local accumulator 
        #pragma unroll(MR)
        for (int i = 0; i < MR; ++i) {
            C_acc[i] = _mm256_setzero_si256();
        }
    } else {
        // Otherwise we load from memory the partial sums
        if (n_remain == NR) {
            // When possible we load to the avx2 registers directly
            #pragma unroll(MR)
            for (int i = 0; i < MR; ++i) {
                C_acc[i] = _mm256_loadu_epi32(&C[i * C_stride]);
            }
        } else {
            // Else we repeatedly load n_remain scalars into tmp, and then load with avx2 form tmp
            int32_t tmp[NR];
            memset(tmp, 0, NR * sizeof(float));

            #pragma unroll(MR)
            for (int i = 0; i < MR; ++i) {
                for (int j = 0; j < n_remain; ++j) {
                    tmp[j] = C[i * C_stride + j];
                }
                C_acc[i] = _mm256_loadu_epi32(tmp);
            }
        }
    }

    // Hot loop
    #pragma unroll(K_UNROLL_BF16)
    for (int k = 0; k < k_pad; ++k) {
        // Load from B 8 x 16bit scalars (bf16)
        __m128i v_b_bf16 = _mm_load_si128(reinterpret_cast<const __m128i*>(&B[k * NR]));

        // Convert from bf16 to fp32:
        // 1. We upcast each 16 bit lane to 32 bits, by extending on the left with zeros
        __m256i v_b_extended = _mm256_cvtepu16_epi32(v_b_bf16);
        // 2. We shift left each 32bit lane by 16, so the original 16bits are on the left (and we have zeros on the right)
        __m256 v_b = _mm256_castsi256_ps(_mm256_slli_epi32(v_b_extended, 16));

        #pragma unroll(MR)
        for (int i = 0; i < MR; ++i) {
            // Convert from bf16 to fp32:
            // 1. Read from A a 16bit scalar 0xABCD and broadcast it to all the 16 lanes of width 16bits
            __m256i v_a_broadcast = _mm256_set1_epi16(A[k * MR + i]);
            // 2. Shift left by 16bits all the 32bit lanes (each lane goes from 0xABCDABCD to 0xABCD0000), and reinterpret as float
            __m256 v_a = _mm256_castsi256_ps(_mm256_slli_epi32(v_a_broadcast, 16));

            // Fused multiply add: C_ij = A_j * B_j + C_ij
            // C_acc[i] = _mm256_fmadd_(v_a, v_b, C_acc[i]);
        }
    }

    // Write back to C
    if (n_remain == NR) {
        // When possible write result to the [m_remain x 8] tile of C, with avx2
        #pragma unroll(MR)
        for (int i = 0; i < MR; ++i) {
            _mm256_storeu_epi32(&C[i * C_stride], C_acc[i]);
        }     
    } else {
        // Else we write with avx2 to tmp, and from tmp store only n_remain values
        float tmp[NR];
        for (int i = 0; i < MR; ++i) {
            _mm256_storeu_epi32(tmp, C_acc[i]);
            for (int j = 0; j < n_remain; ++j) {
                C[i * C_stride + j] = tmp[j];
            }
        }
    }
}

/*  Cleanup micro kernel for bf16 GEMM.
    
    We will write to a [m_remain x n_remain] tile of C.
    B_packed is expected to be zero padded to a size divisible by NR = 8, to allow seamless avx2 loading from it.
    The m_remain and n_remain dimensions can be any int from 1 to 8 inclusive.

    We assume MR == NR == 8
*/
inline void microkernel_cleanup_avx2_int8(
    const int8_t* __restrict__ A, 
    const int8_t* __restrict__ B, 
    int32_t* __restrict__ C, 
    int C_stride, int m_remain, int n_remain, int k_pad,
    bool first_k_block
) {
    // TODO
}

/*  Read a 8x8 block of bfloat16_t (uint16_t) from src matrix, transpose it with avx2 and write it to dst matrix.

    The transpose is performed as follows:
    - At first we have 8 vector registers, each one containing a row of the matrix (8 scalars)
    - We want to end up with 8 vector registers containing the columns of the matrix

    Conceptually, to achieve this we "split" and "merge" recursively:
    - Start from 8 [1 x 8] blocks (these are the initial rows, each one stored in a register)
    - Stage 1: Split in 2 the blocks horizontally and merge pairs vertically into 2 new [2 x 4] "column major" blocks (each one stored in a register)
    - Stage 2: Split in 2 the blocks horizontally and merge pairs vertically into 2 new [4 x 2] "column major" blocks (each one stored in a register)
    - Stage 3: Split in 2 the blocks horizontally and merge pairs vertically into 2 new [8 x 1] "column major" blocks 
        (these are the final columns, each one stored in a register)

    Each "horizontal split" + "vertical merge" is done by unpacking and interleaving the columns of 
    the blocks stored in the registers. In practice, this means that:
    - Stage 1: we unpack and interleave groups of 16 bits ("columns" of 1 scalar)
    - Stage 2: groups of 32 bits (columns of 2 scalars) 
    - Stage 3: groups of 64 bits (columns of 4 scalars) 

    In this way we keep a "column major" layout at each step. This leads at the end to the columns 
    being stored contiguously in 8 registers, which we can finally write directly to dst.
*/
inline void transpose_8x8_avx2_bf16(
    const bfloat16_t* __restrict__ src, 
    bfloat16_t* __restrict__ dst, 
    int src_stride, 
    int dst_stride
) {
    // Here we declare 12 avx2 registers (out of 16)
    // We will explicitly reuse these registers (to avoid spilling)
    __m128i a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3; 

    // Load 8x8 block from src to avx2 registers (8 avx2 registers, each one containing 8 x bf16)
    a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 0]));
    a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 1]));
    a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 2]));
    a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 3]));
    a4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 4]));
    a5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 5]));
    a6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 6]));
    a7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[src_stride * 7]));

    // Stage 1: unpack and interleave groups of 16bits (1 scalar)
    b0 = _mm_unpacklo_epi16(a0, a1);
    b1 = _mm_unpackhi_epi16(a0, a1);
    b2 = _mm_unpacklo_epi16(a2, a3);
    b3 = _mm_unpackhi_epi16(a2, a3);
    a0 = _mm_unpacklo_epi16(a4, a5); // Here we start reusing the free registers (from now on we keep "cycling" them, to avoid spilling)
    a1 = _mm_unpackhi_epi16(a4, a5);
    a2 = _mm_unpacklo_epi16(a6, a7);
    a3 = _mm_unpackhi_epi16(a6, a7);

    // Stage 2: unpack and interleave groups of 32bits (2 scalars)
    a4 = _mm_unpacklo_epi32(b0, b2);
    a5 = _mm_unpackhi_epi32(b0, b2);
    a6 = _mm_unpacklo_epi32(b1, b3);
    a7 = _mm_unpackhi_epi32(b1, b3);
    b0 = _mm_unpacklo_epi32(a0, a2);
    b1 = _mm_unpackhi_epi32(a0, a2);
    b2 = _mm_unpacklo_epi32(a1, a3);
    b3 = _mm_unpackhi_epi32(a1, a3);

    // Stage 3: unpack and interleave groups of 64bits (4 scalars)
    a0 = _mm_unpacklo_epi64(a4, b0);
    a1 = _mm_unpackhi_epi64(a4, b0); 
    a2 = _mm_unpacklo_epi64(a5, b1);
    a3 = _mm_unpackhi_epi64(a5, b1);
    a4 = _mm_unpacklo_epi64(a6, b2);
    a5 = _mm_unpackhi_epi64(a6, b2);
    b0 = _mm_unpacklo_epi64(a7, b3);
    b1 = _mm_unpackhi_epi64(a7, b3);

    // We store the result to the 8x8 block of dst
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 0]), a0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 1]), a1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 2]), a2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 3]), a3);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 4]), a4);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 5]), a5);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 6]), b0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[dst_stride * 7]), b1);
}

/*  Pack a cache level tile of A into the A_packed buffer using a layout optimized for the microkernel.

    We employ the following nested layout so the data can be read contiguously by the microkernel: 
    - A_packed contains ceil(m_end / MR) micropanels (i_register loop outside the microkernel)
    - each micropanel contains k_pad columns (k loop in the microkernel)
    - each column contains m_remain values (i loop in the microkernel); m_remain is always MR for all the columns, 
        except for the columns of the last micropanel, where m_remain can be any int from 1 to MR

    The data of A_packed is accessed as follows: 
        A_packed[i_register * micropanel_stride + k * m_remain + i]

    As said before, m_remain == MR in general, except possibly for the last micropanel.

    We assume:
    - MR == M_REGISTER_TILE_BF16 == 8
    - sizeof(bfloat16_t) == 2 bytes
    - A_packed is allocated with aligned_alloc(ALIGNMENT, ...)
    - A_packed has at least k_pad * m_end * sizeof(bfloat16_t) bytes of allocated memory
    - micropanel_stride is a multiple of MR and of ALIGNMENT / sizeof(bfloat16_t)
*/
inline void packA_avx2_bf16(
    const bfloat16_t *__restrict__ A_tile, /* Pointer to the start of the B cache level tile */
    bfloat16_t *__restrict__ A_packed,     /* Destination buffer */
    int A_stride, /* Offset between the start of two rows of A_tile */
    int micropanel_stride, /* Offset between the start of two consecutive micropanels in A_packed */
    int m_end, int k_end, int k_pad
) {
    constexpr int MR = M_REGISTER_TILE_BF16;
    static_assert(MR == 8, "packA_avx2_bf16 assumes MR == 8");

    int i_register = 0;

    // Main packing loop (with avx2, m_remain == MR == 8)
    for (; (i_register + 1) * MR <= m_end; ++i_register) {
        int k = 0;

        // First we iterate through MRxMR (8x8) blocks of A_tile, transpose them with avx2 and write them to A_packed
        for (; k + MR <= k_end; k += MR) {
            const auto* src = &A_tile[(i_register * MR) * A_stride + k];
            auto* dst = &A_packed[i_register * micropanel_stride + k * MR];
            transpose_8x8_avx2_bf16(src, dst, A_stride, MR); // Here the dst matrix is a MRxMR block inside a micropanel
        }

        // Cleanup k (no avx)
        for (; k < k_end; ++k) {
            const auto* src = &A_tile[(i_register * MR) * A_stride + k];
            auto* dst = &A_packed[i_register * micropanel_stride + k * MR];
            for (int i = 0; i < MR; ++i) {
                dst[i] = src[i * A_stride];
            }
        }

        // Zero pad along k
        for (; k < k_pad; ++k) {
            auto* dst = &A_packed[i_register * micropanel_stride + k * MR];
            _mm_store_si128(reinterpret_cast<__m128i*>(dst), _mm_setzero_si128());
        }
    }

    // Cleanup i (no avx, m_remain < 8)
    int m_remain = m_end - i_register * MR;
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

/*  Pack a cache level tile of B into the B_packed buffer using a layout optimized for the microkernel.

    We employ the following nested layout so the data can be read contiguously by the microkernel: 
    - B_packed contains ceil(n_end / NR) micropanels (j_register loop outside the microkernel)
    - each micropanel contains k_pad rows (k loop in the microkernel)
    - each row contains NR values (avx2 vectorization in the microkernel)

    The memory layout of B_packed is accessed as follows:
        B_packed[j_register * micropanel_stride + k * NR + j]

    We will employ zero padding to ensure that even the rows of the last micropanel contain exactly NR values.

    We assume:
    - NR == N_REGISTER_TILE_BF16 == 8
    - sizeof(bfloat16_t) == 2 bytes
    - B_packed is allocated with aligned_alloc(ALIGNMENT, ...)
    - B_packed has at least k_pad * round_up(n_end, NR) * sizeof(bfloat16_t) bytes of allocated memory
    - micropanel_stride is a multiple of NR and of ALIGNMENT / sizeof(bfloat16_t)
*/
inline void packB_avx2_int8(
    const bfloat16_t *__restrict__ B_tile, /* Pointer to the start of the B cache level tile */
    bfloat16_t *__restrict__ B_packed,     /* Destination buffer */
    int B_stride, /* Offset between the start of two consecutive rows in B_tile */
    int micropanel_stride, /* Offset between the start of two consecutive micropanels in B_packed */
    int n_end, int k_end, int k_pad
) {
    constexpr int NR = N_REGISTER_TILE_BF16;
    static_assert(NR == 8, "packB_avx2_bf16 assumes NR == 8");

    // Main packing loop (SIMD)
    int j_register = 0; 
    for (; (j_register + 1) * NR <= n_end; ++j_register) {
        int k = 0;
        for (; k < k_end; ++k) {
            const auto* src = &B_tile[k * B_stride + (j_register * NR)];
            auto* dst = &B_packed[j_register * micropanel_stride + k * NR];
            // Load from src 8 x 16bit scalars, and then store them to dst
            __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
            _mm_store_si128(reinterpret_cast<__m128i*>(dst), tmp);
        }

        // Pad to zero for all remaining k
        for (; k < k_pad; ++k) {
            auto* dst = &B_packed[j_register * micropanel_stride + k * NR];
            _mm_store_si128(reinterpret_cast<__m128i*>(dst), _mm_setzero_si128());
        }
    }

    // Now we handle the remaining j up to n_end (this can be viewed as a last step in the j_register loop).
    // In other words, we are handling the last micropanel if there are some remaining j
    int n_remain = n_end - j_register * NR;
    if (n_remain == 0) return; 

    // We set directly to zero the whole last micropanel to handle zero-padding.
    // We can set to zero with simd NR > n_remain values since we assumed to own enough memory.
    for (int k = 0; k < k_pad; ++k) {
        auto* dst = &B_packed[j_register * micropanel_stride + k * NR];
        _mm_store_si128(reinterpret_cast<__m128i*>(dst), _mm_setzero_si128());
    }

    // Read the remaining scalars from src and write them to dst
    for (int k = 0; k < k_end; ++k) {
        for (int j = 0; j < n_remain; ++j) {
            auto src_val = B_tile[k * B_stride + (j_register * NR + j)];
            B_packed[j_register * micropanel_stride + k * NR + j] = src_val;
        }
    }
}

// TODO: fix gemm_int8_avx2_tiled_parallel_N
inline void gemm_int8_avx2_tiled_parallel_N(
    const int8_t* __restrict__ A, 
    const int8_t* __restrict__ B, 
    int32_t* __restrict__ C, 
    const int M, const int N, const int K
) {
    // const int num_threads = omp_get_max_threads();
    // const int MR = M_REGISTER_TILE_BF16;
    // const int NR = N_REGISTER_TILE_BF16;
    
    // static_assert(NR == 8, "gemm_bf16_avx2_tiled_parallel_N assumes NR == 8");
    // static_assert(MR == 8, "gemm_bf16_avx2_tiled_parallel_N assumes MR == 8");
    // assert(N > 0 && "N should be > 0");
    // assert(M > 0 && "M should be > 0");
    // assert(K > 0 && "K should be > 0");

    // const int KC = round_up(std::min(K_CACHE_TILE_BF16, K), K_UNROLL_BF16); // We enforce KC to be a nonzero multiple of K_UNROLL_BF16
    // const int MC = std::min(M_CACHE_TILE_BF16, M); // Note that we handle also MC < MR
    // const int NC = round_up(std::clamp(N / num_threads, NR, N_CACHE_TILE_BF16), NR); // We enforce NC to be a nonzero multiple of NR

    // #pragma omp parallel
    // {
    //     // Allocate A_packed, private per thread (ideally stored in L2/L1 cache)
    //     const int max_micropanel_stride_A = round_up(KC * MR, ALIGNMENT / sizeof(int8_t));
    //     const int n_micropanels_A = ceil_division(MC, MR);
    //     size_t elements_A_packed = size_t(n_micropanels_A) * max_micropanel_stride_A;
    //     size_t bytes_A_packed = round_up(n_micropanels_A * max_micropanel_stride_A * sizeof(int8_t), ALIGNMENT);
    //     auto* A_packed = reinterpret_cast<int8_t*>(std::aligned_alloc(ALIGNMENT, bytes_A_packed));

    //     // Allocate B_packed, private per thread (stored in L1/L2/L3 cache depending on N and on num_threads)
    //     const int max_micropanel_stride_B = round_up(KC * NR, ALIGNMENT / sizeof(int8_t));
    //     const int n_micropanels_B = ceil_division(NC, NR);
    //     size_t elements_B_packed = size_t(n_micropanels_B) * max_micropanel_stride_B;
    //     size_t bytes_B_packed = round_up(elements_B_packed * sizeof(int8_t), ALIGNMENT);
    //     auto* B_packed = reinterpret_cast<int8_t*>(std::aligned_alloc(ALIGNMENT, bytes_B_packed));

    //     // (j_cache < j_cache_end) here is equivalent to (j_cache * NC < N)
    //     int j_cache_end = ceil_division(N, NC);

    //     // Cache level tiling 
    //     // j_cache * NC, k_cache * KC and i_cache * MC are "the starting indicies of cache level blocks"
    //     #pragma omp for schedule(static)
    //     for (int j_cache = 0; j_cache < j_cache_end; ++j_cache) {
    //         int n_end = std::min(NC, N - j_cache * NC);

    //         for (int k_cache = 0; k_cache * KC < K; ++k_cache) {
    //             // When packing A and B, we will pad along the k dim with zeros up to k_pad.
    //             // This ensures k_pad is divisible by the unrolling depth.
    //             const int k_end = std::min(KC, K - k_cache * KC);
    //             const int k_pad = round_up(k_end, K_UNROLL_BF16);

    //             // Stride between the start of two consecutive micropanels inside A_packed.
    //             // We round it up to ensure alignment.
    //             // See packA_avx2_bf16 for the A_packed layout explanation.
    //             const int micropanel_stride_A = round_up(k_pad * MR, ALIGNMENT / sizeof(bfloat16_t));

    //             // Stride between the start of two consecutive micropanels inside B_packed.
    //             // We round it up to ensure alignment.
    //             // See packB_avx2_bf16() for the B_packed layout explanation.
    //             const int micropanel_stride_B = round_up(k_pad * NR, ALIGNMENT / sizeof(bfloat16_t));

    //             // Each thread packs a cache level tile of B into a B_packed private buffer
    //             packB_avx2_int8(
    //                 &B[(k_cache * KC) * N + (j_cache * NC)], B_packed, 
    //                 N, micropanel_stride_B, n_end, k_end, k_pad
    //             );

    //             // (i_cache < i_cache_end) here is equivalent to (i_cache * MC < M)
    //             int i_cache_end = ceil_division(M, MC);
    //             for (int i_cache = 0; i_cache < i_cache_end; ++i_cache) {
    //                 int m_end = std::min(MC, M - i_cache * MC);

    //                 // Each thread packs a cache level tile of A into a A_packed private buffer
    //                 packA_avx2_bf16(
    //                     &A[(i_cache * MC) * K + (k_cache * KC)], A_packed, 
    //                     K, micropanel_stride_A, m_end, k_end, k_pad
    //                 );
            
    //                 // Register level tiling
    //                 // i_register * MR and j_register * NR are "the starting offsets of register level blocks inside the cache level blocks"
    //                 int i_register = 0;
    //                 for (; (i_register + 1) * MR <= m_end; ++i_register) {
    //                     for (int j_register = 0; j_register * NR < n_end; ++j_register) {
    //                         int n_remain = std::min(NR, n_end - j_register * NR);
    //                         microkernel_8x8_avx2_bf16(
    //                             &A_packed[i_register * micropanel_stride_A],
    //                             &B_packed[j_register * micropanel_stride_B],
    //                             &C[(i_cache * MC + i_register * MR) * N + (j_cache * NC + j_register * NR)],
    //                             N, n_remain, k_pad,
    //                             (k_cache == 0)
    //                         );
    //                     }
    //                 }
                    
    //                 // Cleanup in case m_end was not divisible by MR
    //                 int m_remain = std::min(MR, m_end - i_register * MR);
    //                 if (m_remain != 0) {
    //                     for (int j_register = 0; j_register * NR < n_end; ++j_register) {
    //                         int n_remain = std::min(NR, n_end - j_register * NR);
    //                         microkernel_cleanup_avx2_bf16(
    //                             &A_packed[i_register * micropanel_stride_A],
    //                             &B_packed[j_register * micropanel_stride_B],
    //                             &C[(i_cache * MC + i_register * MR) * N + (j_cache * NC + j_register * NR)],
    //                             N, m_remain, n_remain, k_pad,
    //                             (k_cache == 0)
    //                         );
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     std::free(A_packed);
    //     std::free(B_packed);
    // }
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

    // if (M > N) {
    //     return gemm_bf16_avx2_tiled_parallel_M(A_ptr, B_ptr, C_ptr, M, N, K);
    // } else {
        return gemm_int8_avx2_tiled_parallel_N(A_ptr, B_ptr, C_ptr, M, N, K);
    // }
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
