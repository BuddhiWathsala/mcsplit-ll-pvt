#include "simd.h"

#include <vector>
#include <algorithm>
#include <iostream>
// Platform-specific SIMD headers
#ifdef __x86_64__
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// Forward declaration of helper functions
namespace simd_sort {
    // NEON-specific helper functions for sorting
    void neon_sort_16_forward(int* arr);
    void neon_sort_16_reverse(int* arr);
}

// Forward sorting function for x86 AVX2
void simd_sort_16_elements_forward(int* arr) {
#ifdef __x86_64__
    // AVX2 implementation (previous implementation)
    __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr));
    __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + 8));

    // Comprehensive sorting network (ascending order)
    __m256i min_v01 = _mm256_min_epi32(v0, v1);
    __m256i max_v01 = _mm256_max_epi32(v0, v1);

    __m256i shuffle1_min = _mm256_shuffle_epi32(min_v01, _MM_SHUFFLE(1, 0, 3, 2));
    __m256i shuffle1_max = _mm256_shuffle_epi32(max_v01, _MM_SHUFFLE(1, 0, 3, 2));

    __m256i min1 = _mm256_min_epi32(min_v01, shuffle1_min);
    __m256i max1 = _mm256_max_epi32(max_v01, shuffle1_max);

    __m256i blend1 = _mm256_blend_epi32(min1, max1, 0b10101010);
    __m256i shuffle2 = _mm256_shuffle_epi32(blend1, _MM_SHUFFLE(2, 3, 0, 1));

    __m256i min2 = _mm256_min_epi32(blend1, shuffle2);
    __m256i max2 = _mm256_max_epi32(blend1, shuffle2);

    __m256i result1 = _mm256_blend_epi32(min2, max2, 0b11001100);

    __m256i shuffle3 = _mm256_shuffle_epi32(result1, _MM_SHUFFLE(0, 1, 2, 3));
    __m256i min3 = _mm256_min_epi32(result1, shuffle3);
    __m256i max3 = _mm256_max_epi32(result1, shuffle3);

    __m256i final_result = _mm256_blend_epi32(min3, max3, 0b11110000);

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr), final_result);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr + 8), final_result);

#elif defined(__ARM_NEON)
    // Use NEON implementation
    simd_sort::neon_sort_16_forward(arr);
#else
    // Fallback for non-SIMD platforms
    std::sort(arr, arr + 16);
#endif
}

// Reverse sorting function for x86 AVX2
void simd_sort_16_elements_reverse(int* arr) {
#ifdef __x86_64__
    // AVX2 implementation (previous implementation)
    __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr));
    __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arr + 8));

    // Comprehensive sorting network (descending order)
    __m256i max_v01 = _mm256_max_epi32(v0, v1);
    __m256i min_v01 = _mm256_min_epi32(v0, v1);

    __m256i shuffle1_max = _mm256_shuffle_epi32(max_v01, _MM_SHUFFLE(1, 0, 3, 2));
    __m256i shuffle1_min = _mm256_shuffle_epi32(min_v01, _MM_SHUFFLE(1, 0, 3, 2));

    __m256i max1 = _mm256_max_epi32(max_v01, shuffle1_max);
    __m256i min1 = _mm256_min_epi32(min_v01, shuffle1_min);

    __m256i blend1 = _mm256_blend_epi32(max1, min1, 0b10101010);
    __m256i shuffle2 = _mm256_shuffle_epi32(blend1, _MM_SHUFFLE(2, 3, 0, 1));

    __m256i max2 = _mm256_max_epi32(blend1, shuffle2);
    __m256i min2 = _mm256_min_epi32(blend1, shuffle2);

    __m256i result1 = _mm256_blend_epi32(max2, min2, 0b11001100);

    __m256i shuffle3 = _mm256_shuffle_epi32(result1, _MM_SHUFFLE(0, 1, 2, 3));
    __m256i max3 = _mm256_max_epi32(result1, shuffle3);
    __m256i min3 = _mm256_min_epi32(result1, shuffle3);

    __m256i final_result = _mm256_blend_epi32(max3, min3, 0b11110000);

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr), final_result);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr + 8), final_result);

#elif defined(__ARM_NEON)
    // Use NEON implementation
    simd_sort::neon_sort_16_reverse(arr);
#else
    // Fallback for non-SIMD platforms
    std::sort(arr, arr + 16, std::greater<int>());
#endif
}

// NEON-specific implementations
namespace simd_sort {
    // Forward sorting for NEON
    void neon_sort_16_forward(int* arr) {
#ifdef __ARM_NEON
        // Load 4 integers into 4 NEON registers
        int32x4_t v0 = vld1q_s32(arr);
        int32x4_t v1 = vld1q_s32(arr + 4);
        int32x4_t v2 = vld1q_s32(arr + 8);
        int32x4_t v3 = vld1q_s32(arr + 12);

        // Sorting network for 4-element chunks
        // First level comparisons
        int32x4_t min01 = vminq_s32(v0, v1);
        int32x4_t max01 = vmaxq_s32(v0, v1);
        int32x4_t min23 = vminq_s32(v2, v3);
        int32x4_t max23 = vmaxq_s32(v2, v3);

        // Cross-chunk comparisons
        int32x4_t min_cross = vminq_s32(min01, min23);
        int32x4_t max_cross = vmaxq_s32(max01, max23);

        // Store results in sorted order
        vst1q_s32(arr, min_cross);
        vst1q_s32(arr + 4, min01);
        vst1q_s32(arr + 8, min23);
        vst1q_s32(arr + 12, max_cross);
#endif
    }

    // Reverse sorting for NEON
    void neon_sort_16_reverse(int* arr) {
#ifdef __ARM_NEON
        // Load 4 integers into 4 NEON registers
        int32x4_t v0 = vld1q_s32(arr);
        int32x4_t v1 = vld1q_s32(arr + 4);
        int32x4_t v2 = vld1q_s32(arr + 8);
        int32x4_t v3 = vld1q_s32(arr + 12);

        // Sorting network for 4-element chunks (descending order)
        // First level comparisons
        int32x4_t max01 = vmaxq_s32(v0, v1);
        int32x4_t min01 = vminq_s32(v0, v1);
        int32x4_t max23 = vmaxq_s32(v2, v3);
        int32x4_t min23 = vminq_s32(v2, v3);

        // Cross-chunk comparisons
        int32x4_t max_cross = vmaxq_s32(max01, max23);
        int32x4_t min_cross = vminq_s32(min01, min23);

        // Store results in reverse sorted order
        vst1q_s32(arr, max_cross);
        vst1q_s32(arr + 4, max01);
        vst1q_s32(arr + 8, max23);
        vst1q_s32(arr + 12, min_cross);
#endif
    }
}

//// Benchmark function for forward sorting
//void benchmark_simd_sort_forward(std::vector<int> arr) {
//    if (arr.size() != 16) {
//        throw std::runtime_error("Input must be exactly 16 elements");
//    }
//
//    auto start = std::chrono::high_resolution_clock::now();
//    simd_sort_16_elements_forward(arr.data());
//    auto end = std::chrono::high_resolution_clock::now();
//
//    std::chrono::duration<double, std::milli> duration = end - start;
//    std::cout << "SIMD Forward Sort Time: " << duration.count() << " ms\n";
//}
//
//// Benchmark function for forward sorting
//void benchmark_sort_forward(std::vector<int> arr) {
//    if (arr.size() != 16) {
//        throw std::runtime_error("Input must be exactly 16 elements");
//    }
//
//    auto start = std::chrono::high_resolution_clock::now();
//    std::sort(arr.begin(), arr.begin()+1);
//    auto end = std::chrono::high_resolution_clock::now();
//
//    std::chrono::duration<double, std::milli> duration = end - start;
//    std::cout << "Forward Sort Time: " << duration.count() << " ms\n";
//}