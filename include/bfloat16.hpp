#pragma once

#include <iostream>

using bfloat16_t = uint16_t;

inline bfloat16_t fp32_to_bf16(float f) {
    uint32_t input_bits = *reinterpret_cast<uint32_t*>(&f);
    uint32_t least_significant_bit = (input_bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + least_significant_bit;
    input_bits += rounding_bias;
    return static_cast<bfloat16_t>(input_bits >> 16);
}

inline float bf16_to_fp32(bfloat16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    return *reinterpret_cast<float*>(&bits);
}