#pragma once

// Round up a to the nearest multiple of b (a itself included)
constexpr inline int round_up(int a, int b) {
    return ((a + b - 1) / b) * b;
}

// Return a / b, rounded up to the nearest int
constexpr inline int ceil_division(int a, int b) {
    return (a + b - 1) / b;
}

// Return val clamped between lo and hi
constexpr inline int clamp(int val, int lo, int hi) {
    if (val < lo) {
        return lo; 
    } else if (val > hi) {
        return hi;
    } else {
        return val;
    }
}
