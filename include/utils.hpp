#pragma once

// Round up a to the nearest multiple of b (a itself included)
constexpr inline int round_up(int a, int b) {
    return ((a + b - 1) / b) * b;
}

// Return a / b, rounded up to the nearest int
constexpr inline int ceil_division(int a, int b) {
    return (a + b - 1) / b;
}
