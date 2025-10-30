// physics.cuh
// Created by Guglielmo Grillo on 20/10/25.
//
#pragma once

constexpr float EPSILON = 10.22f; // Energy in [Kelvin]
constexpr float SIGMA = 2.556f;   // Distances in [Angstrom]

/**
 * @brief  Computes the Lennard-Jones Potential as a proxy for He-He interaction
 * @param r Distance between the two He atoms
 * @return the potential between the two atoms
 * @warning Assumes distances in [$\sigma$] and energy in [$4\epsilon$]
 */
__device__ __forceinline__ float LJ_potential(float const r) {
    const float ri = SIGMA / r;
    const float ri2 = ri * ri;
    const float ri6 = ri2 * ri2 * ri2;
    const float ri12 = ri6 * ri6;
    const float fourEpsilon = 4.f*EPSILON; // [Kelvin]

    return fourEpsilon*(ri12-ri6);
}

/**
 * @brief Trial Pair Wave Function proposed by Wu and Feenbarg (W. L. McMillan, Ground State of Liquid He⁴, Phys. Rev.
 * 138, A442 (1965) and ref therein)
 * @param r Distance between the two particles
 * @param a1 variational parameter 1. It's similar to a cut-off for the interaction
 * @param a2 variational parameter 2. It's similar to the sharpness of the cut-off
 * @return
 */
__device__ __forceinline__ float WU_FEENBERG_TPWF(const float r, const float a1, const float a2) {
    return expf(-powf(a1/r, a2) );
}

/**
 * @brief Logarithm of the function @fn WU_FEENBERG_TPWF
 * @param r Distance between the two particles
 * @param a1 variational parameter 1. It's similar to a cut-off for the interaction
 * @param a2 variational parameter 2. It's similar to the sharpness of the cut-off
 * @return
 */
__device__ __forceinline__ float logWU_FEENBERG_TPWF(const float r, const float a1, const float a2) {
    return -powf(a1/r, a2);
}

/**
 * @brief Derivative of WU_FEENBERG_TPWF computed analytically
 * @param r Distance between the two particles
 * @param a1 variational parameter 1. It's similar to a cut-off for the interaction
 * @param a2 variational parameter 2. It's similar to the sharpness of the cut-off
 * @return value of the derivative
 *
 * For comparison:
 * https://www.wolframalpha.com/input?i=derive+exp%28-%28a_1%2Fr%29%5Ea_2%29
 */
__device__ __forceinline__ float dWU_FEENBERG_TPWF(const float r, const float a1, const float a2) {
    const float t = powf(a1/r, a2);
    return (a2/r)*t*exp(-t);
}

/**
 * @brief Logarithm of the function @fn dWU_FEENBERG_TPWF
 * @param r Distance between the two particles
 * @param a1 variational parameter 1. It's similar to a cut-off for the interaction
 * @param a2 variational parameter 2. It's similar to the sharpness of the cut-off
 * @return value of the derivative
 *
 */
__device__ __forceinline__ float logdWU_FEENBERG_TPWF(const float r, const float a1, const float a2) {
    const float t = powf(a1/r, a2);
    return logf(a2/r*t)-t;
}

/**
 * @brief Second Derivative of WU_FEENBERG_TPWF computed analytically
 * @param r Distance between the two particles
 * @param a1 variational parameter 1. It's similar to a cut-off for the interaction
 * @param a2 variational parameter 2. It's similar to the sharpness of the cut-off
 * @return
 *
 * For comparison:
 * https://www.wolframalpha.com/input?i=second+derive+exp%28-%28a_1%2Fr%29%5Ea_2%29
 */
__device__ __forceinline__ float ddWU_FEENBERG_TPWF(const float r, const float a1, const float a2) {
    const float t = powf(a1/r, a2);
    return a2/(r*r) * expf(-t) * t*(a2*(t-1)-1);
}

/**
 * @brief Logarithm of the function @fn ddWU_FEENBERG_TPWF
 * @param r Distance between the two particles
 * @param a1 variational parameter 1. It's similar to a cut-off for the interaction
 * @param a2 variational parameter 2. It's similar to the sharpness of the cut-off
 * @return
 *
 * For comparison:
 * https://www.wolframalpha.com/input?i=second+derive+exp%28-%28a_1%2Fr%29%5Ea_2%29
 */
__device__ __forceinline__ float logddWU_FEENBERG_TPWF(const float r, const float a1, const float a2) {
    const float t = powf(a1/r, a2);
    return logf(a2/(r*r) * t*(a2*(t-1)-1) )-t;
}
