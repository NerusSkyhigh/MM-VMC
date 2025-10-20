//
// Created by Guglielmo Grillo on 20/10/25.
//
#pragma once

double LJ_potential(double const r);

/**
 * @brief Trial Pair Wave Function proposed by Wu and Feenbarg (W. L. McMillan, Ground State of Liquid He⁴, Phys. Rev.
 * 138, A442 (1965) and ref therein)
 * @param r Distance between the two particles
 * @param a1 variational parameter 1. It's similar to a cut-off for the interaction
 * @param a2 variational parameter 2. It's similar to the sharpness of the cut-off
 * @return
 */
inline double WU_FEENBERG_TPWF(const double r, const double a1, const double a2);

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
inline double dWU_FEENBERG_TPWF(double r, double a1, double a2);

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
inline double ddWU_FEENBERG_TPWF(double r, double a1, double a2);