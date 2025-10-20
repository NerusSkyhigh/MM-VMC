//
// Created by Guglielmo Grillo on 20/10/25.
//
#include <math.h>

#include "physics.h"

/**
 * @brief  Computes the `Aziz & Slaman (1991) compromise potential (LM2M2)` for He-He interaction
 * @param r Distance between the two He atoms
 * @return the potential between the two atoms
 * @warning This potential has a discontinuous derivative
 */
double he_he_potential(const double r) {
    // The potential is negligible at that point
    if(r>14) return 0;

    // [TODO] Find the parameters via VMC
    const double A = 1.89635353e5;
    const double a = 10.70203539;
    // Already scaled to work in rm units
    // const double rm = 2.9695;    /* Å */
    const double C6 = 1.34687065;
    const double C8 = 0.41308398;
    const double C10 = 0.17060159;
    const double beta = -1.90740649;

    const double D = 1.4088;
    const double eps = 10.97;    /* K */


    /* damping functions */
    // [TODO] Check if they are all really the same
    const double F6  = (r < 1.0) ? exp(-pow(D / r - 1.0, 2.0)) : 1.0;
    const double F8  = F6;
    const double F10 = F6;
    //double F12 = F6;

    /* reduced potential V*(x) */
    //[TODO] Store r2i (the inverse) so save time on the division?
    const double r2 = r * r;      // tmp
    double r4 = r2 * r2;    //tmp
    double r6 = r4 * r2;
    double r8 = r4 * r4;
    double r10 = r6 * r4;
    //double r12 = r6 * r6;
    double Vstar = A * exp(-a * r + beta * r2)
                 - F6  * C6  / r6
                 - F8  * C8  / r8
                 - F10 * C10 / r10;
    //- F12 * 0.0 / r12;  /* C12 = 0 for this version */

    return eps * Vstar; // [TODO] Work in eps units
}

/**
 * @brief  Computes the Lennard-Jones Potential as a proxy for He-He interaction
 * @param r Distance between the two He atoms
 * @return the potential between the two atoms
 * @warning Assumes distances in [$\sigma$] and energy in [$4\epsilon$]
 */
double LJ_potential(double const r) {
    const double ri = 1. / r;
    const double ri6 = (ri * ri * ri) * (ri * ri * ri);
    const double ri12 = ri6 * ri6;

    return ri12-ri6;
}

inline double WU_FEENBERG_TPWF(const double r, const double a1, const double a2) {
    return exp(-pow(a1/r, a2) );
}


inline double dWU_FEENBERG_TPWF(double r, double a1, double a2) {
    const double t = pow(a1/r, a2);
    return (a2/r)*t*exp(t);
}


inline double ddWU_FEENBERG_TPWF(double r, double a1, double a2) {
    const double t = pow(a1/r, a2);
    return a2/(r*r) * exp(-t) * t*(a2*(t-1)-1);
}


