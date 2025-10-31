//
// Created by Guglielmo Grillo on 17/10/25.
//

#include <stdbool.h>
#include <stdlib.h>

#include "gg_math.h"

double randn(void) {
    // https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
    // I don't remember how to generate gaussian distributed number so, for now, I'll just trust
    // the Marsaglia polar method described on wikipedia
    static bool available = false;
    static double Y;

    if(available) {
        available = false;
        return Y;
    }
    double U, V, X;
    double S = 2;

    while(S>1 || S==0) {
        U = 2.*rand()/ (double) RAND_MAX - 1.;
        V = 2.*rand()/ (double) RAND_MAX - 1.;
        S = U*U + V*V;
    }
    X = U*sqrt(-2.*log(S)/S);
    Y = V*sqrt(-2.*log(S)/S);

    available = true;
    return X;
}


void computeAveVar(const double* v, size_t size, double* mean, double* var) {
    double x = 0;
    double x2 = 0;

    for (size_t i = 0; i < size; i++) {
        x  += v[i];
        x2 += v[i] * v[i];
    }
    *mean = x / (double) size;
    *var = (x2/ (double) size) - (*mean) * (*mean);
}

void computeAveVarf(const float* v, size_t size, double* mean, double* var) {
    double x = 0;
    double x2 = 0;
    double s;

    for (size_t i = 0; i < size; i++) {
        s = (double) v[i];
        x  += s / (double) size;
        x2 += s*s / (double) size;
    }
    *mean = x;// / (float) size;
    *var = x2 - (*mean) * (*mean);
    //*var = (x2/ (float) size) - (*mean) * (*mean);
}


void computePairwiseDistancesWithPCB(const double* coordinates, size_t n_particles, double L, double* r_utb) {

    for (size_t p1=0; p1<n_particles; p1++) {
        const double x1 = coordinates[3*p1];
        const double y1 = coordinates[3*p1+1];
        const double z1 = coordinates[3*p1+2];

        for (size_t p2=p1+1; p2<n_particles; p2++) {
            const double x2 = coordinates[3*p2];
            const double y2 = coordinates[3*p2+1];
            const double z2 = coordinates[3*p2+2];

            double dx = x1-x2;
            double dy = y1-y2;
            double dz = z1-z2;

            // PBC - Find minimal image
            dx -= L*round(dx/L);
            dy -= L*round(dy/L);
            dz -= L*round(dz/L);

            const size_t idx = UTIDX(p1, p2, n_particles);
            r_utb[idx]      = NORM(dx,dy,dz);
        }
    }
}