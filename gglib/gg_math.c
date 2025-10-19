//
// Created by Guglielmo Grillo on 17/10/25.
//

#include "gg_math.h"

#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

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

    while(S>1) {
        U = 2.*rand()/ (double) RAND_MAX - 1.;
        V = 2.*rand()/ (double) RAND_MAX - 1.;
        S = U*U + V*V;
    }
    X = U*sqrt(-2.*log(S)/S);
    Y = V*sqrt(-2.*log(S)/S);

    available = true;
    return X;
}
