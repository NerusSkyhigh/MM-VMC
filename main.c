#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "gglib/gg_math.h"
#include "gglib/gg_mem.h"

typedef struct {
    int64_t Ntw;    // Number of Target Walkers
    int64_t Ncw;    // Number of Current Walkers
    int64_t MNw;    // Maximum Number of Walkers in the buffer

    int64_t Np;     // Number of particles
    double Et;      // Trial Energy

    int64_t Nts;    // total Number of TimeSteps
    double dt;      // Timestep size

    double* wa;             // Walker Arena for coordinates [MAXIMUM N WALKER * 3*N PARTICLES]
    DoubleBufferArena* Vb;   // potential (V) arena buffer [2*]

} DMCdata;


/**
 * @brief  Computes the `Aziz & Slaman (1991) compromise potential (LM2M2)` for He-He interaction
 * @param r Distance between the two He atoms
 * @return the potential between the two atoms
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
    double F6  = (r < 1.0) ? exp(-pow(D / r - 1.0, 2.0)) : 1.0;
    double F8  = F6;
    double F10 = F6;
    //double F12 = F6;

    /* reduced potential V*(x) */
    //[TODO] Store r2i (the inverse) so save time on the division?
    double r2 = r * r;      // tmp
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


void initUniform(const DMCdata* dmcd) {
    // [TODO] Use the VMC esteem to compute the initial state
    for(int i=0; i<dmcd->Ncw*3*dmcd->Np; i++) {
        dmcd->wa[i] = rand()/(double)RAND_MAX; // [TODO] Choose a box size
    }
}

void diffuse(const DMCdata* dmcd) {
    for(int i=0; i<dmcd->Ncw*3*dmcd->Np; i++) {
        // [TODO] Implement importance sampling
        dmcd->wa[i] +=randn()*dmcd->dt;
    }
}

void computeEnergy(const DMCdata* dmcd) {
    for(int walker=0; walker<dmcd->Ncw; walker++) {
        double V = 0;
        for(int p1=0; p1<dmcd->Np; p1++) {
            int64_t idx1 = walker*(3*dmcd->Np) + 3*p1;
            double x1 = dmcd->wa[idx1];
            double y1 = dmcd->wa[idx1+1];
            double z1 = dmcd->wa[idx1+2];

            for(int p2=p1+1; p2<dmcd->Np; p2++) {
                int64_t idx2 = walker*dmcd->Np + p2;
                double x2 = dmcd->wa[idx2];
                double y2 = dmcd->wa[idx2+1];
                double z2 = dmcd->wa[idx2+2];

                double r = NORM(x1-x2, y1-y2, z1-z2);
                V += he_he_potential(r);
            }
        }
        // [TODO] Consider if it's worth clearing the buffer each time. This code doesn't need it
        dmcd->Vb->buffers[dmcd->Vb->new][walker] = V;
    }

}

void branch(DMCdata* dmcd) {
    // [TODO]
    dmcd->Nts+1;
}



int main(void) {
    /* 1. Allocate resources
     * MEMORY USAGE:
     *  Each particle has 3 coordinates stored in a double (8 bytes):
     *  size = 8 bytes/coordinate * 3 coordinates/particle * 1_000 particles/walkers * 10_000 walkers = 240 MB.
     */
    DMCdata* dmcd = malloc(sizeof(DMCdata));
    dmcd->Ntw = 10000;//10000; // 10_000 walkers
    dmcd->Ncw = dmcd->Ntw;
    dmcd->MNw = 10*dmcd->Ncw; // Let's use a 10x buffer
    dmcd->Np = 100; //1000; // 1_000 Helium atoms

    dmcd->Nts = 1000000; // 1_000_000 timesteps
    dmcd->dt = 1.; // Delta timestep
    printf("Resources loaded\n");

    // 2. Generate walkers
    // [TODO] Check if a double buffer is needed
    dmcd->wa = malloc(sizeof(double)* (3*dmcd->Np)*dmcd->MNw);
    initUniform(dmcd);
    printf("Walkers initialized\n");

    // 2.2 Compute energy for the initial configuration
    dmcd->Vb = malloc(sizeof(DoubleBufferArena));
    initDoubleBufferArena(dmcd->Vb, dmcd->MNw);
    printf("Double buffer initialized\n");

    computeEnergy(dmcd);
    DOUBLE_BUFFER_NEXT_STEP(dmcd->Vb);
    printf("Initial energy computed\n");

    // 3. Main loop of the DMC
    for(int t=0; t<dmcd->Nts; t++) {
        diffuse(dmcd);
        computeEnergy(dmcd);
        branch(dmcd);

        DOUBLE_BUFFER_NEXT_STEP(dmcd->Vb);
        printf("Iteration %6d/%6lld\n", t, dmcd->Nts);
        fflush(stdout);
    }

    return 0;
}
