#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "gglib/gg_math.h"
#include "gglib/gg_mem.h"
#include "physics.h"

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

} VMCdata;



void initUniform(const VMCdata* vmcd) {
    // [TODO] Use the VMC esteem to compute the initial state
    for(int i=0; i<vmcd->Ncw*3*vmcd->Np; i++) {
        vmcd->wa[i] = rand()/(double)RAND_MAX; // [TODO] Choose a box size
    }
}

void diffuse(const VMCdata* dmcd) {
    for(int i=0; i<dmcd->Ncw*3*dmcd->Np; i++) {
        // [TODO] Implement importance sampling
        dmcd->wa[i] +=randn()*dmcd->dt;
    }
}

void computeEnergy(const VMCdata* vmcd) {
    for(int walker=0; walker<vmcd->Ncw; walker++) {
        double V = 0;
        for(int p1=0; p1<vmcd->Np; p1++) {
            int64_t idx1 = walker*(3*vmcd->Np) + 3*p1;
            double x1 = vmcd->wa[idx1];
            double y1 = vmcd->wa[idx1+1];
            double z1 = vmcd->wa[idx1+2];

            for(int p2=p1+1; p2<vmcd->Np; p2++) {
                int64_t idx2 = walker*vmcd->Np + p2;
                double x2 = vmcd->wa[idx2];
                double y2 = vmcd->wa[idx2+1];
                double z2 = vmcd->wa[idx2+2];

                double r = NORM(x1-x2, y1-y2, z1-z2);
                //V += he_he_potential(r);
                V += LJ_potential(r);
            }
        }
        // [TODO] Consider if it's worth clearing the buffer each time. This code doesn't need it
        vmcd->Vb->buffers[vmcd->Vb->new][walker] = V;
    }

}

void branch(VMCdata* dmcd) {
    // [TODO]
    dmcd->Nts+1;
}



int main(void) {
    /* 1. Allocate resources
     * MEMORY USAGE:
     *  Each particle has 3 coordinates stored in a double (8 bytes):
     *  size = 8 bytes/coordinate * 3 coordinates/particle * 1_000 particles/walkers * 10_000 walkers = 240 MB.
     */
    VMCdata* dmcd = malloc(sizeof(VMCdata));
    dmcd->Ntw = 10000;//10000; // 10_000 walkers
    dmcd->Ncw = dmcd->Ntw;
    dmcd->MNw = 10*dmcd->Ncw; // Let's use a 10x buffer
    dmcd->Np = 200; //1000; // 1_000 Helium atoms

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
    clock_t begin = clock();
    for(int t=0; t<dmcd->Nts; t++) {
        diffuse(dmcd);
        computeEnergy(dmcd);
        branch(dmcd);

        DOUBLE_BUFFER_NEXT_STEP(dmcd->Vb);

        if(t % 10 == 0) {
            clock_t end = clock();
            double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
            printf("Iteration %6d/%6lld (took %lf s; %lf per iteration)\n", t, dmcd->Nts, time_spent,time_spent/10);
            fflush(stdout);
            begin = clock();
        }

    }

    return 0;
}
