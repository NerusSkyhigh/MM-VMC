#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "gglib/gg_math.h"
#include "gglib/gg_mem.h"
#include "physics.h"

typedef struct {
    int64_t Np;     // Number of particles

    double* El;     // Local Energy
    double aEl;     // Average Local Energy
    double* x;      // Positions
    double a1;
    double a2;

    int64_t Nts;    // total Number of TimeSteps
    double dt;      // Timestep size

    DoubleBufferArena* Vb;   // potential (V) arena buffer [2*]

    //[TODO] Initialization is yet to be done
} VMCdata;



void initUniform(const VMCdata* vmcd) {
    for(int i=0; i<vmcd->Np*3*vmcd->Np; i++) {
        // [TODO] Add periodic boundary conditions
        vmcd->x[i] = rand()/(double)RAND_MAX;
    }
}

void diffuse(const VMCdata* dmcd) {
    for(int i=0; i<dmcd->Ncw*3*dmcd->Np; i++) {
        // [TODO] Implement importance sampling
        dmcd->x[i] +=randn()*dmcd->dt;
    }
}

void computeEnergy(const VMCdata* vmcd) {
    double El = 0;
    for(int pair=0; pair<vmcd->Np*(vmcd->Np-1)/2; pair++) {
        // NDR: For a CPU the usual for(i) {for(j)} is faster, but this form will come handy on the GPU
        const int p1 = (int) (1+sqrt(1+8*pair))/2;
        const int p2 = pair-p1*(p1-1)/2;

        const double x1 = vmcd->x[p1];
        const double y1 = vmcd->x[p1+1];
        const double z1 = vmcd->x[p1+2];

        const double x2 = vmcd->x[p2];
        const double y2 = vmcd->x[p2+1];
        const double z2 = vmcd->x[p2+2];

        const double r = NORM(x1-x2,y1-y2,z1-z2);

        const double dfov = dWU_FEENBERG_TPWF(r, );

        // [TODO] Include -hbar/2m
    }

    for(int walker=0; walker<vmcd->Ncw; walker++) {
        double V = 0;
        for(int p1=0; p1<vmcd->Np; p1++) {
            int64_t idx1 = walker*(3*vmcd->Np) + 3*p1;
            double x1 = vmcd->x[idx1];
            double y1 = vmcd->x[idx1+1];
            double z1 = vmcd->x[idx1+2];

            for(int p2=p1+1; p2<vmcd->Np; p2++) {
                int64_t idx2 = walker*vmcd->Np + p2;
                double x2 = vmcd->x[idx2];
                double y2 = vmcd->x[idx2+1];
                double z2 = vmcd->x[idx2+2];

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
    VMCdata* vmcd = malloc(sizeof(VMCdata));
    vmcd->Np = 100; // 100 Helium atoms

    vmcd->Nts = 1000000; // 1_000_000 timesteps
    vmcd->dt = 1.; // Delta timestep
    printf("Resources loaded\n");

    // 2. Generate positions
    vmcd->x = malloc(sizeof(double)* (3*vmcd->Np) );
    initUniform(vmcd);
    printf("Walkers initialized\n");

    // 2.2 Compute energy for the initial configuration
    vmcd->Vb = malloc(sizeof(DoubleBufferArena));
    initDoubleBufferArena(vmcd->Vb, vmcd->MNw);
    printf("Double buffer initialized\n");

    computeEnergy(vmcd);
    DOUBLE_BUFFER_NEXT_STEP(vmcd->Vb);
    printf("Initial energy computed\n");

    // 3. Main loop of the DMC
    clock_t begin = clock();
    for(int t=0; t<vmcd->Nts; t++) {
        diffuse(vmcd);
        computeEnergy(vmcd);
        branch(vmcd);

        DOUBLE_BUFFER_NEXT_STEP(vmcd->Vb);

        if(t % 10 == 0) {
            clock_t end = clock();
            double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
            printf("Iteration %6d/%6lld (took %lf s; %lf per iteration)\n", t, vmcd->Nts, time_spent,time_spent/10);
            fflush(stdout);
            begin = clock();
        }

    }

    return 0;
}
