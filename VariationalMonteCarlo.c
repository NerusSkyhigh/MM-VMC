#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "gglib/gg_math.h"
#include "gglib/gg_mem.h"
#include "physics.h"

typedef struct {
    int64_t Np;         // Number of particles

    double El;         // Local Energy
    double aEl;         // Average Local Energy
    DoubleBuffer* xb;   // Positions
    double a1;
    double a2;

    double Psi2;

    int64_t Nts;    // total Number of TimeSteps
    double dt;      // Timestep size
} VMCdata;



void initUniform(const VMCdata* vmcd) {
    for(int i=0; i<vmcd->Np*3*vmcd->Np; i++) {
        // [TODO] Implement simulation box
        // [TODO] Add periodic boundary conditions
        vmcd->xb->buffers[vmcd->xb->new][i] = rand()/(double) RAND_MAX;
    }
}

void diffuse(const VMCdata* vmcd) {
    for(int i=0; i<3*vmcd->Np; i++) {
        // [TODO] Implement sampling with velocity
        // [TODO] Find a good value for \Delta R
        double const deltaR = 0.1;
        vmcd->xb->buffers[vmcd->xb->new][i] = vmcd->xb->buffers[vmcd->xb->old][i]+randn()*deltaR;
    }
}

/**
 * @brief Computes the energy of the position stored in the vmcd->xb->new buffer
 * @return Local energy of the configuration
 */
double computeEnergy(const VMCdata* vmcd) {
    double El = 0;
    for(int pair=0; pair<vmcd->Np*(vmcd->Np-1)/2; pair++) {
        // NDR: For a CPU the usual for(i) {for(j)} is faster, but this form will come handy on the GPU
        const int p1 = (int) (1+sqrt(1+8*pair))/2;
        const int p2 = pair-p1*(p1-1)/2;

        const double x1 = vmcd->xb->buffers[vmcd->xb->new][p1];
        const double y1 = vmcd->xb->buffers[vmcd->xb->new][p1+1];
        const double z1 = vmcd->xb->buffers[vmcd->xb->new][p1+2];

        const double x2 = vmcd->xb->buffers[vmcd->xb->new][p2];
        const double y2 = vmcd->xb->buffers[vmcd->xb->new][p2+1];
        const double z2 = vmcd->xb->buffers[vmcd->xb->new][p2+2];

        const double r = NORM(x1-x2,y1-y2,z1-z2);

        const double f   = WU_FEENBERG_TPWF(r, vmcd->a1, vmcd->a2);
        const double df  = dWU_FEENBERG_TPWF(r, vmcd->a1, vmcd->a2);  // (df/dr)
        const double ddf = ddWU_FEENBERG_TPWF(r, vmcd->a1, vmcd->a2); // (ddf/dr2)
        const double dfof = df/f;

        // [TODO] Include -hbar/2m with correct units
        El += - ( ddf/f - (dfof)*(dfof) + 2/r * dfof ) + LJ_potential(r);
    }
    return El;
}

/**
 * @brief Computes the wavefunciton squared (aka the non normalized probability of a configuration)
 * @return The wavefunction squared
 * @warning This function assumes a bosonic system with single wave function given by `WU_FEENBERG_TPWF`.
 */
double computePsi2(const VMCdata* vmcd) {
    double Psi = 1;
    for(int pair=0; pair<vmcd->Np*(vmcd->Np-1)/2; pair++) {
        // NDR: For a CPU the usual for(i) {for(j)} is faster, but this form will come handy on the GPU
        const int p1 = (int) (1+sqrt(1+8*pair))/2;
        const int p2 = pair-p1*(p1-1)/2;

        const double x1 = vmcd->xb->buffers[vmcd->xb->new][p1];
        const double y1 = vmcd->xb->buffers[vmcd->xb->new][p1+1];
        const double z1 = vmcd->xb->buffers[vmcd->xb->new][p1+2];

        const double x2 = vmcd->xb->buffers[vmcd->xb->new][p2];
        const double y2 = vmcd->xb->buffers[vmcd->xb->new][p2+1];
        const double z2 = vmcd->xb->buffers[vmcd->xb->new][p2+2];

        const double r = NORM(x1-x2,y1-y2,z1-z2);
        const double f   = WU_FEENBERG_TPWF(r, vmcd->a1, vmcd->a2);

        Psi *= f*f;
    }
    return Psi;
}



int main(void) {
    srand(time(NULL));

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
    vmcd->xb = malloc(sizeof(DoubleBuffer));
    initDoubleBuffer( &(vmcd->xb), vmcd->Np);
    initUniform(vmcd);
    printf("Walkers initialized\n");

    // 2.2 Compute energy for the initial configuration
    vmcd->El = computeEnergy(vmcd);
    printf("Initial energy computed.\n");
    vmcd->Psi2 = computePsi2(vmcd);
    printf("Psi2 computed.\n");
    DOUBLE_BUFFER_NEXT_STEP(vmcd->xb);

    // 3. Main loop of the VMC
    clock_t begin = clock();
    for(int t=0; t<vmcd->Nts; t++) {
        diffuse(vmcd);
        const double Psi2 = computePsi2(vmcd);

        // GPU do not like branching, so I do not check if the ratio is >= 1, the algorithm works nevertheless and
        // no resources are wasted (on GPU)
        const double r = rand()/(double) RAND_MAX;
        if(r<=Psi2/vmcd->Psi2) {
            // Monte Carlo Move accepted
            vmcd->El = computeEnergy(vmcd); // Compute new local energy
            vmcd->Psi2 = Psi2; // Copy the new probability
            DOUBLE_BUFFER_NEXT_STEP(vmcd->xb); // Accept the new position
        }

        // [TODO] Add code to change the vlaue of the parameters



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
