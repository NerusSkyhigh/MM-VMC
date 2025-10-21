#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "gglib/gg_math.h"
#include "gglib/gg_mem.h"
#include "physics.h"

typedef struct {
    size_t Np;         // Number of particles

    double El;         // current Local Energy
    CyclicBuffer* Eb;  // Energy buffer where to accumulate energy

    DoubleBuffer* xb;   // Positions
    double* F;         // Drift Velocity F
    double a1;
    double a2;

    double Psi2;

    size_t Nts;    // total Number of TimeSteps
    double dt;      // Timestep size

    // Box is centered in 0 and has size L: [-L/2; L/2]
    double L;
} VMCdata;



void initUniform(const VMCdata* vmcd) {
    const double L = vmcd->L;
    for(u_int64_t i=0; i<vmcd->Np*3*vmcd->Np; i++) {
        vmcd->xb->buffers[vmcd->xb->new][i] = L*(rand()/(double) RAND_MAX-0.5);
    }
}

void diffuse(const VMCdata* vmcd) {
    /* NOTE:
     *  The McMillan et al. use a simple diffusion, while this implementation uses Importance Sampling. The formula
     *  relation for importance sampling is
     *        $\mathbf{r}_{new} = \mathbf{r}_{old} + \Xi + D \mathbf{F}(\mathbf{r}_{old}) \delta t$
     *     with:
     *        $\Xi$ Normal distributed with mean 0 and variance $2D\delta t$.
     *        $\mathbf{F}(\mathbf{r}_{old})= \frac{1}{f} \nabla f$ drift velocity
     *  source https://compphysics.github.io/ComputationalPhysics2/doc/LectureNotes/_build/html/vmcdmc.html#importance-sampling
     *      (actually, the 2010 version of the book, but the link should provide the same info)
     *
     *  From the McMillan paper, we know that $\sigma=L/2$, therefore $2 D \delta t = L^2/4 \implies D \delta t = L^2/8$.
     *  This substitution, allows us to ignore the exact value of $D$ and $\delta t$ as long as their product is constant.
     */
    const double L = vmcd->L;
    const double deltaR = L/2.; // Value used in W. L. McMillan, Ground State of Liquid He⁴, Phys. Rev. 138, A442 (1965)
    const double Ddt = deltaR*deltaR/2;
    const double sqrt3i = 0.57735026919; // sqrt(1/3); The variance of \vec{R} is one, we sample one component at a time

    /* // SIMPLE DIFFUSION
    for(int i=0; i<3*vmcd->Np; i++) {
        // [TODO] Implement sampling with velocity
        double xn = vmcd->xb->buffers[vmcd->xb->old][i]+randn()*deltaR; // x new
        xn -= L * floor(xn/L + 0.5); // PBC; Use floorf on GPU
        vmcd->xb->buffers[vmcd->xb->new][i] = xn;
    } */

    // Importance Sampling
    memset(vmcd->F, 0, vmcd->Np*sizeof(double)); // Clear buffer for accumulation

    for(size_t pair=0; pair<vmcd->Np*(vmcd->Np-1)/2; pair++) {
        // NDR: For a CPU the usual for(i) {for(j>i)} is faster, but this form will come handy on the GPU
        const size_t p1 = (u_int64_t) (1+sqrt(1+8*pair))/2;
        const size_t p2 = pair-p1*(p1-1)/2;

        const double x1 = vmcd->xb->buffers[vmcd->xb->new][p1];
        const double y1 = vmcd->xb->buffers[vmcd->xb->new][p1+1];
        const double z1 = vmcd->xb->buffers[vmcd->xb->new][p1+2];

        const double x2 = vmcd->xb->buffers[vmcd->xb->new][p2];
        const double y2 = vmcd->xb->buffers[vmcd->xb->new][p2+1];
        const double z2 = vmcd->xb->buffers[vmcd->xb->new][p2+2];

        const double r = NORM(x1-x2,y1-y2,z1-z2);

        const double f   = WU_FEENBERG_TPWF(r, vmcd->a1, vmcd->a2);
        const double df  = dWU_FEENBERG_TPWF(r, vmcd->a1, vmcd->a2);  // (df/dr)
        const double dfof = df/f;
        const double F = dfof/r;

        const double Fx = F*(x1-x2);
        const double Fy = F*(y1-y2);
        const double Fz = F*(z1-z2);

        // p1
        vmcd->F[p1]   += Fx;
        vmcd->F[p1+1] += Fy;
        vmcd->F[p1+2] += Fz;

        //p2
        vmcd->F[p1]   -= Fx;
        vmcd->F[p1+1] -= Fy;
        vmcd->F[p1+2] -= Fz;
    }

    for(u_int64_t i=0; i<3*vmcd->Np; i++) {
        double xn = vmcd->xb->buffers[vmcd->xb->old][i] + // x_old
                    sqrt3i*randn()*deltaR + // Xi
                    Ddt*vmcd->F[i]; // (D*\delta t)*F(r_old) Drift Velocity
        xn -= L * floor(xn/L + 0.5); // PBC
        vmcd->xb->buffers[vmcd->xb->new][i] = xn;
    }

}

/**
 * @brief Computes the energy of the position stored in the vmcd->xb->new buffer
 * @return Local energy of the configuration
 */
double computeEnergy(const VMCdata* vmcd) {
    double El = 0;
    for(size_t pair=0; pair<vmcd->Np*(vmcd->Np-1)/2; pair++) {
        // NDR: For a CPU the usual for(i) {for(j)} is faster, but this form will come handy on the GPU
        const size_t p1 = (size_t) (1+sqrt(1+8*pair))/2;
        const size_t p2 = pair-p1*(p1-1)/2;

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
    for(size_t pair=0; pair<vmcd->Np*(vmcd->Np-1)/2; pair++) {
        // NDR: For a CPU the usual for(i) {for(j)} is faster, but this form will come handy on the GPU
        const size_t p1 = (size_t) (1+sqrt(1+8*pair))/2;
        const size_t p2 = pair-p1*(p1-1)/2;

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


// For now I'll just sample the parameter with the following grid:
// a1: 2:0.1:4
// a2: 4:0.1:6
// Non funziona perchè i parametri sono per il loro sistema specifico
int main(void) {
    printf("Started\n");
    srand( (unsigned int) time(NULL));
    const double RHO = 0.02277696793; // Density from W. L. McMillan, Ground State of Liquid He⁴, Phys. Rev.138, A442 (1965)

    /* 1. Allocate resources
     * MEMORY USAGE:
     *  Each particle has 3 coordinates stored in a double (8 bytes):
     *  size = 8 bytes/coordinate * 3 coordinates/particle * 1_000 particles/walkers * 10_000 walkers = 240 MB.
     */
    VMCdata* vmcd = malloc(sizeof(VMCdata));
    vmcd->Np = 100; // 100 Helium atoms
    vmcd->L = round( pow(vmcd->Np/RHO, 1./3.) );

    // Initial conditions from Density from W. L. McMillan, Ground State of Liquid He⁴, Phys. Rev.138, A442 (1965)
    vmcd->a1 = 2.51;
    vmcd->a2 = 5;

    vmcd->Nts = 2<<12; // timesteps
    vmcd->dt = 1.; // Delta timestep
    printf("Parameters set\n");

    vmcd->Eb = malloc( sizeof(CyclicBuffer));
    initCyclicBuffer(vmcd->Eb, 2<<11); // Average over the last 256 values of the energy
    printf("Resources loaded\n");

    // 2. Generate positions
    vmcd->xb = malloc(sizeof(DoubleBuffer));
    initDoubleBuffer( vmcd->xb, vmcd->Np);
    initUniform(vmcd);
    vmcd->F = malloc(sizeof(double)*3*vmcd->Np);
    printf("Walker initialized\n");

    // 2.2 Compute energy for the initial configuration
    vmcd->El = computeEnergy(vmcd);
    printf("Initial energy computed.\n");
    vmcd->Psi2 = computePsi2(vmcd);
    printf("Psi2 computed.\n");
    DOUBLE_BUFFER_NEXT_STEP(vmcd->xb);
    CYCLIC_BUFFER_PUSH(vmcd->Eb, vmcd->El);

    // 3. Main loop of the VMC
    clock_t begin = clock();
    for(u_int64_t t=0; t<vmcd->Nts; t++) {
        diffuse(vmcd);
        const double Psi2 = computePsi2(vmcd);

        // GPU do not like branching, so I do not check if the ratio is >= 1, the algorithm works nevertheless and
        // no resources are wasted (on GPU)
        const double r = rand()/(double) RAND_MAX;
        if(r<=Psi2/vmcd->Psi2) {
            // Monte Carlo Move accepted
            vmcd->El = computeEnergy(vmcd);    // Compute new local energy
            vmcd->Psi2 = Psi2;                 // Copy the new probability
            DOUBLE_BUFFER_NEXT_STEP(vmcd->xb); // Accept the new position
        }
        CYCLIC_BUFFER_PUSH(vmcd->Eb, vmcd->El); // Add energy of the move to the buffer


        if(t % 1000 == 0) {
            clock_t end = clock();
            double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
            printf("Iteration %6llu/%5zu (took %lf s; %lf ms per iteration)\n", t, vmcd->Nts, time_spent,time_spent);
            fflush(stdout);
            begin = clock();
        }

    }

    return 0;
}
