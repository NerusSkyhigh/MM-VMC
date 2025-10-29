// VariationalMonteCarlo.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "gglib/gg_math.h"
#include "gglib/gg_mem.h"
#include "physics.h"


typedef struct VMCdata {
    size_t Np;              // Number of particles

    double logPsi2;         // <Psi|Psi> is proportional to probability P(R)
    double El;              // current Local Energy
    CyclicBuffer* Eb;       // Energy buffer where to accumulate energy

    DoubleBuffer* coord;    // Positions
    double* pair_r;         // Pairwise distances

    double* F;              // Drift Velocity F
    double* r_new;          // Buffer for the computation of the pariwise distance

    double deltaR;          // Esteem for the diplacement in the MC move
    double a1;              // Parameters for the trial wf
    double a2;

    size_t Ntseq;           // Number of timesteps for the equilibration
    size_t Nts;             // total Number of TimeSteps
    size_t decTs;           // Decorrelation timesteps (euristic)
    double L;               // Box is centered in 0 and has size L: [-L/2; L/2]
} VMCdata;



void initPositionsOnGrid(const VMCdata* vmcd) {
    const double L = vmcd->L;
    const double cbrN = ceil(pow( (double) vmcd->Np, 1./3.)); // CuBicRoot(N)
    size_t i = 0;

    for(size_t x=0; x<cbrN; x++) {
        for(size_t y=0; y<cbrN; y++) {
            for(size_t z=0; z<cbrN; z++) {
                if(i>=vmcd->Np)
                    break;
                vmcd->coord->next[3*i+0] = L*(x/cbrN-0.5);
                vmcd->coord->next[3*i+1] = L*(y/cbrN-0.5);
                vmcd->coord->next[3*i+2] = L*(z/cbrN-0.5);
                i++;
            }
        }
    }
}

void diffuseWithImportanceSampling(const VMCdata* vmcd) {
    /* NOTE:
     *  McMillan et al. use a simple diffusion, while this implementation uses Importance Sampling. The formula relation
     *  for importance sampling is
     *        $\mathbf{r}_{new} = \mathbf{r}_{old} + \Xi + D \delta t \mathbf{F}(\mathbf{r}_{old})$
     *     with:
     *        $\Xi$ Normal distributed with mean 0 and variance $2D\delta t$.
     *        $\mathbf{F}(\mathbf{r}_{old})= \frac{1}{f} \nabla f$ drift velocity
     *  source https://compphysics.github.io/ComputationalPhysics2/doc/LectureNotes/_build/html/vmcdmc.html#importance-sampling
     *      (actually, the 2010 version of the book, but the link should provide the same info)
     *
     *  $\sigma=$ is tuned at runtime. We have that $2 D \delta t = \sigma^2 \implies D \delta t = \sigma^2 /2$.
     *  This substitution allows us to ignore the exact value of $D$ and $\delta t$ as long as their product is constant.
     */
    //const double sqrt3i = 0.57735026919; // sqrt(1/3); The variance of \vec{R} is one, we sample one component at a time
    const double L = vmcd->L;
    const double deltaR = vmcd->deltaR;
    const double Ddt = deltaR*deltaR/2.;


    memset(vmcd->F, 0, 3*vmcd->Np*sizeof(double)); // Clear buffer for accumulation

    for(size_t p1=0; p1<vmcd->Np; p1++) {
        const double x1 = vmcd->coord->prev[3*p1+0];
        const double y1 = vmcd->coord->prev[3*p1+1];
        const double z1 = vmcd->coord->prev[3*p1+2];

        for(size_t p2=p1+1; p2<vmcd->Np; p2++) {
            const double x2 = vmcd->coord->prev[3*p2+0];
            const double y2 = vmcd->coord->prev[3*p2+1];
            const double z2 = vmcd->coord->prev[3*p2+2];

            double dx = x1-x2;
            double dy = y1-y2;
            double dz = z1-z2;

            // PBC - Find minimal image;
            // As I need dx, dy, dz (and the coordinates for later update) I can't simpy use the cached r here
            dx -= L*round(dx/L);
            dy -= L*round(dy/L);
            dz -= L*round(dz/L);

            // And, as I already have dx, dy, dz, computing the NORM is (probably) faster than searching for the element
            // of vmcd->pair_r->next[r_idx];
            const double r = NORM(dx,dy,dz);

            const double f   =  WU_FEENBERG_TPWF(r, vmcd->a1, vmcd->a2);
            const double df  = dWU_FEENBERG_TPWF(r, vmcd->a1, vmcd->a2);  // (df/dr)

            const double F = 2./r * df/f; // Auxiliary constant for the drift velocity in radial coordinates over the distance
            const double Fx = F*dx;
            const double Fy = F*dy;
            const double Fz = F*dz;

            // p1
            vmcd->F[3*p1+0] += Fx;
            vmcd->F[3*p1+1] += Fy;
            vmcd->F[3*p1+2] += Fz;

            //p2
            vmcd->F[3*p2+0] -= Fx;
            vmcd->F[3*p2+1] -= Fy;
            vmcd->F[3*p2+2] -= Fz;
        }
        // By the end ot the p2 cycle, all the accumulation over p1 is done. So I can compute the new position
        //          n_new = x_old + Xi + (D*\delta t)*F(r_old) [old + gaussian drift + drift velocity]
        double xn = x1 + /*sqrt3i */ randn()*deltaR + Ddt*vmcd->F[3*p1+0];
        double yn = y1 + /*sqrt3i */ randn()*deltaR + Ddt*vmcd->F[3*p1+1];
        double zn = z1 + /*sqrt3i */ randn()*deltaR + Ddt*vmcd->F[3*p1+2];

        // Impose PCB on new positions
        xn -= L * round(xn/L);
        yn -= L * round(yn/L);
        zn -= L * round(zn/L);

        // Update
        vmcd->coord->next[3*p1+0] = xn;
        vmcd->coord->next[3*p1+1] = yn;
        vmcd->coord->next[3*p1+2] = zn;
    }
}

/**
 * @brief Computes the energy from the distances stored in the vmcd->pair_r
 * @return Local energy of the configuration
 */
double computeEnergy(const VMCdata* vmcd) {
    double El = 0;

    for (size_t p1=0; p1<vmcd->Np; p1++) {
        for(size_t p2=p1+1; p2<vmcd->Np; p2++) {

            const size_t r_idx = UTIDX(p1, p2, vmcd->Np);
            const double r = vmcd->pair_r[r_idx];

            const double f   =   WU_FEENBERG_TPWF(r, vmcd->a1, vmcd->a2);
            const double df  =  dWU_FEENBERG_TPWF(r, vmcd->a1, vmcd->a2);   // (df/dr)
            const double ddf = ddWU_FEENBERG_TPWF(r, vmcd->a1, vmcd->a2);   // (ddf/dr2)
            const double dfof = df/f;

            const double hbar2_2me = -6.06233735; // [Kelvin * Angstrom^2]

            El += hbar2_2me*( ddf/f - (dfof)*(dfof) + 2/r * dfof ) + LJ_potential(r);
        }
    }

    return El;
}

/**
 * @brief Computes the wavefunciton squared from the distances stored in vmcd->pair_r
 * @return The logaritm of the wavefunction squared
 * @warning This function assumes a bosonic system with single wave function given by `WU_FEENBERG_TPWF`.
 */
double computeLogPsi2(const VMCdata* vmcd) {
    double logPsi = 0;

    for (size_t p1=0; p1<vmcd->Np; p1++) {
        for (size_t p2=p1+1; p2<vmcd->Np; p2++) {

            const size_t r_idx = UTIDX(p1, p2, vmcd->Np);
            const double r = vmcd->pair_r[r_idx];

            const double logf = logWU_FEENBERG_TPWF(r, vmcd->a1, vmcd->a2);
            logPsi += logf;
        }
    }
    return 2.*logPsi;
}



int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage %s [a1] [a2] [Np]\n", argv[0]);
        exit(-1);
    }

    srand( (unsigned int) time(NULL) );
    const double RAND_MAX_i = 1. / ( (double) RAND_MAX );

    VMCdata* vmcd = malloc(sizeof(VMCdata));
    const double RHO = 0.02277696793; // Density from W. L. McMillan, Ground State of Liquid He⁴, Phys. Rev.138, A442 (1965)

    /// 1. Allocate resources
    vmcd->Ntseq = 8192;                             // equilibration timesteps
    vmcd->Nts = 16384;                              // run timesteps
    vmcd->decTs = 256;                              // decorrelation timesteps
    const int acc_step = 512;

    vmcd->Np = (size_t) atoi(argv[3]); //200;                                 // 200 Helium atoms
    vmcd->L = pow(vmcd->Np/RHO, 1./3.);             // Box size constructed to have the same density as in the paper

    vmcd->a1 = atof(argv[1]); //2.51;                                // Initial conditions for density in W. L. McMillan, Ground State
    vmcd->a2 = atof(argv[2]); //5;                                   // of Liquid He⁴, Phys. Rev.138, A442 (1965)
    vmcd->deltaR = 0.16;                             // Initial esteem for MC step displacement

    vmcd->Eb = malloc( sizeof(CyclicBuffer));   // Average over the last 2048 values of the energy
    initCyclicBuffer(vmcd->Eb, vmcd->Nts/vmcd->decTs-10);

    // 2. Init positions
    vmcd->coord = malloc(sizeof(DoubleBuffer));         // Double buffer for R_old and R_new
    initDoubleBuffer( vmcd->coord, 3*vmcd->Np);

    // 2.1 Init pairwise distances
    const size_t Npairs = vmcd->Np*(vmcd->Np-1)/2;
    vmcd->pair_r = malloc( sizeof(double)* Npairs);

    // 2.2 Init drift velocity
    vmcd->F = malloc(sizeof(double)*3*vmcd->Np);        // Drift velocity


    // 2.3 Init arrays for the MC of the parameters
    double* aveE = malloc(sizeof(double)*2048);
    double* varE = malloc(sizeof(double)*2048);

    // 3. Main loop of the VMC
    char* filename = malloc(256);
    sprintf(filename, "np%zu_a1%.2lf_a2%.2f.csv", vmcd->Np, vmcd->a1, vmcd->a2);
    FILE* f = fopen(filename, "w");
    fprintf(f, "aveE, varE, a1, a2\n");

    vmcd->deltaR = 0.1;


    // Monte Carlo loop for a1 e a2
    for(size_t a=0; a<256; a++) {
        printf("[%4zu/256] {%.2lf, %.2lf} ", a, vmcd->a1, vmcd->a2);

        initPositionsOnGrid(vmcd);                              // Distribute particle on a Grid
        computePairwiseDistancesWithPCB(vmcd->coord->next, vmcd->Np, vmcd->L, vmcd->pair_r);
        vmcd->logPsi2 = computeLogPsi2(vmcd);   // Compute initial Log probability
        DOUBLE_BUFFER_NEXT_STEP(vmcd->coord); // 2.4 Move coordinates from R_new to R_old


        // Equilibration
        size_t accepted_moves = 0;
        //printf("(dR, a)\t");
        for(size_t t=0; t<vmcd->Ntseq; t++) {
            diffuseWithImportanceSampling(vmcd);
            computePairwiseDistancesWithPCB(vmcd->coord->next, vmcd->Np, vmcd->L, vmcd->pair_r);

            const double logPsi2 = computeLogPsi2(vmcd);
            const double deltaLogPsi2 = logPsi2-vmcd->logPsi2;
            const double logr = log(rand()*RAND_MAX_i);
            if(logr <= deltaLogPsi2) {
                accepted_moves++;                     // Monte Carlo move accepted
                vmcd->logPsi2 = logPsi2;              // Copy the new probability
                DOUBLE_BUFFER_NEXT_STEP(vmcd->coord); // Accept the new position
            }

            // Adapt deltaR to have an acceptance of ~50% every acc_step.
            if( (t+1)%acc_step==0) {
                double acc = (double) accepted_moves/ (double) acc_step;
                const double acc_target = 0.5;

                //printf("(%.0e, %3.0lf%%)\t", vmcd->deltaR/vmcd->L, 100*acc);
                printf("%3.0lf%% ", 100*acc);
                vmcd->deltaR *= 1.0 + 0.5*(acc-acc_target);
                accepted_moves = 0;
            }
        }
        vmcd->El = computeEnergy(vmcd);

        // Monte Carlo loop for the WF
        for(size_t t=0; t< vmcd->Nts; t++) {
            diffuseWithImportanceSampling(vmcd);
            computePairwiseDistancesWithPCB(vmcd->coord->next, vmcd->Np, vmcd->L, vmcd->pair_r);

            const double logPsi2 = computeLogPsi2(vmcd);
            const double deltaLogPsi2 = logPsi2-vmcd->logPsi2;
            if( deltaLogPsi2>0 || log(rand()*RAND_MAX_i)<= deltaLogPsi2) {
                vmcd->El = computeEnergy(vmcd);         // Compute new local energy
                vmcd->logPsi2 = logPsi2;                // Copy the new probability
                DOUBLE_BUFFER_NEXT_STEP(vmcd->coord);   // Accept the new position
            }
            if(t%vmcd->decTs == 0) {
                // Decorrelate energy by placing only one sample per 128
                CYCLIC_BUFFER_PUSH(vmcd->Eb, vmcd->El);
            }
        }

        computeAveStd(vmcd->Eb->data, vmcd->Eb->capacity, &(aveE[a]), &(varE[a]));
        printf("|\taveE=%.2e, varE=%8.2e\n", aveE[a], varE[a]);
        printf("\t %zu steps in %lfs (%lfms/iter)\t", vmcd->Nts+vmcd->Ntseq, time_spent, 1000*time_spent/(vmcd->Nts+vmcd->Ntseq));

        fprintf(f, "%.8e, %.8e, %.6lf, %.6lf\n", aveE[a], varE[a], a1[a], a2[a]);
        printf("\n");
    }

    fclose(f);

    // [TODO] Free memory of all the structs
    free(a1);
    free(a2);
    free(aveE);
    free(varE);
    free(vmcd->pair_r);
    free(vmcd->F);
    freeDoubleBuffer(vmcd->coord);
    free(vmcd->coord);
    freeCyclicBuffer(vmcd->Eb);
    free(vmcd->Eb);
    free(vmcd);

    return 0;
}

