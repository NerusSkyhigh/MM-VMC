// VariationalMonteCarlo.cu
#include <curand_kernel.h>
#include <cub/cub.cuh>

#include <stdio.h>
#include <stdlib.h>

#include "ggpulib/gg_math.cuh"
#include "ggpulib/gg_mem.cuh"
extern "C" {
    #include "gglib/gg_math.h"
    #include "gglib/gg_io.h"
}

#include "physics.cuh"


typedef struct VMCdata {
    // Properties of the system
    unsigned int Np;              // Number of particles
    float rho;

    // Parameters of time evolution
    unsigned int Ntseq;           // Number of timesteps for the equilibration
    unsigned int Nts;             // total Number of TimeSteps
    unsigned int decTs;           // Decorrelation timesteps (euristic)
    unsigned int acc_steps;       // Number of steps to sample to adapt the value of gamma

    // Simulation box
    float L;               // Box is centered in 0 and has size L: [-L/2; L/2]

    // Microscopic quantities
    DoubleBuffer* coord;    // Positions
    float* h_coord;
    unsigned int* pairs;
    float* dr;         // Pairwise distances
    float* dxyz;           // Distances along one axis
    float* F;              // Drift Velocity F
    float* r_new;          // Buffer for the computation of the pariwise distance

    // Macroscopic quantities
    float* d_logPsi2;          // <Psi|Psi> is proportional to probability P(R)
    float* v_logPsi2;       // Vector of the contribution to logPsi2
    float h_logPsi2;

    CyclicBuffer* Eb;       // Energy buffer where to accumulate energy
    float* d_El;              // current Local Energy
    float* v_El;
    float h_El;

    // Meta properties
    curandState_t* rng_states;
    float gamma;          // Esteem for the displacement in the MC move
    float a1;              // Parameters for the trial wf
    float a2;

} VMCdata;


/**
 * @brief Store the indices of the pairs in the vector provided.
 * @param pairs Vector where to store the pairs linearly. Expected length is 2*n_particles*(n_particles-1)/2 = 2*n_pairs
 * @param n_particles the number of particles
 */
__global__ void d_storePairs(unsigned int* pairs, const unsigned int n_particles) {
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid < n_particles*(n_particles-1)/2) {
        // n_particles > p1 > p2 > 0
        const unsigned int p1 = (unsigned int) floorf( (1+sqrtf(1+8*tid))/2 );
        const unsigned int p2 = tid - p1*(p1-1)/2;

        pairs[2*tid+0] = p1;
        pairs[2*tid+1] = p2;
        //printf("[d_storePairs %u] %u-%u\n", tid, p1, p2);
    }
}

__global__ void d_initPositionsOnCube(float* coordinates, unsigned int n_particles, float L) {
    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if (tid < n_particles) {
        // [MINOR TODO] If this function become hot I could micro-optimize it by computing
        // the two constant on the CPU. They have the same values in all the kernel calls.
        const unsigned int ns = (unsigned int) ceilf( powf(n_particles, 1.f/3.f) );
        const float step = L/ns;

        // x changes faster than y, which changes faster than z
        const unsigned int xid = tid % ns;
        const unsigned int yid = (tid / ns) % ns;
        const unsigned int zid = tid / (ns*ns);

        coordinates[3*tid+0] = xid*step - L/2.f; //+ origin[0];
        coordinates[3*tid+1] = yid*step - L/2.f; //+ origin[1];
        coordinates[3*tid+2] = zid*step - L/2.f; // + origin[2];
        //printf("[d_initPositionsOnCube %u] %f %f %f\n", tid, coordinates[3*tid+0], coordinates[3*tid+1], coordinates[3*tid+2]);
    }
}

/**
 * @brief Fill a vector with the contribution to logPsi2 for each pair
 * @param dr distances between pair of particles
 * @param n_pairs the number of pairs
 * @param a1 variational coefficient 1
 * @param a2 variational coefficient 2
 * @param v_logPsi2 vector of size `n_pairs` that will store the contributions
 * @warning This function assumes a bosonic system with single wave function given by `WU_FEENBERG_TPWF`.
 */
__global__ void d_computeLogPsi2(float* dr, unsigned int n_pairs, float a1, float a2, float* v_logPsi2) {
    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if (tid < n_pairs) {
        v_logPsi2[tid] = 2.f * logWU_FEENBERG_TPWF(dr[tid], a1, a2);
        //printf("[d_computeLogPsi2 %u] 2*logWU_FEENBERG_TPWF(%f, %f, %f)=%f\n", tid, dr[tid], a1, a2, v_logPsi2[tid]);
    }
}


__global__ void d_computeQuantumForce(const float a1, const float a2, unsigned int* pairs, unsigned int n_pairs, float* dxyz, float* dr, float* F) {
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
    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if (tid<n_pairs) {
        const unsigned int p1 = pairs[2*tid];
        const unsigned int p2 = pairs[2*tid+1];

        const float dx = dxyz[3*tid+0];
        const float dy = dxyz[3*tid+1];
        const float dz = dxyz[3*tid+2];
        const float r = dr[tid];

        const float f   =  WU_FEENBERG_TPWF(r, a1, a2);
        const float df  = dWU_FEENBERG_TPWF(r, a1, a2);  // (df/dr)

        const float k = 2.f/r * df/f; // Auxiliary constant for the drift velocity in radial coordinates over the distance
        const float Fx = k*dx;
        const float Fy = k*dy;
        const float Fz = k*dz;

        // [TODO] Find a way to prevent these atomicAdds. They WILL slow down execution
        // chatgpt suggests `use warp-level reductions per particle or shared-memory accumulation by block.`
        // look into it

        // p1
        atomicAdd( &(F[3*p1+0]), Fx);
        atomicAdd( &(F[3*p1+1]), Fy);
        atomicAdd( &(F[3*p1+2]), Fz);

        //p2
        atomicAdd( &(F[3*p2+0]), -Fx);
        atomicAdd( &(F[3*p2+1]), -Fy);
        atomicAdd( &(F[3*p2+2]), -Fz);
    }
}



__global__ void d_diffuse(const float L, const float gamma, const unsigned int n_particles, float* prev_coord, float* F, curandState_t* states,
                        float* next_coord) {
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

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if(tid < n_particles) {
        curandState_t localState = states[tid];

        const float Ddt = gamma*gamma/2.;

        const float x = prev_coord[3*tid+0];
        const float y = prev_coord[3*tid+1];
        const float z = prev_coord[3*tid+2];

        // By the end ot the p2 cycle, all the accumulation over p1 is done. So I can compute the new position
        //          n_new = x_old + Xi + (D*\delta t)*F(r_old) [old + gaussian drift + drift velocity]
        float xn = x + curand_normal(&localState)*gamma + Ddt*F[3*tid+0];
        float yn = y + curand_normal(&localState)*gamma + Ddt*F[3*tid+1];
        float zn = z + curand_normal(&localState)*gamma + Ddt*F[3*tid+2];
        states[tid] = localState;

        // Impose PCB on new positions
        xn -= L * nearbyintf(xn/L);
        yn -= L * nearbyintf(yn/L);
        zn -= L * nearbyintf(zn/L);

        // Update
        next_coord[3*tid+0] = xn;
        next_coord[3*tid+1] = yn;
        next_coord[3*tid+2] = zn;
    }
}

/**
 * @brief Computes the energy from the distances stored in the vmcd->dr
 */
__global__ void d_computeEnergy(unsigned int* pairs, unsigned int n_pairs, float* dr, const float a1, const float a2, float* v_energy) {
    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if (tid<n_pairs) {
        const float r = dr[tid];

        const float f   =   WU_FEENBERG_TPWF(r, a1, a2);
        const float df  =  dWU_FEENBERG_TPWF(r, a1, a2);   // (df/dr)
        const float ddf = ddWU_FEENBERG_TPWF(r, a1, a2);   // (ddf/dr2)
        const float dfof = df/f;
        // [TODO] Transform this in constant expression
        const float hbar2_2me = -6.06233735; // [Kelvin * Angstrom^2]

        v_energy[tid] = hbar2_2me*( ddf/f - (dfof)*(dfof) + 2.f/r * dfof ) + LJ_potential(r);
    }
}



int main(int argc, char** argv) {
    float RAND_MAX_i = 1.f / RAND_MAX;

    VMCdata* vmcd = (VMCdata*) malloc(sizeof(VMCdata));
    // Properties of the system
    vmcd->Np = 200;//256;                                 // 200 Helium atoms
    vmcd->rho = 0.02277696793f;                     // Density from W. L. McMillan, Ground State of Liquid He⁴, Phys. Rev.138, A442 (1965)

    // Parameters of time evolution
    vmcd->Ntseq =    16384;                             // equilibration timesteps
    vmcd->Nts =   32768; //16*65536;                              // run timesteps
    vmcd->decTs = 1; //64;                              // decorrelation timesteps 16384/128= 128
    vmcd->acc_steps = 256;                          // gamma is adapted 8192/256 = 32 times
    size_t accepted_moves = 0;

    // Simulation box
    vmcd->L = powf(vmcd->Np/vmcd->rho, 1.f/3.f);        // Box size constructed to have the same density as in the paper
    printf("L=%f\n", vmcd->L);


    // Microscopic quantities
    // -- Init positions --
    vmcd->coord = (DoubleBuffer*) malloc(sizeof(DoubleBuffer));         // Double buffer for R_old and R_new
    d_initDoubleBuffer(vmcd->coord, 3*vmcd->Np);
    vmcd->h_coord = (float*) malloc(3*vmcd->Np*sizeof(float));

    // -- Init pairwise distances --
    const unsigned int n_pairs = vmcd->Np*(vmcd->Np-1)/2;
    CUDA_CHECK( cudaMalloc(&(vmcd->pairs),  2*n_pairs*sizeof(unsigned int)) );
    CUDA_CHECK( cudaMalloc(&(vmcd->dxyz),   3*n_pairs*sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&(vmcd->dr),     n_pairs*sizeof(float)) );

    // -- Init drift velocity --
    CUDA_CHECK( cudaMalloc(&(vmcd->F), 3*vmcd->Np*sizeof(float)) ); // Drift velocity

    // Macroscopic quantities
    // -- Init log WaveFunction --
    CUDA_CHECK( cudaMalloc(&(vmcd->v_logPsi2), n_pairs*sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&(vmcd->d_logPsi2), sizeof(float)) );
    float h_proplogPsi2 = 0.f;

    // -- Init Energy --
    vmcd->Eb = (CyclicBuffer*) malloc( sizeof(CyclicBuffer));   // Average over the last 2048 values of the energy
    h_initCyclicBuffer(vmcd->Eb, vmcd->Nts/vmcd->decTs-1);
    CUDA_CHECK( cudaMalloc( &(vmcd->v_El), n_pairs*sizeof(float)));
    CUDA_CHECK( cudaMalloc(&(vmcd->d_El), sizeof(float)) );

    // Meta properties
    vmcd->a1 = 2.6f;//2.51f;                                // Initial conditions for density in W. L. McMillan, Ground State
    vmcd->a2 = 5.f;                                   // of Liquid He⁴, Phys. Rev.138, A442 (1965)
    vmcd->gamma = 0.16f;                            // Initial esteem for MC step displacement

    CUDA_CHECK( cudaMalloc( &(vmcd->rng_states), vmcd->Np*sizeof(curandState_t) ) );

    const int threads = 32;
    const int blocks_particles = (vmcd->Np + threads - 1) / threads;
    const int blocks_pairs = (n_pairs + threads - 1) / threads;

    // Fill the values
    setupRNG<<<blocks_particles, threads>>>(vmcd->rng_states, 0, vmcd->Np);

    //FILE* fp_energy = fopen("../energy200.csv", "w");
    FILE* fp_ave_energy = fopen("../ave_energy200.csv", "w");
    if( fp_ave_energy == NULL ) {
        printf("Error opening file\n");
    }
    //fprintf(fp_energy, "a1, a2, Timestep, Total Energy [K], Energy per particle [erg]\n");
    fprintf(fp_ave_energy, "a1, a2, Average Total Energy [K], Standard Deviation [K], Energy per particle [erg]\n");

    for(vmcd->a1 = 2.55f; vmcd->a1 < 2.65f; vmcd->a1 += 0.001f) {
        for(vmcd->a2 = 4.9f; vmcd->a2 < 5.5f; vmcd->a2 += 0.01f) {
            vmcd->gamma = 0.16f;
            accepted_moves = 0;
            printf("{a1=%.2e; a2=%.2e} ", vmcd->a1, vmcd->a2);

            d_initPositionsOnCube<<<blocks_particles, threads>>>(vmcd->coord->next, vmcd->Np, vmcd->L);  // Distribute particle on a Grid
            d_storePairs<<<blocks_pairs, threads>>>(vmcd->pairs, vmcd->Np);
            cudaDeviceSynchronize(); // [MINOR TODO] I'm using the default stream so this is not essential.

            // [TODO] Find correct parameters for this kernel execution
            d_computePairwiseDistancesWithPCB<<<blocks_pairs, threads>>>(vmcd->coord->next, vmcd->pairs, vmcd->Np, vmcd->L, vmcd->dxyz, vmcd->dr);
            d_computeLogPsi2<<<blocks_pairs, threads>>>(vmcd->dr, n_pairs, vmcd->a1, vmcd->a2, vmcd->v_logPsi2);

            // Phase 1: Query how much temporary storage is needed. As the length of `vmcd->v_logPsi2` does not change, this
            //          needs to be done only once.
            void* d_temp_storage = nullptr;     // NULL pointer acts as a flag for `cub::DeviceReduce::Sum` to change the value
            size_t temp_storage_bytes = 0;      // of `temp_storage_bytes` without computing anything
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vmcd->v_logPsi2, vmcd->d_logPsi2, n_pairs);
            CUDA_CHECK( cudaMalloc(&d_temp_storage, temp_storage_bytes) );
            // Phase 2: Actually perform the reduction
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vmcd->v_logPsi2, vmcd->d_logPsi2, n_pairs);
            CUDA_CHECK( cudaMemcpy( &(vmcd->h_logPsi2),  vmcd->d_logPsi2,sizeof(float), cudaMemcpyDeviceToHost));

            DOUBLE_BUFFER_NEXT_STEP(vmcd->coord); // Move coordinates from R_new to R_old
            cudaDeviceSynchronize();

            // Monte Carlo for equilibration
            printf("Eq: ");
            for(unsigned int t=0; t< vmcd->Ntseq; t++) {
                CUDA_CHECK( cudaMemset(vmcd->F, 0, 3*vmcd->Np*sizeof(float)) );                                                                      // Clear buffer for accumulation
                d_computeQuantumForce<<<blocks_particles, threads>>>(vmcd->a1, vmcd->a2, vmcd->pairs, n_pairs, vmcd->dxyz, vmcd->dr, vmcd->F);                  // Compute quantum force
                d_diffuse<<<blocks_particles, threads>>>(vmcd->L, vmcd->gamma, vmcd->Np, vmcd->coord->prev, vmcd->F, vmcd->rng_states, vmcd->coord->next);      // Diffuse particles
                d_computePairwiseDistancesWithPCB<<<blocks_pairs, threads>>>(vmcd->coord->next, vmcd->pairs, vmcd->Np, vmcd->L, vmcd->dxyz, vmcd->dr);     // Compute pairwise distance
                d_computeLogPsi2<<<blocks_pairs, threads>>>(vmcd->dr, n_pairs, vmcd->a1, vmcd->a2, vmcd->v_logPsi2);                                            // Compute logPsi
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vmcd->v_logPsi2, vmcd->d_logPsi2, n_pairs);                             // Aggregate logPsi in a single value
                CUDA_CHECK( cudaMemcpy(&h_proplogPsi2,  vmcd->d_logPsi2,sizeof(float), cudaMemcpyDeviceToHost) );                                    // Copy the value on the CPU
                cudaDeviceSynchronize();                                                                                                                        // Wait for kernel to execute

                const float deltaLogPsi2 = h_proplogPsi2 - vmcd->h_logPsi2;
                if( deltaLogPsi2>0 || logf( rand()*RAND_MAX_i)<= deltaLogPsi2) {
                    accepted_moves++;
                    vmcd->h_logPsi2 = h_proplogPsi2;        // Copy the new probability
                    DOUBLE_BUFFER_NEXT_STEP(vmcd->coord);   // Accept the new position
                }

                if( (t+1)% vmcd->acc_steps==0) {
                    float acc = (float) accepted_moves/ (float) vmcd->acc_steps;
                    const float acc_target = 0.5;

                    //printf("[%3.0lf%%, gamma=%.3e->", 100*acc, vmcd->gamma);
                    vmcd->gamma *= 1.f + 0.5f*(acc-acc_target);
                    //printf("%.3e] ", vmcd->gamma);
                    printf("[%2.0lf%%, %.3e] ", 100*acc, vmcd->gamma);
                    accepted_moves = 0;
                }
            }


            printf("\tMC run: ");
            //LammpsDat* positions = (LammpsDat*) malloc(sizeof(LammpsDat));
            //initLammpsData(positions, "positions.lammpstrj", vmcd->Np, vmcd->L);


            // Monte Carlo loop for the WF
            for(unsigned int t=0; t< vmcd->Nts; t++) {
                CUDA_CHECK( cudaMemset(vmcd->F, 0, 3*vmcd->Np*sizeof(float)) ); // Clear buffer for accumulation
                d_computeQuantumForce<<<blocks_particles, threads>>>(vmcd->a1, vmcd->a2, vmcd->pairs, n_pairs, vmcd->dxyz, vmcd->dr, vmcd->F);
                d_diffuse<<<blocks_particles, threads>>>(vmcd->L, vmcd->gamma, vmcd->Np, vmcd->coord->prev, vmcd->F, vmcd->rng_states, vmcd->coord->next);
                d_computePairwiseDistancesWithPCB<<<blocks_pairs, threads>>>(vmcd->coord->next, vmcd->pairs, vmcd->Np, vmcd->L, vmcd->dxyz, vmcd->dr);
                d_computeLogPsi2<<<blocks_pairs, threads>>>(vmcd->dr, n_pairs, vmcd->a1, vmcd->a2, vmcd->v_logPsi2);
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vmcd->v_logPsi2, vmcd->d_logPsi2, n_pairs);
                CUDA_CHECK( cudaMemcpy(&h_proplogPsi2,  vmcd->d_logPsi2,sizeof(float), cudaMemcpyDeviceToHost) );
                cudaDeviceSynchronize();

                const float deltaLogPsi2 = h_proplogPsi2 - vmcd->h_logPsi2;
                if( deltaLogPsi2>0 || logf( rand()*RAND_MAX_i)<= deltaLogPsi2) {
                    accepted_moves++;
                    d_computeEnergy<<<blocks_pairs, threads>>>(vmcd->pairs, n_pairs, vmcd->dr, vmcd->a1, vmcd->a2, vmcd->v_El); // Compute new local energy
                    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vmcd->v_El, vmcd->d_El, n_pairs);
                    CUDA_CHECK( cudaMemcpy(&(vmcd->h_El), vmcd->d_El, sizeof(float), cudaMemcpyDeviceToHost) );

                    vmcd->h_logPsi2 = h_proplogPsi2;                // Copy the new probability
                    DOUBLE_BUFFER_NEXT_STEP(vmcd->coord);   // Accept the new position

                    //fprintf(fp_energy, "%.2f, %.2f, %u, %.6e, %.6e\n", vmcd->a1, vmcd->a2, t, vmcd->h_El, 1.38e-16 * ((double) vmcd->h_El / (double) vmcd->Np)  );
                }

                if(t%vmcd->decTs == 0) {
                    //printf("[%u/%u] El=%.4e\n", t, vmcd->Nts, vmcd->h_El);
                    CYCLIC_BUFFER_PUSH(vmcd->Eb, vmcd->h_El);
                    CUDA_CHECK( cudaMemcpy(vmcd->h_coord,  vmcd->coord->prev,3*vmcd->Np*sizeof(float), cudaMemcpyDeviceToHost) );
                    //writeLammpsDatFrame(positions, vmcd->h_coord);
                }
            }

            double aveE, varE;
            computeAveVarf(vmcd->Eb->data, vmcd->Eb->capacity, &aveE, &varE);
            printf("aveE=%.6e [K] stdE=%.6e [K], aveE/N=%.6e [erg]\n\n", aveE, sqrtf(varE), 1.38e-16 * ((double) aveE / (double) vmcd->Np) );
            fprintf(fp_ave_energy, "%.2f, %.2f, %.6e, %.6e, %.6e\n", vmcd->a1, vmcd->a2, aveE, sqrtf(varE), 1.38e-16 * ((double) aveE / (double) vmcd->Np));
        }
    }

    //fclose(fp_energy);
    fclose(fp_ave_energy);
    // [MAJOR TODO] Free memory

    return 0;
}
