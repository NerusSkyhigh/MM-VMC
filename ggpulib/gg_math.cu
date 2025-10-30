// gg_math.cu
// Created by Guglielmo Grillo on 17/10/25.
//

#include <curand_kernel.h>

#include "gg_math.cuh"

__global__ void d_computePairwiseDistancesWithPCB(const float* coordinates, const unsigned int* pairs,
                                                  const unsigned int n_particles, const float L,
                                                  float* dxyz, float* r_utb) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_particles*(n_particles-1)/2) {
        unsigned int p1 = pairs[2*tid];
        unsigned int p2 = pairs[2*tid + 1];

        const float x1 = coordinates[3*p1];
        const float y1 = coordinates[3*p1+1];
        const float z1 = coordinates[3*p1+2];

        const float x2 = coordinates[3*p2];
        const float y2 = coordinates[3*p2+1];
        const float z2 = coordinates[3*p2+2];

        float dx = x1-x2;
        float dy = y1-y2;
        float dz = z1-z2;

        // PBC - Find minimal image
        dx -= L*rint(dx/L);
        dy -= L*rint(dy/L);
        dz -= L*rint(dz/L);

        dxyz[3*tid+0] = dx;
        dxyz[3*tid+1] = dy;
        dxyz[3*tid+2] = dz;
        r_utb[tid] = NORM(dx,dy,dz);
    }
}


__global__ void setupRNG(curandState_t* states, unsigned long long seed, int n_seeds) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_seeds) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}