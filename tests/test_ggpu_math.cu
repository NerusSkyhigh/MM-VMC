#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
extern "C" {
#include "../gglib/gg_math.h"
}
#include "../ggpulib/gg_math.cuh"

#define CUDA_CHECK(ans) ASSERT_EQ((ans), cudaSuccess)

TEST(GgGpuMathTest, d_computePairwiseDistancesWithPCB_CubeNoPCB) {
    const int n_particles = 8;
    const float L = 10.0f;

    std::vector<float> coords(3 * n_particles);
    int idx = 0;
    for (int x = 0; x < 2; ++x) {
        for (int y = 0; y < 2; ++y) {
            for (int z = 0; z < 2; ++z) {
                if (idx >= n_particles) {
                    printf("Too many points\n");
                    break;
                }
                coords[3*idx + 0] = static_cast<float>(x);
                coords[3*idx + 1] = static_cast<float>(y);
                coords[3*idx + 2] = static_cast<float>(z);
                idx++;
            }
        }
    }

    const int n_pairs = n_particles * (n_particles - 1) / 2;
    std::vector<unsigned int> pairs(2 * n_pairs);
    int pair_idx = 0;
    for (int p1 = 0; p1 < n_particles; ++p1) {
        for (int p2 = p1 + 1; p2 < n_particles; ++p2) {
            pairs[2 * pair_idx]     = static_cast<unsigned int>(p1);
            pairs[2 * pair_idx + 1] = static_cast<unsigned int>(p2);
            ++pair_idx;
        }
    }

    std::vector<float> dxyz(3 * n_pairs, 0.0f);
    std::vector<float> r_utb_dev(n_pairs, 0.0f);   // device -> host copy uses float
    std::vector<double> r_utb_cpu(n_pairs, 0.0);   // CPU reference uses double (your CPU function)

    float *d_coords = nullptr, *d_dxyz = nullptr, *d_rutb = nullptr;
    unsigned int* d_pairs;
    CUDA_CHECK(cudaMalloc(&d_coords, coords.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pairs, pairs.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_dxyz, dxyz.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rutb, r_utb_dev.size() * sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
    printf("cudaMalloc Worked\n");

    CUDA_CHECK(cudaMemcpy(d_coords, coords.data(), coords.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pairs, pairs.data(), pairs.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaGetLastError());
    printf("cudaMemcpy Worked\n");

    // launch
    int threads = std::min(32, n_pairs);
    int blocks = (n_pairs + threads - 1) / threads;
    d_computePairwiseDistancesWithPCB<<<blocks, threads>>>(d_coords, d_pairs, n_particles, L, d_dxyz, d_rutb);
    CUDA_CHECK(cudaGetLastError());

    printf("Launching %d blocks of %d threads (n_pairs=%d)\n", blocks, threads, n_pairs);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

    // copy results back
    CUDA_CHECK(cudaMemcpy(dxyz.data(), d_dxyz, dxyz.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r_utb_dev.data(), d_rutb, r_utb_dev.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Build CPU reference (convert coords to double as CPU function expects double)
    std::vector<double> coords_d(3 * n_particles);
    for (size_t i = 0; i < coords.size(); ++i) coords_d[i] = static_cast<double>(coords[i]);
    computePairwiseDistancesWithPCB(coords_d.data(), n_particles, static_cast<double>(L), r_utb_cpu.data());

    // compare
    for (int i = 0; i < n_pairs; ++i) {
        EXPECT_NEAR(r_utb_cpu[i], static_cast<double>(r_utb_dev[i]), 1e-5);
    }

    // simple check on first pair (example)
    EXPECT_NEAR(r_utb_cpu[0], static_cast<double>(r_utb_dev[0]), 1e-5);

    CUDA_CHECK(cudaFree(d_coords));
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_dxyz));
    CUDA_CHECK(cudaFree(d_rutb));
}
