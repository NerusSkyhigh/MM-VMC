// gg_cuda_macro.cuh
// Created by gu on 30/10/25.
//

#pragma once

#include <stdio.h>

#define CUDA_CHECK(ans) do { \
    cudaError_t err = (ans); \
    if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(err); \
    } \
} while (0)
