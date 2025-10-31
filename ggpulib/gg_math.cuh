// gg_math.h
// Created by Guglielmo Grillo on 17/10/25.
//
#pragma once
#include <curand_kernel.h>

/** @file gg_math.cuh
 *  @brief Implementation of useful functions not available in the standard library
 */


/**
 * @brief Computes the norm of a 3D vector form the coordinates
 * @param x first coordinate of the vector
 * @param y second coordinate of the vector
 * @param z third coordinate of the vector
 */
#define NORM(x, y, z) sqrt( (x)*(x) + (y)*(y) + (z)*(z) )

/**
 * @brief Computes the pairwise distances between a particle and each closest image of other particles and stores the result in an upper triangula buffer (p1 major)
 * @param coordinates the coordinates of the particles. Expected length: 3*n_particles
 * @param pairs linear vector with the indices of the pairs in the coordinate vector. p1 major, Expected length: (n_particles)*(n_particles-1)/2
 * @param n_particles the number of particles
 * @param L the size of the simulation box
 * @param dxyz Upper Triangular Buffer where to store the distances along the axis. p1 major, Expected length: 3*(n_particles)*(n_particles-1)/2
 * @param r_utb Upper Triangular Buffer where to store the distances. p1 major, Expected length: (n_particles)*(n_particles-1)/2
 */
__global__ void d_computePairwiseDistancesWithPCB(const float* coordinates, const unsigned int* pairs,
                                                  unsigned int n_particles, float L,
                                                  float* dxyz, float* r_utb);


/**
 * @brief Prepares an array of random states
 * @param states an array of `curandState_t` of size n_seeds
 * @param seed the initial seed
 * @param n_seeds the number of states to initialize
 */
__global__ void setupRNG(curandState_t* states, unsigned long long seed, int n_seeds);