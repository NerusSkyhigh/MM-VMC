// gg_math.h
// Created by Guglielmo Grillo on 17/10/25.
//
#pragma once
#include <math.h>

/** @file gg_math.h
 *  @brief Implementation of useful functions not available in the standard library
 */


/**  double randn()
 * @brief Generates a gaussian distributed number with average 0 and std 1
 * @warning Relies on the rand() library from stdlib
 */
double randn(void);

/**
 * @brief Computes the norm of a 3D vector form the coordinates
 * @param x first coordinate of the vector
 * @param y second coordinate of the vector
 * @param z third coordinate of the vector
 */
#define NORM(x, y, z) sqrt( (x)*(x) + (y)*(y) + (z)*(z) )

/**
 * @brief Computes the average and the standard deviation of a given vector
 * @param v vector of size @param size that contains the samples
 * @param size the size of the vector @param v
 * @param mean where to save the mean computed
 * @param var where to save the variance computed as $<v^2>-<v>^2$
 */
void computeAveStd(const double* v, size_t size, double* mean, double* var);


/** @fn UPPER_TRIANGULAR_INDEX
 * @brief UPPER_TRIANGULAR_INDEX computes the linear index for an upper triangular scheme. p1-major
 * @param p1 the index of the first element (p1-major)
 * @param p2 the index of the second element
 * @param N the number of possible values for p1/p2
 */
#define UTIDX(p1, p2, N) ( (p1)*(2*(N)-(p1)-1) / 2 + ((p2)-(p1)-1) )

/**
 * @brief Computes the pairwise distances between a particle and each closest image of other particles and stores the result in a upper triangula buffer (p1 major)
 * @param coordinates the cordinates of the particles. Expected length: 3*n_particles
 * @param n_particles the number of particles
 * @param L the size of the simulation box
 * @param r_utb Upper Triangular Buffer where to store the distances
 */
void computePairwiseDistancesWithPCB(const double* coordinates, size_t n_particles, double L, double* r_utb);
