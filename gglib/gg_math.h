//
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
