//
// Created by Guglielmo Grillo on 17/10/25.
//
#pragma once
#include <stdint.h>

/** @struct DoubleBufferArena
 * @brief Double buffer
 */
typedef struct DoubleBufferArena {
    double* data;
    double** buffers;
    int64_t N;
    short old;
    short new;
} DoubleBufferArena;
// [TODO] Consider if it's better to store the pointers in new and old instead of the indices

/**
 *@brief Prepare the double buffer for the next iteration
 * @param a The pointer to the DoubleBufferArena to prepare for the next step
 */
#define DOUBLE_BUFFER_NEXT_STEP(a) do { \
    (a)->old  = 1 - (a)->old; \
    (a)->new = 1 - (a)->new; \
    memset((a)->buffers[(a)->new], 0, sizeof(double)*(a)->N);\
} while (0)

/** @fn initDoubleBufferArena(DoubleBufferArena* dba, int64_t n_elements)
 * @brief Initialize a double buffer with n elements (of double type)
 * @param dba the double buffer to initialize
 * @param n_elements the number of elements in each buffer
 */
void initDoubleBufferArena(DoubleBufferArena* dba, int64_t n_elements);

/** @fn freeDoubleBufferArena(DoubleBufferArena* dba)
 * @brief frees the memory associateds to the double buffer
 * @param dba the double buffer to free
 * @warning does NOT free the struct itself
 */
void freeDoubleBufferArena(DoubleBufferArena* dba);
