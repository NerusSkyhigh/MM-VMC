// gg_mem.cuh
// Created by Guglielmo Grillo on 17/10/25.
//
#pragma once
#include <string.h>

#include "gg_cuda_macro.cuh"

/** @struct DoubleBuffer
 * @brief Double buffer
 * @param N Size of a single buffer. Effective length is 2xN
 * @param data All the data in the struct, both old and new values. Total size is 2N
 * @param old Reference to the start of the buffer of old data
 * @param new Reference to the start of the buffer of new data
 * @warning Memory is not cleared on init. Use `DOUBLE_BUFFER_CLEAR_NEXT` and `DOUBLE_BUFFER_CLEAR_PREV`
 */
typedef struct DoubleBuffer {
    unsigned int N;
    float* data;
    float* prev;
    float* next;
} DoubleBuffer;

/**
 *@brief Prepare the double buffer for the next iteration
 * @param db The pointer to the DoubleBuffer to prepare for the next step
 */
#define DOUBLE_BUFFER_NEXT_STEP(db) do { \
    float* tmp = (db)->prev; \
    (db)->prev  = (db)->next; \
    (db)->next = tmp; \
} while (0)

/**
 *@brief Clear the writing buffer for accumulation in the next step
 * @param db The pointer to the DoubleBuffer where the next buffer is stored
 */
#define DOUBLE_BUFFER_CLEAR_NEXT(db) do {\
    CUDA_CHECK( cudaMemset((db)->next, 0, sizeof(float)*(db)->N) );\
} while (0)

/**
 *@brief Clear the prev buffer
 * @param db The pointer to the DoubleBuffer where the prev buffer is stored
 */
#define DOUBLE_BUFFER_CLEAR_PREV(db) do {\
    CUDA_CHECK( cudaMemset((db)->prev, 0, sizeof(float)*(db)->N) );\
} while (0)

/** @fn initDoubleBuffer(DoubleBuffer* db, int64_t n_elements)
 * @brief init the memory associateds to the double buffer
 * @param db the double buffer to init
 * @param n_elements the number of elements for each buffer
 * @warning Memory is not cleared on init. Use `DOUBLE_BUFFER_CLEAR_NEXT` and `DOUBLE_BUFFER_CLEAR_PREV`
 */
void d_initDoubleBuffer(DoubleBuffer* db, const unsigned int n_elements);

/** @fn freeDoubleBuffer(DoubleBuffer* dba)
 * @brief frees the memory associateds to the double buffer
 * @param db the double buffer to free
 * @warning does NOT free the struct itself
 */
void d_freeDoubleBuffer(DoubleBuffer* db);



/**
 * @struct CyclicBuffer
 * @brief Buffer to store the last `capacity` items
 * @param capacity the maximum number of items to store
 */
typedef struct CyclicBuffer {
    unsigned int capacity;
    unsigned int i;
    float* data;
} CyclicBuffer;

/**
 *@brief Push a new element into the buffer. If the buffer is full, the oldest item is replaced
 * @param cb The pointer to the CyclicBuffer where to push the element
 * @param e the element to push
 * @warning As the oldest element is replaced, there is not garantee that latest element is the last one
 */
#define CYCLIC_BUFFER_PUSH(cb, e) do { \
    (cb)->data[ (cb)->i % (cb)->capacity] = e; \
    (cb)->i++; \
} while (0)


/** @fn initCyclicBuffer(DoubleBuffer* dba, int64_t capacity)
 * @brief init the memory associateds to the cyclic buffer
 * @param cb pointer to the cyclic buffer to init
 * @param capacity the capacity of the buffer
 */
void d_initCyclicBuffer(CyclicBuffer* cb, const unsigned int capacity);
void h_initCyclicBuffer(CyclicBuffer* cb, const unsigned int capacity);

/** @fn freeCyclicBuffer(DoubleBuffer* dba)
 * @brief frees the memory associateds to the cyclic buffer
 * @param cb the cyclic buffer to free
 * @warning does NOT free the struct itself
 */
void d_freeCyclicBuffer(CyclicBuffer* cb);
