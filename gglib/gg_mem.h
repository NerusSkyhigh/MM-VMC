//
// Created by Guglielmo Grillo on 17/10/25.
//
#pragma once
#include <string.h>

/** @struct DoubleBuffer
 * @brief Double buffer
 */
typedef struct DoubleBuffer {
    double* data;
    double** buffers;
    size_t N;
    short old;
    short new;
} DoubleBuffer;
// [TODO] Consider if it's better to store the pointers in new and old instead of the indices

/**
 *@brief Prepare the double buffer for the next iteration
 * @param db The pointer to the DoubleBuffer to prepare for the next step
 */
#define DOUBLE_BUFFER_NEXT_STEP(db) do { \
    (db)->old  = 1 - (db)->old; \
    (db)->new = 1 - (db)->new; \
} while (0)

/**
 *@brief Clear the writing buffer for accumulation in the next step
 * @param db The pointer to the DoubleBuffer where the next buffer is stored
 */
#define DOUBLE_BUFFER_CLEAR_NEXT(db) do {\
    memset((db)->buffers[(db)->new], 0, sizeof(double)*(db)->N);\
} while (0)

/** @fn initDoubleBuffer(DoubleBuffer* db, int64_t n_elements)
 * @brief init the memory associateds to the double buffer
 * @param db the double buffer to init
 * @param n_elements the number of elements for each buffer
 * @warning does NOT free the struct itself
 */
void initDoubleBuffer(DoubleBuffer* db, const size_t n_elements);

/** @fn freeDoubleBuffer(DoubleBuffer* dba)
 * @brief frees the memory associateds to the double buffer
 * @param db the double buffer to free
 * @warning does NOT free the struct itself
 */
void freeDoubleBuffer(const DoubleBuffer* db);



/**
 * @struct CyclicBuffer
 * @brief Buffer to store the last `capacity` items
 * @param capacity the maximum number of items to store
 */
typedef struct CyclicBuffer {
    size_t capacity;
    size_t i;
    double* data;
} CyclicBuffer;

/**
 *@brief Push a new element in the buffer. If the buffer is full, the oldest item is replaced
 * @param cb The pointer to the CyclicBuffer where to push the element
 * @param e the element to push
 * @warning As the oldest element is replaced, there is not garantee that latest element is the last one
 */
#define CYCLIC_BUFFER_PUSH(cb, e) do { \
    (cb)->data[ (cb)->i % (cb)->capacity] = e; \
} while (0)


/** @fn initCyclicBuffer(DoubleBuffer* dba, int64_t capacity)
 * @brief init the memory associateds to the cyclic buffer
 * @param cb pointer to the cyclic buffer to init
 * @param capacity the capacity of the buffer
 */
void initCyclicBuffer(CyclicBuffer* cb, const size_t capacity);

/** @fn freeCyclicBuffer(DoubleBuffer* dba)
 * @brief frees the memory associateds to the cyclic buffer
 * @param cb the cyclic buffer to free
 * @warning does NOT free the struct itself
 */
void freeCyclicBuffer(CyclicBuffer* cb);
