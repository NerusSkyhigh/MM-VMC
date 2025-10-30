// gg_mem.cu
// Created by Guglielmo Grillo on 17/10/25.
//

#include "gg_cuda_macro.cuh"
#include "gg_mem.cuh"


void d_initDoubleBuffer(DoubleBuffer* db, const unsigned int n_elements) {
    db->N = n_elements;
    CUDA_CHECK( cudaMalloc( (void**) &(db->data), 2*n_elements*sizeof(float)) );
    db->prev = db->data;
    db->next = db->data+n_elements;
}

void d_freeDoubleBuffer(DoubleBuffer* db) {
    CUDA_CHECK( cudaFree(db->data) );
    db->N = 0;
}


void d_initCyclicBuffer(CyclicBuffer* cb, const unsigned int capacity) {
    cb->capacity = capacity;
    CUDA_CHECK( cudaMalloc( (void**) &(cb->data), capacity*sizeof(float)) );
    cb->i = 0;
}

void d_freeCyclicBuffer(CyclicBuffer* cb) {
    CUDA_CHECK( cudaFree(cb->data) );
    cb->capacity = 0;
}
