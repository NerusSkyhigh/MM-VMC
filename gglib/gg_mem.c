// gg_mem.c
// Created by Guglielmo Grillo on 17/10/25.
//

#include "gg_mem.h"

#include <stdlib.h>


void initDoubleBuffer(DoubleBuffer* db, const size_t n_elements) {
    db->N = n_elements;
    db->data = malloc(sizeof(double)*2*n_elements);
    db->prev = &(db->data[0]);
    db->next = &(db->data[n_elements]);
}

void freeDoubleBuffer(DoubleBuffer* db) {
    free(db->data);
    db->N = 0;
}


void initCyclicBuffer(CyclicBuffer* cb, const size_t capacity) {
    cb->data = malloc(sizeof(double)*capacity);
    cb->capacity = capacity;
    cb->i = 0;
}

void freeCyclicBuffer(CyclicBuffer* cb) {
    free(cb->data);
    cb->capacity = 0;
}
