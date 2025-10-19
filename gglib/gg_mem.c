//
// Created by Guglielmo Grillo on 17/10/25.
//

#include "gg_mem.h"

#include <stdlib.h>


void initDoubleBufferArena(DoubleBufferArena* dba, int64_t n_elements) {
    dba->N = n_elements;
    dba->data = malloc(sizeof(double)*2*n_elements);

    dba->buffers = malloc(sizeof(double*)*2);

    dba->buffers[0] = &(dba->data[0]);
    dba->old = 0;

    dba->buffers[1] = &(dba->data[n_elements]);
    dba->new = 1;
}

void freeDoubleBufferArena(DoubleBufferArena* dba) {
    free(dba->data);
    free(dba->buffers);
}

