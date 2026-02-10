#ifndef NN_H
#define NN_H

#include <stddef.h>
#include "matrices/matrix.c"
#include "arena.c" 
#include "activations/activation.h"

typedef struct {
    size_t *arc;
    size_t arc_size;
    nn_mat *ws;
    nn_mat *bs;
    nn_mat *zs;
    nn_mat *as;

    size_t allocated_size;
    void *allocated_block; 
} nn;

void nn_init(nn *model, nn_arena *arena, size_t *arc, size_t arc_size);
void nn_forward_pass(nn *model, Activation *actv, Activation *actv_hidden);
void nn_train(nn *model);
void nn_backprog(nn *model, nn_arena *arena,
                nn_mat *dataset, size_t target_start_col,
                Activation *actv, Activation *actv_hidden);


#ifdef NN_IMPLEMENTATION
#include "nn_impl.inc"

#endif // NN_IMPLEMENTATION
#endif // NN_H

