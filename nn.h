#ifndef NN_H
#define NN_H

#include <stddef.h>
#include "matrix.h"
#include "arena.h" 

typedef struct {
    size_t *arc;
    size_t arc_size;
    nn_mat *ws;
    nn_mat *bs;
    nn_mat *zs;
    nn_mat *as;
} nn;

typedef struct {
    nn_arena arena;
    float (*activation)(float);
    float learning_rate;
} nn_config;

void nn_init(nn *model, nn_arena *arena, size_t *arc, size_t arc_size);
void nn_forward_pass(nn *model);
void nn_train(nn *model);
void nn_backprog(nn *model);

#ifdef NN_IMPLEMENTATION
#include "nn.c"
#endif // NN_IMPLEMENTATION

#endif // NN_H

