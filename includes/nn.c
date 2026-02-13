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
} NN;

typedef struct {
    nn_mat w;
    nn_mat b;
    nn_mat z;
    nn_mat a;
    size_t nodes;
    NN_Activation *activation;
} NN_Layer;

typedef struct {
    size_t count; 
    size_t capacity; 
    NN_Layer *items; 
} NN_Layers;

typedef struct {
   size_t allocated_size;
   NN_Layers layers;
   void *allocated_block; 
} NN_Layered;

/* Regular NN */
void nn_init(NN *model, nn_arena *arena, size_t *arc, size_t arc_size);
void nn_add_layer(NN *model, NN_Activation *actv, NN_Activation *actv_hidden);
void nn_forward_pass(NN *model, NN_Activation *actv, NN_Activation *actv_hidden);
void nn_backprog(NN *model, nn_arena *arena, nn_mat *dataset, size_t target_start_col, NN_Activation *actv, NN_Activation *actv_hidden);

/* Layered NN */
void nn_layered_init(NN_Layered *model);
void nn_layered_add_layer(nn_arena *arena, NN_Layered *model, size_t nodes, NN_Activation *activation);
void nn_layered_forward_pass(NN_Layered *model);
void nn_layered_backprog(nn_arena *arena, NN_Layered *model, nn_mat *dataset, size_t target_start_col);

#ifdef NN_IMPLEMENTATION

#include "nn_impl.inc"
#include "nn_layered_impl.inc"

#endif // NN_IMPLEMENTATION
#endif // NN_H
