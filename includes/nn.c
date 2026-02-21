#ifndef NN_H
#define NN_H

#include <stdlib.h> 
#include <string.h> 
#include <stdarg.h> 
#include <stddef.h>

#include "arena.c" 
#include "activations/activation.h"
#include "matrices/matrix.c"
#include "matrices/matrix_dyn.c"
#include "nn.c"
#include "utils/dynamic_array.h"
#include "utils/nn_assert.h" 

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
} NN;

void nn_init(NN *model);
void nn_add_layer(NN_Arena *arena, NN *model, size_t nodes, NN_Activation *activation);
void nn_add_predefined_layer(NN *model, NN_Layer layer);
void nn_forward_pass(NN *model);
void nn_backprog(NN_Arena *arena, NN *model, nn_mat *dataset, size_t target_start_col);
void nn_dump(NN *model, const char *path);

#ifdef NN_IMPLEMENTATION

void nn_init(NN *model)
{
    NN_Layers layers = {0};
    model->layers = layers;
}

void nn_dump(NN *model, const char *path)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        perror("failed to open file");
        return;
    };

    fprintf(fp, "#include \"matrices/matrix_dyn.c\"\n");
    fprintf(fp, "#include \"matrices/matrix.c\"\n");
    da_foreach(NN_Layer, layer, &model->layers) {
        size_t i = layer - model->layers.items;
        // fprintf(fp, "activation_func %s\n", layer->activation->ops->get_name());
        fprintf(fp, "nn_mat a%zu = nn_mdyn_make_mat(&arena, %zu, %zu, (float [])", i, layer->a.rows, layer->a.cols);
        nn_mat_fprintf(&layer->a, fp); 
        fprintf(fp, ";\n", i);
        fprintf(fp, "z%zu[] = ", i);
        fprintf(fp, "nn_mat z%zu = nn_mdyn_make_mat(&arena, %zu, %zu, (float [])", i, layer->z.rows, layer->z.cols);
        nn_mat_fprintf(&layer->z, fp);
        fprintf(fp, ");\n");
        fprintf(fp, "float b%zu[] = ", i);
        fprintf(fp, "nn_mat b%zu = nn_mdyn_make_mat(&arena, %zu, %zu, (float [])", i, layer->b.rows, layer->b.cols);
        nn_mat_fprintf(&layer->b, fp);
        fprintf(fp, ");\n");
        fprintf(fp, "float w%zu[] = ", i);
        fprintf(fp, "nn_mat w%zu = nn_mdyn_make_mat(&arena, %zu, %zu, (float [])", i, layer->w.rows, layer->w.cols);
        nn_mat_fprintf(&layer->w, fp);
        fprintf(fp, ");\n");
        fprintf(fp, "NN_Layer layer%d = { .a = a%d, .w = w%d, .z = z%d, .b = b%d, .nodes = 2, .activation = relu.actv, };\n", i, i, i, i, i);
        fprintf(fp, "nn_add_predefined_layer(&model, layer%d);\n", i);
    }
    fclose(fp);
}

static void copy_matrix(nn_mat *dst, nn_mat *src)
{
    dst->cols = src->cols;
    dst->rows = src->rows;
    memcpy(dst->es, src->es, src->cols * src->rows * sizeof(float));
}

void nn_add_layer(NN_Arena *arena, NN *model, size_t nodes, NN_Activation *activation)
{
    nn_mat a;
    nn_mat z;
    nn_mat w;
    nn_mat b;

    NN_Layer new_layer = {0};
    if (model->layers.count == 0) {
        size_t rows = 1;
        size_t cols = nodes;

        a = nn_mdyn_make_zero_filled_mat(arena, rows, cols);
        z = nn_mdyn_make_zero_filled_mat(arena, rows, cols);
        w = nn_mdyn_make_zero_filled_mat(arena, rows, cols);
        b = nn_mdyn_make_zero_filled_mat(arena, rows, cols);
    } else {
        NN_Layer last_layer = da_last(&model->layers);
        size_t rows = 1;
        size_t cols = nodes;

        a = nn_mdyn_make_zero_filled_mat(arena, rows, cols);
        z = nn_mdyn_make_zero_filled_mat(arena, rows, cols);
        w = nn_mdyn_make_randomly_filled_mat(arena, last_layer.a.cols, nodes);
        b = nn_mdyn_make_randomly_filled_mat(arena, rows, cols);
    }
    new_layer.a = a;
    new_layer.z = z;
    new_layer.w = w;
    new_layer.b = b;
    new_layer.nodes = nodes;
    new_layer.activation = activation;
    da_append(&model->layers, new_layer);
}

void nn_add_predefined_layer(NN *model, NN_Layer layer)
{
    if (model->layers.count > 0) {
        NN_Layer prev_layer = da_last(&model->layers);
        NN_ASSERT(layer.a.cols == layer.z.cols, "INVALID SIZE: A != Z");
        NN_ASSERT(prev_layer.w.cols == layer.w.rows, "INVALID SIZE: W(L-1) != W(L)");
        NN_ASSERT(layer.w.cols == layer.a.cols, "INVALID SIZE: W != A");
    }
    da_append(&model->layers, layer);
}

void nn_forward_pass(NN *model)
{
    for (size_t i = 1; i < model->layers.count; ++i) {
        nn_mat_mul(&model->layers.items[i - 1].a, &model->layers.items[i].w, &model->layers.items[i].z);
        nn_mat_add(&model->layers.items[i].z, &model->layers.items[i].b, &model->layers.items[i].z);
        nn_mat_map(&model->layers.items[i].z, model->layers.items[i].activation->ops->regular(model->layers.items[i].activation), &model->layers.items[i].a);
    }
}

void nn_backprog(NN_Arena *arena, NN *model, nn_mat *dataset, size_t target_start_col) 
{
    NN_Layer *layers = model->layers.items;
    size_t state = arena->count;
    size_t arcsz = model->layers.count; 
    size_t max_layer = layers[0].a.cols;
    nn_mat *input = &layers[0].a;
    NN_Layer last_layer = da_last(&model->layers);
    nn_mat *output = &last_layer.a;

    for (size_t i = 1; i < arcsz; ++i) {
        if (layers[i].a.cols > max_layer) max_layer = layers[i].a.cols;
    }

    nn_mat dc_dz = nn_mdyn_make_zero_filled_mat(arena, 1, max_layer);
    for (size_t i = 0; i < dataset->rows; ++i) {
        /* 
         * Slice the input columns from the dataset and then put it
         * inside the first layer (the input layer)
         */
        nn_mat dataset_row = nn_mdyn_slice(arena, dataset, i, i + 1, 0, target_start_col);
        memcpy(input->es, dataset_row.es, dataset_row.cols * dataset_row.rows * sizeof(float));

         /* slice the target columns from the dataset */
        nn_mat target = nn_mdyn_slice(arena, dataset, i, i + 1, target_start_col, dataset->cols);

        nn_forward_pass(model);

        /*
         * First step, calculate the dC/daL, and then calculate the
         * dc/dzL = aL - aL * aL
         */ 

        /* Finding da/dz 
         * 2/n (y - x)
         */
        nn_mat dc_da = nn_mdyn_sub(arena, output, &target);
        nn_mat_mul_scalar(&dc_da, (2.0f/(float) arcsz), &dc_da);

        /* Finding the initial da/dz 
         * d_activation(z)
         */
        nn_mat zs_output = model->layers.items[arcsz - 1].z;
        nn_mat da_dz = nn_mdyn_make_mat(arena, zs_output.rows, zs_output.cols, zs_output.es);
        nn_mat_map(&zs_output, last_layer.activation->ops->derived(last_layer.activation), &da_dz);

        /* Finding the initial dc/dz */
        nn_mat temp_dc_dz = nn_mdyn_hadamard(arena, &da_dz, &dc_da);
        copy_matrix(&dc_dz, &temp_dc_dz);

        float lr = 0.5;
        for (size_t k = 1; k < arcsz; ++k) {
            // Memory Arena Checkpoint
            size_t layer_state = arena->count; 
            NN_Layer curr_layer = model->layers.items[(arcsz - k) - 1];
            nn_mat al_1 = model->layers.items[(arcsz - k) - 1].a;
            nn_mat al_1_t = nn_mdyn_transpose(arena, &al_1);

            /* Calculate dc/dw */
            nn_mat dc_dw = nn_mdyn_mul(arena, &al_1_t, &dc_dz);
            dc_dw = nn_mdyn_transpose(arena, &dc_dw);
            
            nn_mat wl = model->layers.items[arcsz - k].w;
            nn_mat *bl = &model->layers.items[arcsz - k].b;
            nn_mat wl_t = nn_mdyn_transpose(arena, &wl);
            
            /* Updating the bias */
            {
                nn_mat temp = nn_mdyn_make_mat(arena, dc_dz.rows, dc_dz.cols, dc_dz.es);
                nn_mat_mul_scalar(&temp, lr, &temp);
                nn_mat_sub(bl, &dc_dz, bl);
            }
            
            nn_mat_mul_scalar(&dc_dw, lr,&dc_dw);
            /* TODO: Figure out a better way to update the weights */
            nn_mat tmp = nn_mdyn_make_randomly_filled_mat(arena, wl_t.rows, wl_t.cols);
            for (size_t j = 0; j < wl_t.rows; ++j) {
                memcpy(tmp.es + (j * dc_dw.cols * dc_dw.rows), dc_dw.es, dc_dw.rows * dc_dw.cols * sizeof(float));
            }
           
            /*
             * Calculate the new dc_dz for the previous layer
             */
            nn_mat new_wl_t = nn_mdyn_sub(arena, &wl_t, &tmp);
            nn_mat new_wl = nn_mdyn_transpose(arena, &new_wl_t);
            memcpy(model->layers.items[arcsz - k].w.es, new_wl.es, new_wl.cols * new_wl.rows * sizeof(float));

            nn_mat zl_1 = model->layers.items[(arcsz - k) - 1].z;
            nn_mat dz_dzl_1 = nn_mdyn_make_mat(arena, zl_1.rows, zl_1.cols, zl_1.es);
            nn_mat_map(&zl_1, curr_layer.activation->ops->derived(curr_layer.activation), &dz_dzl_1);

            nn_mat tmp_dc_dz = nn_mdyn_mul(arena, &dc_dz, &wl_t);
            tmp_dc_dz = nn_mdyn_hadamard(arena, &tmp_dc_dz, &dz_dzl_1);
            copy_matrix(&dc_dz, &tmp_dc_dz);
            
            nn_arena_reset_to(arena, layer_state);
        }
    }
    nn_arena_reset_to(arena, state);
}

#endif // NN_IMPLEMENTATION
#endif // NN_H
