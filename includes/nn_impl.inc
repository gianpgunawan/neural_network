#ifndef NN_IMPL_INC
#define NN_IMPL_INC

#include <math.h> 
#include <stdlib.h> 
#include <string.h> 
#include <stdarg.h> 

#include "activations/activation.h"
#include "arena.c"
#include "matrices/matrix.c"
#include "matrices/matrix_dyn.c"
#include "nn.c"
#include "utils/nn_assert.h" 

static float loss_mse(nn_mat *dataset);

/*
 * Initialize a neural network model specified by the struct `nn`
 * Instead of using malloc, this function receives an arena and all
 * the allocated memory is continuos and the first pointer of the
 * allocated block is saved inside the `allocated_block` field with
 * the size of allocated block saved inside the `allocated_size` field
 *
 * preallocated_block looks like this:
 * [ws][bs][zs][as][arc_allocated][elements of w matrices][elements of b matrices][elements of z matrices][elements of a matrices]
 */
void nn_init(NN *model, nn_arena *arena, size_t *arc, size_t arc_size)
{
    size_t oldsz = arena->count; 
    nn_mat *ws = nn_arena_alloc(arena, (arc_size) * sizeof(nn_mat));
    nn_mat *bs = nn_arena_alloc(arena, (arc_size) * sizeof(nn_mat));
    nn_mat *zs = nn_arena_alloc(arena, arc_size * sizeof(nn_mat));
    nn_mat *as = nn_arena_alloc(arena, arc_size * sizeof(nn_mat));

    size_t *arc_allocated = nn_arena_alloc(arena, arc_size * sizeof(size_t));

    memcpy(arc_allocated, arc, arc_size * sizeof(size_t));
    nn_arena warena = {0}; 
    nn_arena barena = {0}; 
    nn_arena zarena = {0}; 
    nn_arena aarena = {0}; 

    for (size_t i = 0; i < arc_size; ++i) {
        if (i == 0) {
            size_t size = arc[i];
            aarena.capacity += (size * sizeof(float));
            zarena.capacity += (size * sizeof(float));
        } else {
            size_t size = arc[i];
            warena.capacity += arc[i - 1] * size * sizeof(float);
            barena.capacity += size * sizeof(float);
            aarena.capacity += size * sizeof(float);
            zarena.capacity += size * sizeof(float);
        }
    }
   
    nn_arena_init_from_block(&warena, nn_arena_alloc(arena, warena.capacity), warena.capacity);
    nn_arena_init_from_block(&barena, nn_arena_alloc(arena, barena.capacity), barena.capacity);
    nn_arena_init_from_block(&zarena, nn_arena_alloc(arena, zarena.capacity), zarena.capacity);
    nn_arena_init_from_block(&aarena, nn_arena_alloc(arena, aarena.capacity), aarena.capacity);
    

    for (size_t i = 0; i < arc_size; ++i) {
        if (i == 0) {
            size_t size = arc[i];
            as[i] = nn_mdyn_make_zero_filled_mat(&aarena, 1, size);
            zs[i] = nn_mdyn_make_zero_filled_mat(&zarena, 1, size);
        } else {
            size_t size = arc[i];
            nn_mat weight = nn_mdyn_make_randomly_filled_mat(&warena, as[i - 1].cols, size);
            nn_mat bias = nn_mdyn_make_randomly_filled_mat(&barena, as[i - 1].rows, weight.cols);
            nn_mat a = nn_mdyn_make_zero_filled_mat(&aarena, as[i - 1].rows, weight.cols);
            nn_mat z = nn_mdyn_make_zero_filled_mat(&zarena, as[i - 1].rows, weight.cols);
            ws[i] = weight;
            bs[i] = bias;
            zs[i] = z;
            as[i] = a;
        }
    }

    model->ws = ws;
    model->as = as;
    model->zs = zs;
    model->bs = bs;
    model->arc = arc_allocated;
    model->allocated_block = ws;
    model->allocated_size = arena->count - oldsz;
    model->arc_size = arc_size;
}

 // [ws][bs][zs][as][arc_allocated][elements of w matrices][elements of b matrices][elements of z matrices][elements of a matrices]
void nn_load_model(nn_arena *arena, NN *model, const char *path)
{
    size_t oldsz = arena->count; 
    size_t arc_size = model->arc_size;
    size_t *arc = model->arc;
    nn_mat *ws = nn_arena_alloc(arena, (arc_size - 1) * sizeof(nn_mat));
    nn_mat *bs = nn_arena_alloc(arena, (arc_size - 1) * sizeof(nn_mat));
    nn_mat *zs = nn_arena_alloc(arena, arc_size * sizeof(nn_mat));
    nn_mat *as = nn_arena_alloc(arena, arc_size * sizeof(nn_mat));

    size_t *arc_allocated = nn_arena_alloc(arena, arc_size * sizeof(size_t));

    memcpy(arc_allocated, arc, arc_size);
    nn_arena warena = {0}; 
    nn_arena barena = {0}; 
    nn_arena zarena = {0}; 
    nn_arena aarena = {0}; 

    for (size_t i = 0; i < arc_size; ++i) {
        if (i == 0) {
            size_t size = arc[i];
            aarena.capacity += (size * sizeof(float));
            zarena.capacity += (size * sizeof(float));
        } else {
            size_t size = arc[i];
            warena.capacity += arc[i - 1] * size * sizeof(float);
            barena.capacity += size * sizeof(float);
            aarena.capacity += size * sizeof(float);
            zarena.capacity += size * sizeof(float);
        }
    }
   
    nn_arena_init_from_block(&warena, nn_arena_alloc(arena, warena.capacity), warena.capacity);
    nn_arena_init_from_block(&barena, nn_arena_alloc(arena, barena.capacity), barena.capacity);
    nn_arena_init_from_block(&zarena, nn_arena_alloc(arena, zarena.capacity), aarena.capacity);
    nn_arena_init_from_block(&aarena, nn_arena_alloc(arena, aarena.capacity), zarena.capacity);

    for (size_t i = 0; i < arc_size; ++i) {
        if (i == 0) {
            size_t size = arc[i];
            as[i] = nn_mdyn_make_zero_filled_mat(&aarena, 1, size);
            zs[i] = nn_mdyn_make_zero_filled_mat(&zarena, 1, size);
        } else {
            size_t size = arc[i];
            nn_mat weight = nn_mdyn_make_randomly_filled_mat(&warena, as[i - 1].cols, size);
            nn_mat bias = nn_mdyn_make_randomly_filled_mat(&barena, as[i - 1].rows, weight.cols);
            nn_mat a = nn_mdyn_make_zero_filled_mat(&aarena, as[i - 1].rows, weight.cols);
            nn_mat z = nn_mdyn_make_zero_filled_mat(&zarena, as[i - 1].rows, weight.cols);
            ws[i] = weight;
            bs[i] = bias;
            zs[i] = z;
            as[i] = a;
        }
    }
    model->ws = ws;
    model->as = as;
    model->zs = zs;
    model->bs = bs;
    model->arc = arc_allocated;
    model->allocated_block = ws;
    model->allocated_size = arena->count - oldsz;
    model->arc_size = arc_size;
}

void nn_forward_pass(NN *model, NN_Activation *actv_hidden, NN_Activation *actv)
{
    for (size_t i = 1; i < model->arc_size; ++i) {
        nn_mat_mul(&model->as[i - 1], &model->ws[i], &model->zs[i]);
        nn_mat_add(&model->zs[i], &model->bs[i], &model->zs[i]);
        if (i == model->arc_size - 1) {
            nn_mat_map(&model->zs[i], actv->ops->regular, &model->as[i]);
        } else {
            nn_mat_map(&model->zs[i], actv_hidden->ops->regular, &model->as[i]);
        }
    }
}

static void copy_matrix(nn_mat *dst, nn_mat *src)
{
    dst->cols = src->cols;
    dst->rows = src->rows;
    memcpy(dst->es, src->es, src->cols * src->rows * sizeof(float));
}

void nn_backprog(NN *model, nn_arena *arena,
                nn_mat *dataset, size_t target_start_col,
                NN_Activation *actv, NN_Activation *actv_hidden)
{
    size_t state = arena->count;
    size_t arcsz = model->arc_size; 
    size_t max_layer = model->arc[0];
    nn_mat *input = &model->as[0];
    nn_mat *output = &model->as[arcsz - 1];

    for (size_t i = 1; i < arcsz; ++i) {
        if (model->arc[i] > max_layer) max_layer = model->arc[i];
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

        nn_forward_pass(model, actv_hidden, actv);

        /*
         * First step, calculate the dC/daL, and then calculate the
         * dc/dzL = aL - aL * aL
         */ 

        /* Finding da/dz 
         * 2/n (y - x)
         */
        nn_mat dc_da = nn_mdyn_sub(arena,  output, &target);
        nn_mat_mul_scalar(&dc_da, (2.0f/(float) arcsz), &dc_da);

        /* Finding the initial da/dz 
         * d_activation(z)
         */
        nn_mat zs_output = model->zs[arcsz - 1];
        nn_mat da_dz = nn_mdyn_make_mat(arena, zs_output.rows, zs_output.cols, zs_output.es);
        nn_mat_map(&zs_output, actv->ops->derived, &da_dz);

        /* Finding the initial dc/dz */
        nn_mat temp_dc_dz = nn_mdyn_hadamard(arena, &da_dz, &dc_da);
        copy_matrix(&dc_dz, &temp_dc_dz);

        float lr = 0.5;
        for (size_t k = 1; k < arcsz; ++k) {
            // Memory Arena Checkpoint
            size_t layer_state = arena->count; 

            nn_mat al_1 = model->as[(arcsz - k) - 1];
            nn_mat al_1_t = nn_mdyn_transpose(arena, &al_1);

            /* Calculate dc/dw */
            nn_mat dc_dw = nn_mdyn_mul(arena, &al_1_t, &dc_dz);
            dc_dw = nn_mdyn_transpose(arena, &dc_dw);
            
            nn_mat wl = model->ws[arcsz - k];
            nn_mat *bl = &model->bs[arcsz - k];
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
            memcpy(model->ws[arcsz - k].es, new_wl.es, new_wl.cols * new_wl.rows * sizeof(float));

            nn_mat zl_1 = model->zs[(arcsz - k) - 1];
            nn_mat dz_dzl_1 = nn_mdyn_make_mat(arena, zl_1.rows, zl_1.cols, zl_1.es);
            nn_mat_map(&zl_1, actv_hidden->ops->derived, &dz_dzl_1);

            nn_mat tmp_dc_dz = nn_mdyn_mul(arena, &dc_dz, &wl_t);
            tmp_dc_dz = nn_mdyn_hadamard(arena, &tmp_dc_dz, &dz_dzl_1);
            copy_matrix(&dc_dz, &tmp_dc_dz);
            
            nn_arena_reset_to(arena, layer_state);
        }
    }

    nn_arena_reset_to(arena, state);
}

float loss_mse(nn_mat *dataset)
{
    float sum = 0.0;
    for (size_t i = 0; i < dataset->rows; ++i) {
        float target = NN_MAT_AT(dataset, i, 2);
        float result = NN_MAT_AT(dataset, i, 3);
        sum += (target - result) * (target - result);
    }
    return sum / (float) dataset->rows;
}

#endif // NN_IMPL_INC
