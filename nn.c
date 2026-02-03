#include <math.h> 
#include <stdlib.h> 
#include <string.h> 
#include "nn.h"
#include "arena.h"
#include "matrix.h"
#include <math.h> 
#include <stdarg.h> 
#include "nn_assert.h" 

static inline float get_randf();
static nn_mat make_mat(nn_arena *arena, size_t row, size_t col, float *es);
static nn_mat make_out(nn_arena *arena,nn_mat *m1, nn_mat *m2);
static float sigmoidf(float x);
static float ReLUf(float x);
static float loss_mse(nn_mat *dataset);
static float zero();

static float zero() {
    return 0.0f;
}

float *make_float_array(nn_arena *arena,int count, ...) {
    va_list args;
    va_start(args, count);
    float *es = nn_arena_alloc(arena, count * sizeof(float));
    NN_ASSERT(es != NULL, "Buy more RAM lol");
    for (int i = 0; i < count; i++) {
        double d = va_arg(args, double); // ALWAYS double
        es[i] = (float) d; 
    }

    va_end(args);
    return es;
}

static nn_mat mul(nn_arena *arena, nn_mat *a, nn_mat *b)
{
    nn_mat m = make_out(arena, a, b);
    nn_mat_mul(a, b, &m); 
    return m;
}

static nn_mat transpose(nn_arena *arena, nn_mat *a)
{
    nn_mat b = make_mat(arena, a->cols, a->rows, a->es);
    nn_mat_transpose(a, &b);
    return b;
}

static nn_mat add(nn_arena *arena, nn_mat *a, nn_mat *b)
{
    nn_mat m = make_mat(arena, a->rows, a->cols, a->es);
    nn_mat_add(a, b, &m); 
    return m;
}

static nn_mat sub(nn_arena *arena, nn_mat *a, nn_mat *b)
{
    nn_mat m = make_mat(arena, a->rows, a->cols, a->es);
    nn_mat_sub(a, b, &m); 
    return m;
}

static nn_mat hdmrt(nn_arena *arena, nn_mat *a, nn_mat *b)
{
    nn_mat m = make_mat(arena, a->rows, a->cols, a->es);
    nn_mat_hdmrt(a, b, &m); 
    return m;
}

static nn_mat make_randomly_filled_mat(nn_arena *arena, size_t rows, size_t cols)
{
    nn_mat mat = {0};
    float *es = (float *) nn_arena_alloc(arena, rows * cols * sizeof(float));
    nn_mat_init(&mat, rows, cols, es);
    nn_mat_fill_func(&mat, get_randf);
    return mat;
}

static nn_mat make_zero_filled_mat(nn_arena *arena, size_t rows, size_t cols)
{
    nn_mat mat = {0};
    float *es = (float *) nn_arena_alloc(arena, rows * cols * sizeof(float));
    nn_mat_init(&mat, rows, cols, es);
    nn_mat_fill_func(&mat, zero);
    return mat;
}

void nn_init(nn *model, nn_arena *arena, size_t *arc, size_t arc_size)
{
    nn_mat *ws = nn_arena_alloc(arena, (arc_size - 1) * sizeof(nn_mat));
    nn_mat *bs = nn_arena_alloc(arena, (arc_size - 1) * sizeof(nn_mat));
    nn_mat *os = nn_arena_alloc(arena, (arc_size - 1) * sizeof(nn_mat));
    nn_mat *zs = nn_arena_alloc(arena, arc_size * sizeof(nn_mat));
    nn_mat *as = nn_arena_alloc(arena, arc_size * sizeof(nn_mat));
    for (size_t i = 0; i < arc_size; ++i) {
        if (i == 0) {
            size_t size = arc[i];
            // input layer 
            as[i] = make_zero_filled_mat(arena, 1, size);
            zs[i] = make_zero_filled_mat(arena, 1, size);
        } else {
            size_t size = arc[i];
            nn_mat weight = make_randomly_filled_mat(arena, as[i - 1].cols, size);
            nn_mat bias = make_randomly_filled_mat(arena, as[i - 1].rows, weight.cols);
            nn_mat a = make_zero_filled_mat(arena, as[i - 1].rows, weight.cols);
            nn_mat z = make_zero_filled_mat(arena, as[i - 1].rows, weight.cols);
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
    model->arc = arc;
    model->arc_size = arc_size;
}

void nn_forward_pass(nn *model)
{
    for (size_t i = 1; i < model->arc_size; ++i) {
        nn_mat_mul(&model->as[i - 1], &model->ws[i], &model->zs[i]);
        nn_mat_add(&model->zs[i], &model->bs[i], &model->zs[i]);
        nn_mat_map(&model->zs[i], &sigmoidf, &model->as[i]);
    }
}

void nn_train(nn *model)
{

}

/* Need Dataset to get the target */
void nn_backprog(nn *model, nn_arena *arena)
{
    // note: dataset is still hardcoded in the function.
    // The 3rd column is for the target. The 4th column is used for
    // storing the result after the forward pass.
    // TODO: fix the implementation of this function to be dataset
    // agnostic.
    nn_mat dataset = {0};
    size_t state = arena->count;
    const size_t ROWS = 4; 
    const size_t COLS = 4; 
    float templ[] = {
        1, 1, 1, 0,
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0, 0,
    };
    float *es = nn_arena_alloc(arena, ROWS * COLS * sizeof(float));
    memcpy(es, templ, ROWS * COLS * sizeof(float));
    nn_mat_init(&dataset, ROWS, COLS, es);
    size_t arcsz = model->arc_size; 
    nn_mat *input = &model->as[0];
    for (size_t i = 0; i < ROWS; ++i) {
        NN_MAT_AT(input, 0, 0) = NN_MAT_AT(&dataset, i, 0);
        NN_MAT_AT(input, 0, 1) = NN_MAT_AT(&dataset, i, 1);
        nn_forward_pass(model);

        nn_mat output = model->as[arcsz - 1];
        nn_mat target = make_mat(arena, 1, 1, (float[]){0});

        /* Add target to the 3rd column */
        NN_MAT_AT(&target, 0, 0) = NN_MAT_AT(&dataset, i, 2);
        
        /*
         * First step, calculate the dC/daL, and then calculate the
         * dc/dzL = aL - aL * aL
         */ 
        nn_mat dc_da = sub(arena, &output, &target);
        nn_mat_mul_scalar(&dc_da, (2.0f/(float) arcsz), &dc_da);

        nn_mat dc_dz = hdmrt(arena, &output, &output);
        dc_dz = sub(arena, &output, &dc_dz);
        dc_dz = hdmrt(arena, &dc_dz, &dc_da);

        float lr = 0.01;        
        for (size_t i = 1; i < arcsz; ++i) {

            nn_mat al_1 = model->as[(arcsz - i) - 1];
            nn_mat al_1_t = transpose(arena, &al_1);

            /* Calculate dc/dw */
            nn_mat dc_dw = mul(arena, &al_1_t, &dc_dz);
            dc_dw = transpose(arena, &dc_dw);
            
            nn_mat wl = model->ws[arcsz - i];
            nn_mat *bl = &model->bs[arcsz - i];
            nn_mat wl_t = transpose(arena, &wl);
            nn_mat_sub(bl, &dc_dz, bl);
            nn_mat_mul_scalar(&dc_dw, lr,&dc_dw);
            
            /* TODO: Figure out a better way to update the weights */
            nn_mat tmp = make_randomly_filled_mat(arena, wl_t.rows, wl_t.cols);
            for (size_t j = 0; j < wl_t.rows; ++j) {
                memcpy(tmp.es + (j * dc_dw.cols * dc_dw.rows), dc_dw.es, dc_dw.rows * dc_dw.cols * sizeof(float));
            }
           
            /*
             * Calculate the new dc_dz for the previous layer
             */
            nn_mat new_wl_t = sub(arena, &wl_t, &tmp);
            wl = transpose(arena, &new_wl_t);
            memcpy(model->ws[arcsz - i].es, wl.es, wl.cols * wl.rows * sizeof(float));

            tmp = hdmrt(arena, &al_1, &al_1);
            tmp = sub(arena, &al_1, &tmp);

            dc_dz = mul(arena, &dc_dz, &new_wl_t);
            dc_dz = hdmrt(arena, &dc_dz, &tmp);
            
        }
    }
    nn_arena_reset_to(arena, state);
}

static inline float get_randf()
{
    return (float) rand() / (float)RAND_MAX;
}

static nn_mat make_mat(nn_arena *arena, size_t row, size_t col, float *es)
{
    nn_mat mat = {0};
    float *tbl = (float*) nn_arena_alloc(arena, sizeof(float) * row * col);
    NN_ASSERT(tbl != NULL, "Buy new computer lmao");
    memcpy(tbl, es, sizeof(float) * row * col);
    nn_mat_init(&mat, row, col, tbl);
    return mat;
}

static nn_mat make_out(nn_arena *arena, nn_mat *m1, nn_mat *m2)
{
    NN_ASSERT(m1 != NULL, "m1 is NULL");
    NN_ASSERT(m2 != NULL, "m2 is NULL");
    size_t r = m1->rows;
    size_t c = m2->cols;
    return make_mat(arena,r, c, (float[]){0});
}

static float sigmoidf(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

static float ReLUf(float x)
{
    return fmax(0, x);
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
