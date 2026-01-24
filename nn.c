#include <math.h> 
#include <stdlib.h> 
#include <string.h> 
#include "nn.h"
#include "arena.h"
#include "matrix.h"
#include <math.h> 

static inline float get_randf();
static nn_mat make_mat(nn_arena *arena, size_t row, size_t col, float *es);
static nn_mat make_out(nn_arena *arena,nn_mat *m1, nn_mat *m2);
static float sigmoidf(float x);
static float ReLUf(float x);
static float loss_mse(nn_mat *dataset);

static nn_mat make_randomly_filled_mat(nn_arena *arena, size_t rows, size_t cols)
{
    nn_mat mat = {0};
    float *es = (float *) nn_arena_alloc(arena, rows * cols * sizeof(float));
    nn_mat_init(&mat, rows, cols, es);
    nn_mat_fill_func(&mat, get_randf);
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
            as[i] = make_randomly_filled_mat(arena, 1, size);
            zs[i] = make_randomly_filled_mat(arena, 1, size);
        } else {
            size_t size = arc[i];
            nn_mat weight = make_randomly_filled_mat(arena, as[i - 1].cols, size);
            nn_mat bias = make_randomly_filled_mat(arena, as[i - 1].rows, weight.cols);
            nn_mat a = make_randomly_filled_mat(arena, as[i - 1].rows, weight.cols);
            nn_mat z = make_randomly_filled_mat(arena, as[i - 1].rows, weight.cols);
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

}

void nn_train(nn *model)
{

}

void nn_backprog(nn *model)
{

}

static inline float get_randf()
{
    return (float)rand() / (float)RAND_MAX;
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
