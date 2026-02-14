#ifndef MATRIX_DYN_C
#define MATRIX_DYN_C

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "arena.c" 
#include "matrices/matrix.c" 

nn_mat nn_mdyn_make_mat(NN_Arena *arena, size_t row, size_t col, float *es);
nn_mat nn_mdyn_make_out(NN_Arena *arena,nn_mat *m1, nn_mat *m2);
float *nn_mdyn_make_float_array(NN_Arena *arena,int count, ...);
nn_mat nn_mdyn_mul(NN_Arena *arena, nn_mat *a, nn_mat *b);
nn_mat nn_mdyn_transpose(NN_Arena *arena, nn_mat *a);
nn_mat nn_mdyn_add(NN_Arena *arena, nn_mat *a, nn_mat *b);
nn_mat nn_mdyn_sub(NN_Arena *arena, nn_mat *a, nn_mat *b);
nn_mat nn_mdyn_hadamard(NN_Arena *arena, nn_mat *a, nn_mat *b);
nn_mat nn_mdyn_slice(NN_Arena *arena, nn_mat *a, size_t row1, size_t row2, size_t col1, size_t col2);
nn_mat nn_mdyn_make_randomly_filled_mat(NN_Arena *arena, size_t rows, size_t cols);
nn_mat nn_mdyn_make_zero_filled_mat(NN_Arena *arena, size_t rows, size_t cols);

static inline float get_randf();
static inline float zero();

#ifdef MATRIX_DYN_IMPLEMENTATION

static inline float get_randf()
{
    return (float) rand() / (float)RAND_MAX;
}

static inline float zero()
{
    return 0.0f; 
}

nn_mat nn_mdyn_make_mat(NN_Arena *arena, size_t row, size_t col, float *es)
{
    nn_mat mat = {0};
    float *tbl = (float*) nn_arena_alloc(arena, sizeof(float) * row * col);
    NN_ASSERT(tbl != NULL, "Buy new computer lmao");
    memcpy(tbl, es, sizeof(float) * row * col);
    nn_mat_init(&mat, row, col, tbl);
    return mat;
}


float *make_float_array(NN_Arena *arena,int count, ...) {
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

nn_mat nn_mdyn_mul(NN_Arena *arena, nn_mat *a, nn_mat *b)
{
    nn_mat m = nn_mdyn_make_out(arena, a, b);
    nn_mat_mul(a, b, &m); 
    return m;
}

nn_mat nn_mdyn_make_out(NN_Arena *arena, nn_mat *m1, nn_mat *m2)
{
    NN_ASSERT(m1 != NULL, "m1 is NULL");
    NN_ASSERT(m2 != NULL, "m2 is NULL");
    size_t r = m1->rows;
    size_t c = m2->cols;
    return nn_mdyn_make_mat(arena,r, c, (float[]){0});
}


nn_mat nn_mdyn_transpose(NN_Arena *arena, nn_mat *a)
{
    nn_mat b = nn_mdyn_make_mat(arena, a->cols, a->rows, a->es);
    nn_mat_transpose(a, &b);
    return b;
}

nn_mat nn_mdyn_add(NN_Arena *arena, nn_mat *a, nn_mat *b)
{
    nn_mat m = nn_mdyn_make_mat(arena, a->rows, a->cols, a->es);
    nn_mat_add(a, b, &m); 
    return m;
}

nn_mat nn_mdyn_sub(NN_Arena *arena, nn_mat *a, nn_mat *b)
{
    nn_mat m = nn_mdyn_make_mat(arena, a->rows, a->cols, a->es);
    nn_mat_sub(a, b, &m); 
    return m;
}

nn_mat nn_mdyn_hadamard(NN_Arena *arena, nn_mat *a, nn_mat *b)
{
    nn_mat m = nn_mdyn_make_mat(arena, a->rows, a->cols, a->es);
    nn_mat_hadamard(a, b, &m); 
    return m;
}

nn_mat nn_mdyn_slice(NN_Arena *arena, nn_mat *a, size_t row1, size_t row2, size_t col1, size_t col2)
{
    size_t rows = row2 - row1;
    size_t cols = col2 - col1;
    nn_mat m = nn_mdyn_make_mat(arena, rows, cols, nn_arena_alloc(arena, rows * cols * sizeof(float)));
    nn_mat_slice(a, row1, row2, col1, col2, &m);
    return m;
}

nn_mat nn_mdyn_make_randomly_filled_mat(NN_Arena *arena, size_t rows, size_t cols)
{
    nn_mat mat = {0};
    float *es = (float *) nn_arena_alloc(arena, rows * cols * sizeof(float));
    nn_mat_init(&mat, rows, cols, es);
    nn_mat_fill_func(&mat, get_randf);
    return mat;
}

nn_mat nn_mdyn_make_zero_filled_mat(NN_Arena *arena, size_t rows, size_t cols)
{
    nn_mat mat = {0};
    float *es = (float *) nn_arena_alloc(arena, rows * cols * sizeof(float));
    nn_mat_init(&mat, rows, cols, es);
    nn_mat_fill_func(&mat, zero);
    return mat;
}

#endif // MATRIX_DYN_IMPLEMENTATION 
#endif // MATRIX_DYN_C
