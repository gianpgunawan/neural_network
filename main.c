#include <string.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#include "implementations.h"

#include "arena.h"
#include "matrix.h"
#include "nn_assert.h"
#include "nn.h"

bool write_entire_file(const char *path, const void *data, size_t size, const char *mode)
{
    bool result = true;

    const char *buf = NULL;
    FILE *f = fopen(path, mode);
    if (f == NULL) {
        result = false;
        goto defer;
    }

    buf = (const char*)data;
    while (size > 0) {
        size_t n = fwrite(buf, 1, size, f);
        if (ferror(f)) {
            result = false;
            goto defer;
        }
        size -= n;
        buf  += n;
    }

defer:
    if (f) fclose(f);
    return result;
}

bool read_model(nn_arena *arena, const char *path, nn *model)
{
    bool result = true;

    const char *buf = NULL;
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        result = false;
        goto defer;
    }

    if (fseek(f, 0, SEEK_END) < 0) {
        result = false;
        goto defer;
    }

    long size = ftell(f);
    if (fseek(f, 0, SEEK_SET) < 0) {
        result = false;
        goto defer;
    }

    /* First, read only the bare model `nn` */
    nn *modelblock = nn_arena_alloc(arena, sizeof(nn));
    memcpy(model, modelblock, sizeof(nn));

    void *newptr = nn_arena_alloc(arena, size - sizeof(nn));
    size_t n = fread(model, 1, sizeof(nn), f);

    if (n <= 0) {
        result = false;
        goto defer;
    }

    void *oldptr = model->allocated_block;

    /* Second, read all the preallocated block */
    size_t blocksz = size - sizeof(nn);
    while (blocksz > 0) {
        size_t n = fread(newptr, 1, blocksz, f);
        if (ferror(f)) {
            result = false;
            goto defer;
        }
        blocksz -= n;
    }

    model->allocated_block = newptr;
    /* Third, Update all invalid pointers by eliminating the old ones and adding them with the new one */
    
    ptrdiff_t offset = (uint8_t *)(model->ws) - (uint8_t *)(oldptr);

    model->ws   = (nn_mat *)((uint8_t *)newptr + ((uint8_t *)(model->ws)  - (uint8_t *)(oldptr)));
    model->as   = (nn_mat *)((uint8_t *)newptr + ((uint8_t *)(model->as)  - (uint8_t *)(oldptr)));
    model->zs   = (nn_mat *)((uint8_t *)newptr + ((uint8_t *)(model->zs)  - (uint8_t *)(oldptr)));
    model->bs   = (nn_mat *)((uint8_t *)newptr + ((uint8_t *)(model->bs)  - (uint8_t *)(oldptr)));
    model->arc  = (size_t *)((uint8_t *)newptr + ((uint8_t *)(model->arc) - (uint8_t *)(oldptr)));
    
    for (size_t i = 0; i < model->arc_size; ++i) {
        if (i == 0) {
            model->as[i].es = (float *) ((uint8_t *)model->allocated_block + ((uint8_t *)(model->as[i].es) - (uint8_t *)(oldptr)));
            model->zs[i].es = (float *) ((uint8_t *)model->allocated_block + ((uint8_t *)(model->zs[i].es) - (uint8_t *)(oldptr)));
        } else {
            model->as[i].es = (float *) ((uint8_t *)model->allocated_block + ((uint8_t *)(model->as[i].es) - (uint8_t *)(oldptr)));
            model->zs[i].es = (float *) ((uint8_t *)model->allocated_block + ((uint8_t *)(model->zs[i].es) - (uint8_t *)(oldptr)));
            model->bs[i].es = (float *) ((uint8_t *)model->allocated_block + ((uint8_t *)(model->bs[i].es) - (uint8_t *)(oldptr)));
            model->ws[i].es = (float *) ((uint8_t *)model->allocated_block + ((uint8_t *)(model->ws[i].es) - (uint8_t *)(oldptr)));
        }
    }
 
defer:
    if (f) fclose(f);
    return result;
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    nn_arena arena = {0};
    size_t arena_sz = 256 * 1000 * 1000; // 256 MBs
    nn_arena_init(&arena, arena_sz);

    nn model = {0};
    
    if (argc < 3) {
        printf("Input the inputs first");
        return 0;
    }

    bool toggle = 1;
    if (toggle) {
        read_model(&arena, ".\\model", &model);
    } else {
        size_t arc[] = {2, 2, 1};
        size_t arc_len = sizeof(arc) / sizeof(arc[0]);
        nn_init(&model, &arena, arc, arc_len);

        for (size_t i = 0; i < 500000; ++i) {
            nn_backprog(&model, &arena); 
        }
    }
    char *end;
    char *end2;
    float x = strtof(argv[1], &end);
    float y = strtof(argv[2], &end2);
    if (end == argv[1] || end2 == argv[2]) {
        printf("Invalid Input");
        return 1;
    }
    
    NN_MAT_AT(&model.as[0], 0, 0) = x;
    NN_MAT_AT(&model.as[0], 0, 1) = y;
    nn_forward_pass(&model);
    
    printf("\n");
    nn_mat_print(&(model.as[0]));
    nn_mat_print(&(model.as[2]));
    
    // nn_mat_print(&(model.zs[0]));
    // nn_mat_print(&(model.zs[1]));

    if (!toggle) {
        size_t oldpos = arena.count;
        uint8_t *buffer = (uint8_t *)nn_arena_alloc(&arena, sizeof(model) + model.allocated_size);
        memcpy(buffer, &model, sizeof(model));
        memcpy(buffer + sizeof(model), model.allocated_block, model.allocated_size);
        write_entire_file("model", buffer, sizeof(model) + model.allocated_size, "wb");
        nn_arena_reset_to(&arena, oldpos);
    }
    
    return 0;
}

float func()
{
    static int i = 0;
    return ++i;
}

nn_mat slice(nn_arena *arena, nn_mat *m, size_t row1, size_t row2, size_t col1, size_t col2)
{
    nn_mat out;
    out.cols = col2 - col1;
    out.rows = row2 - row1;
    out.es = nn_arena_alloc(arena, out.cols * out.rows * sizeof(float));
    nn_mat_slice(m, row1, row2, col1, col2, &out);
    return out;
}

int main2()
{
    nn_arena arena = {0};
    size_t arena_sz = 256 * 1000 * 1000; // 256 MBs
    nn_arena_init(&arena, arena_sz);

    nn_mat m = {0};
    float *es = nn_arena_alloc(&arena, 12 * sizeof(float));
    nn_mat_init(&m, 4, 3, es);
    // nn_mat_fill_func(&m, func);
    memcpy(es, (float[]) {
                1, 0, 1,
                1, 1, 1,
                0, 1, 1,
                0, 0, 0,
            }, m.cols * m.rows * sizeof(float));

    nn_mat b = slice(&arena, &m, 0, m.rows, m.cols - 1, m.cols);

    // nn_mat_print(&m);
    nn_mat_print(&b);
    printf("ROW %zu\n", b.rows);
    printf("COL %zu\n", b.cols);
    return 0;
}




