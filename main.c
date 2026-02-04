#include <string.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#include "implementations.h"

#include "arena.h"
#include "matrix.h"
#include "nn_assert.h"
#include "nn.h"

bool write_entire_file(const char *path, const void *data, size_t size)
{
    bool result = true;

    const char *buf = NULL;
    FILE *f = fopen(path, "wb");
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

int main2()
{
    srand(time(NULL));
    nn_arena arena = {0};
    nn_arena prime_arena = {0};
    size_t arena_sz = 256 * 1000 * 1000; // 256 MBs
    nn_arena_init(&prime_arena, arena_sz);
    nn_arena_init(&arena, arena_sz);


    nn model = {0};
    size_t arc[] = {2, 2, 1};
    size_t arc_len = sizeof(arc) / sizeof(arc[0]);
    nn_init(&model, &arena, arc, arc_len);
    size_t modelsz = arena.count;

    for (size_t i = 0; i < 100; ++i) {
        nn_backprog(&model, &arena); 
    }

    NN_MAT_AT(&model.as[0], 0, 0) = 0;
    NN_MAT_AT(&model.as[0], 0, 1) = 1;
    
    nn_forward_pass(&model);
    nn_mat_print(&(model.as[0]));
    nn_mat_print(&(model.as[model.arc_size - 1]));

    return 0;
}

float func()
{
    static int i = 0;
    return ++i;
}

int main()
{
    nn_arena arena = {0};
    size_t arena_sz = 256 * 1000 * 1000; // 256 MBs
    nn_arena_init(&arena, arena_sz);

    nn_mat m = {0};
    float *es = nn_arena_alloc(&arena, 25 * sizeof(float));
    nn_mat_init(&m, 5, 5, es);
    nn_mat_fill_func(&m, func);

    nn_mat b = {0};
    es = nn_arena_alloc(&arena, 2 * 2 * sizeof(float));
    nn_mat_init(&b, 2, 2, es);
    
    printf("%zu\n", m.rows);
    nn_mat_slice(&m, 2, 4, 3, 5, &b);
    nn_mat_print(&m);
    nn_mat_print(&b);
    return 0;
}




