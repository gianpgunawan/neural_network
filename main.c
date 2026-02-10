#include <string.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#include "activations/sigmoid.c"
#include "activations/relu.c"
#include "arena.c"
#include "matrices/matrix.c"
#include "nn.c"
#include "utils/nn_assert.h"

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

int main(int argc, char **argv)
{
    srand(time(NULL));
    nn_arena arena = {0};
    size_t arena_sz = 256 * 1000 * 1000; // 256 MBs
    nn_arena_init(&arena, arena_sz);

    nn model = {0};
    
    if (argc < 3) {
        printf("enter the inputs first");
        return 0;
    }

    bool toggle = 0;
    size_t arc[] = {2, 2, 1};
    size_t arc_len = sizeof(arc) / sizeof(arc[0]);
    nn_init(&model, &arena, arc, arc_len);

    nn_mat dataset = {0};
    const size_t ROWS = 4;
    const size_t COLS = 3;
    const size_t target_start_col = 2;

    Sigmoid sig = {0}; sigmoid_init(&sig);
    Relu relu = {0}; relu_init(&relu);

    float and_ds[] = {
        1, 1, 1,
        1, 0, 0,
        0, 1, 0,
        0, 0, 0,
    };

    float xor_ds[] = {
        1, 1, 0,
        1, 0, 1,
        0, 1, 1,
        0, 0, 0,
    };

    float *es = nn_arena_alloc(&arena, ROWS * COLS * sizeof(float));
    memcpy(es, xor_ds, ROWS * COLS * sizeof(float));

    nn_mat_init(&dataset, ROWS, COLS, es);
    for (size_t i = 0; i < 100000; ++i) {
        nn_backprog(&model, &arena, &dataset, target_start_col, sig.actv, relu.actv);
    }
    char *end;
    char *end2;
    float x = strtof(argv[1], &end);
    float y = strtof(argv[2], &end2);
    if (end == argv[1] || end2 == argv[2]) {
        printf("Invalid Input\n");
        return 1;
    }
    
    // nn_backprog(&model, &arena); 
    NN_MAT_AT(&model.as[0], 0, 0) = x;
    NN_MAT_AT(&model.as[0], 0, 1) = y;

    nn_forward_pass(&model, relu.actv, sig.actv);
        
    printf("\n");
    nn_mat_print(&(model.as[0]));
    nn_mat_print(&(model.as[2]));
    
    return 0;
}
