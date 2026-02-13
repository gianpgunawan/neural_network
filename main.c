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
#include "utils/dynamic_array.h"

int extract_float_from_flags(float *vals, int argc, char **argv)
{
    if (argc < 3) {
        printf("enter the inputs first");
        return 0;
    }

    char *end;
    char *end2;
    float x = strtof(argv[1], &end);
    float y = strtof(argv[2], &end2);
    if (end == argv[1] || end2 == argv[2]) {
        return -1;
    }
    vals[0] = x;
    vals[1] = y;

    return 0;
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    nn_arena arena = {0};
    size_t arena_sz = 256 * 1000 * 1000; // 256 MBs
    nn_arena_init(&arena, arena_sz);

    // activations
    NN_Activation_Sigmoid sig = {0}; nn_activation_sigmoid_init(&sig);
    NN_Activation_ReLU relu = {0}; nn_activation_relu_init(&relu);

    NN new_model = {0}; nn_init(&new_model);
    nn_add_layer(&arena, &new_model, 2, relu.actv);
    nn_add_layer(&arena, &new_model, 2, relu.actv);
    nn_add_layer(&arena, &new_model, 1, sig.actv);
    
    float inputs[] = {0, 0};
    extract_float_from_flags(inputs, argc, argv);
    
    nn_mat dataset = {0};
    const size_t ROWS = 4;
    const size_t COLS = 3;
    const size_t target_start_col = 2;

#if 1 
    const char *name = "AND DATASET";
    float ds[] = {
        1, 1, 1,
        1, 0, 0,
        0, 1, 0,
        0, 0, 0,
    };
#else
    const char *name = "XOR DATASET";
    float ds[] = {
        1, 1, 0,
        1, 0, 1,
        0, 1, 1,
        0, 0, 0,
    };
#endif
    size_t epochs = 100 * 1000;
    printf("%s\n", name);
    float *es = nn_arena_alloc(&arena, ROWS * COLS * sizeof(float));
    memcpy(es, ds, ROWS * COLS * sizeof(float));
    nn_mat_init(&dataset, ROWS, COLS, es);
    for (size_t i = 0; i < epochs; ++i) {
        nn_backprog(&arena, &new_model, &dataset, target_start_col);
    }
    NN_MAT_AT(&new_model.layers.items[0].a, 0, 0) = inputs[0];
    NN_MAT_AT(&new_model.layers.items[0].a, 0, 1) = inputs[1];

    nn_forward_pass(&new_model);
        
    printf("\n");
    nn_mat_print(&(new_model.layers.items[0].a));
    nn_mat_print(&(da_last(&new_model.layers).a));
    return 0;
}
