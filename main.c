#include <string.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#include "activations/sigmoid.c"
#include "activations/softmax.c"
#include "activations/relu.c"
#include "arena.c"
#include "matrices/matrix.c"
#include "matrices/matrix_dyn.c"
#include "nn.c"
#include "utils/nn_assert.h"
#include "utils/dynamic_array.h"
#include "bitmap.h"

#define MODEL_IMPLEMENTATION
#include "model.h"

int extract_float_from_args(float *vals, int argc, char **argv)
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

    NN_Arena arena = {0};
    size_t arena_sz = 256 * 1024 * 1024; // 256 MBs
    nn_arena_init(&arena, arena_sz);
    
    NN model = {0}; nn_init(&model);

    NN_Activation_Sigmoid sig = {0}; nn_activation_sigmoid_init(&sig);
    NN_Activation_ReLU relu = {0}; nn_activation_relu_init(&relu);
    NN_Activation_Softmax softmax = {0}; nn_activation_softmax_init(&softmax, &model);
    
//    nn_add_layer(&arena, &model, 2, &relu.actv);
//    nn_add_layer(&arena, &model, 2, &relu.actv);
//    nn_add_layer(&arena, &model, 1, &sig.actv);
    model_load(&arena, &model);
    float inputs[] = {0, 0};
    extract_float_from_args(inputs, argc, argv);
    
    nn_mat dataset = {0};
    const size_t ROWS = 4;
    const size_t COLS = 3;
    const size_t target_start_col = 2;

#define XOR 
#if defined(AND)
    const char *name = "AND DATASET";
    float ds[] = {
        1, 1, 1, 0,
        1, 0, 0, 1,
        0, 1, 0, 1,
        0, 0, 0, 1,
    };
#elif defined(ADDITION)
    const char *name = "ADDITION DATASET";
    float ds[] = {
        1, 1, 2,
        1, 0, 1,
        0, 1, 1,
        0, 0, 0,
        3, 1, 4,
        2, 0, 2,
        0, 3, 3,
        3, 0, 3,
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
    //for (size_t i = 1; i < epochs; ++i) {
    //    nn_backprog(&arena, &model, &dataset, target_start_col);
    //}
    NN_MAT_AT(&model.layers.items[0].a, 0, 0) = inputs[0];
    NN_MAT_AT(&model.layers.items[0].a, 0, 1) = inputs[1];

    nn_forward_pass(&model);
    nn_mat_print(&(model.layers.items[0].a));
    nn_mat_print(&(da_last(&model.layers).a));

    nn_dump(&arena, &model, "model.h", "model");
    return 0;
}
