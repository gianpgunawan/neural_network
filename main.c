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
#include "matrices/matrix_dyn.c"
#include "nn.c"
#include "utils/nn_assert.h"
#include "utils/dynamic_array.h"
#include "model.c"

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
    size_t arena_sz = 256 * 1000 * 1000; // 256 MBs
    nn_arena_init(&arena, arena_sz);

    // activations
    NN_Activation_Sigmoid sig = {0}; nn_activation_sigmoid_init(&sig);
    NN_Activation_ReLU relu = {0}; nn_activation_relu_init(&relu);

    NN new_model = {0}; nn_init(&new_model);

//    nn_add_layer(&arena, &new_model, 2, relu.actv);
//    nn_add_layer(&arena, &new_model, 2, relu.actv);
//    nn_add_layer(&arena, &new_model, 1, sig.actv);

    // LAYER 1 
    nn_mat z1 = nn_mdyn_make_mat(&arena, 1, 2, (float[]) {
                0.0, 0.0,
            });
    nn_mat a1 = nn_mdyn_make_mat(&arena, 1, 2, (float[]) {
                0.5, 0.5,
            });
    nn_mat w1 = nn_mdyn_make_mat(&arena, 1, 2, (float[]) {
                0.5, 0.5,
            });
    nn_mat b1 = nn_mdyn_make_mat(&arena, 1, 2, (float[]) {
                0.5, 0.5,
            });

    // LAYER 2
    nn_mat z2 = nn_mdyn_make_mat(&arena, 1, 2, (float[]) {
                0.0, 0.0,
            });
    nn_mat a2 = nn_mdyn_make_mat(&arena, 1, 2, (float[]) {
                0.5, 0.5,
            });
    nn_mat w2 = nn_mdyn_make_mat(&arena, 2, 2, (float[]) {
                0.5, 0.5,
                0.5, 0.5,
            });
    nn_mat b2 = nn_mdyn_make_mat(&arena, 1, 2, (float[]) {
                0.5, 0.5,
            });

    // LAYER 3 
    nn_mat w3 = nn_mdyn_make_mat(&arena, 2, 1, (float[]) {
                0.5,
                0.5,
            });
    nn_mat z3 = nn_mdyn_make_mat(&arena, 1, 1, (float[]) {
                0.5,
            });
    nn_mat a3 = nn_mdyn_make_mat(&arena, 1, 1, (float[]) {
                0.0,
            });
    nn_mat b3 = nn_mdyn_make_mat(&arena, 1, 1, (float[]) {
                0.5,
            });

    NN_Layer layer1 = {
        .a = a1,
        .w = w1,
        .z = z1,
        .b = b1,
        .nodes = 2,
        .activation = relu.actv,
    };

    NN_Layer layer2 = {
        .a = a2,
        .w = w2,
        .z = z2,
        .b = b2,
        .nodes = 2,
        .activation = relu.actv,
    };
    
    NN_Layer layer3 = {
        .a = a3,
        .w = w3,
        .z = z3,
        .b = b3,
        .nodes = 1,
        .activation = sig.actv,
    };

    nn_add_predefined_layer(&new_model, layer1);
    nn_add_predefined_layer(&new_model, layer2);
    nn_add_predefined_layer(&new_model, layer3);

    float inputs[] = {0, 0};
    extract_float_from_args(inputs, argc, argv);
    
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
    nn_mat_print(&(new_model.layers.items[0].a));
    nn_mat_print(&(da_last(&new_model.layers).a));
    nn_dump(&new_model, "model.c");
    return 0;
}
