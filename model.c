#include "arena.c"
#include "matrices/matrix_dyn.c"
#include "matrices/matrix.c"
#include "nn.c"

void load_model(NN model, NN_Arena arena)
{
    nn_mat a0 = nn_mdyn_make_mat(&arena, 1, 2, (float []){
        1.000000, 1.000000, 
    })
    ;
    float z0[] = nn_mat z0 = nn_mdyn_make_mat(&arena, 1, 2, (float []){
        0.000000, 0.000000, 
    }
    );
    float b0[] = nn_mat b0 = nn_mdyn_make_mat(&arena, 1, 2, (float []){
        0.500000, 0.500000, 
    }
    );
    float w0[] = nn_mat w0 = nn_mdyn_make_mat(&arena, 1, 2, (float []){
        0.500000, 0.500000, 
    }
    );
    NN_Layer layer0 = { .a = a0, .w = w0, .z = z0, .b = b0, .nodes = 2, .activation = relu.actv, };
    nn_add_predefined_layer(&model, layer0);
    nn_mat a1 = nn_mdyn_make_mat(&arena, 1, 2, (float []){
        2.047315, 2.047315, 
    })
    ;
    float z1[] = nn_mat z1 = nn_mdyn_make_mat(&arena, 1, 2, (float []){
        2.047315, 2.047315, 
    }
    );
    float b1[] = nn_mat b1 = nn_mdyn_make_mat(&arena, 1, 2, (float []){
        -2.047332, -2.047332, 
    }
    );
    float w1[] = nn_mat w1 = nn_mdyn_make_mat(&arena, 2, 2, (float []){
        2.047323, 2.047323, 
        2.047325, 2.047325, 
    }
    );
    NN_Layer layer1 = { .a = a1, .w = w1, .z = z1, .b = b1, .nodes = 2, .activation = relu.actv, };
    nn_add_predefined_layer(&model, layer1);
    nn_mat a2 = nn_mdyn_make_mat(&arena, 1, 1, (float []){
        0.998602, 
    })
    ;
    float z2[] = nn_mat z2 = nn_mdyn_make_mat(&arena, 1, 1, (float []){
        6.571050, 
    }
    );
    float b2[] = nn_mat b2 = nn_mdyn_make_mat(&arena, 1, 1, (float []){
        -6.322371, 
    }
    );
    float w2[] = nn_mat w2 = nn_mdyn_make_mat(&arena, 2, 1, (float []){
        3.148860, 
        3.148860, 
    }
    );
    NN_Layer layer2 = { .a = a2, .w = w2, .z = z2, .b = b2, .nodes = 2, .activation = relu.actv, };
    nn_add_predefined_layer(&model, layer2);
}

