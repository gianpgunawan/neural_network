#ifndef MODEL_H
#define MODEL_H

#include "arena.c"
#include "matrices/matrix_dyn.c"
#include "matrices/matrix.c"
#include "activations/sigmoid.c"
#include "activations/softmax.c"
#include "nn.c"

void model_load(NN_Arena *arena, NN *model);

#ifdef MODEL_IMPLEMENTATION

static NN_Activation_Sigmoid sigmoid = {0};
static NN_Activation_ReLU relu = {0};
static NN_Activation_Softmax softmax = {0};
void model_load(NN_Arena *arena, NN *model)
{
   nn_activation_relu_init(&relu);
   nn_activation_softmax_init(&softmax, model);
   nn_activation_sigmoid_init(&sigmoid);
   nn_mat a0 = nn_mdyn_make_mat(arena, 1, 2, (float []){
    0.000000, 0.000000, 
}
);
   nn_mat z0 = nn_mdyn_make_mat(arena, 1, 2, (float []){
    0.000000, 0.000000, 
}
   );
   nn_mat b0 = nn_mdyn_make_mat(arena, 1, 2, (float []){
    0.000000, 0.000000, 
}
   );
   nn_mat w0 = nn_mdyn_make_mat(arena, 1, 2, (float []){
    0.000000, 0.000000, 
}
   );
   NN_Layer layer0 = { .a = a0, .w = w0, .z = z0, .b = b0, .nodes = 2, .activation = &relu.actv, };
   nn_add_predefined_layer(model, layer0);
   nn_mat a1 = nn_mdyn_make_mat(arena, 1, 2, (float []){
    0.000000, 0.000000, 
}
);
   nn_mat z1 = nn_mdyn_make_mat(arena, 1, 2, (float []){
    -4.016846, -0.000013, 
}
   );
   nn_mat b1 = nn_mdyn_make_mat(arena, 1, 2, (float []){
    -4.016846, -0.000013, 
}
   );
   nn_mat w1 = nn_mdyn_make_mat(arena, 2, 2, (float []){
    4.016838, 2.951495, 
    4.016845, 2.951482, 
}
   );
   NN_Layer layer1 = { .a = a1, .w = w1, .z = z1, .b = b1, .nodes = 2, .activation = &relu.actv, };
   nn_add_predefined_layer(model, layer1);
   nn_mat a2 = nn_mdyn_make_mat(arena, 1, 1, (float []){
    0.003445, 
}
);
   nn_mat z2 = nn_mdyn_make_mat(arena, 1, 1, (float []){
    -5.667471, 
}
   );
   nn_mat b2 = nn_mdyn_make_mat(arena, 1, 1, (float []){
    -5.667471, 
}
   );
   nn_mat w2 = nn_mdyn_make_mat(arena, 2, 1, (float []){
    -6.200844, 
    4.070874, 
}
   );
   NN_Layer layer2 = { .a = a2, .w = w2, .z = z2, .b = b2, .nodes = 2, .activation = &sigmoid.actv, };
   nn_add_predefined_layer(model, layer2);
}
#endif // MODEL_IMPLEMENTATION
#endif // MODEL_H