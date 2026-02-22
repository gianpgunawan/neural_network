#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <math.h>
#include "activations/activation.h"
#include "matrices/matrix.c"
#include "matrices/matrix_dyn.c"
#include "nn.c"
#include "utils/container_of.h"
#include "utils/dynamic_array.h"

typedef struct {
    NN_Activation actv;
    NN *model;
} NN_Activation_Softmax;

void nn_activation_softmax_init(NN_Activation_Softmax *softmax, NN *model);

#ifdef SOFTMAX_IMPLEMENTATION

static float func(float x);
static float dfunc(float x);
static NN_Activation_Func get_func(NN_Activation *a);
static NN_Activation_Func get_dfunc(NN_Activation *a);

static const char *get_name(NN_Activation *a);

const char *SOFTMAX_TAG_NAME = "softmax";
static NN *softmax_model;

static NN_Activation_Ops actv_ops = {
    .regular = get_func,
    .derived = get_dfunc,
    .get_name = get_name,
};

static NN_Activation actv = {
    .ops = &actv_ops,
};

void nn_activation_softmax_init(NN_Activation_Softmax *softmax, NN *model)
{
    softmax->actv = actv;
    softmax_model = model;
}

static const char *get_name(NN_Activation *a)
{
    return SOFTMAX_TAG_NAME;
}

static float func(float x)
{
    NN_Layer outl = da_last(&(softmax_model->layers));
    size_t row = 0;
    double sumd = 0.0;
    for (size_t col = 0; col < outl.z.cols; ++col) {
        double v = NN_MAT_AT(&outl.z, row, col);
        sumd += exp(v);
    }
    return exp(x)/sumd;
}

static float dfunc(float x)
{
    NN_Layer outl = da_last(&(softmax_model->layers));
    size_t row = 0;
    double sumd = 0.0;
    for (size_t col = 0; col < outl.z.cols; ++col) {
        double v = NN_MAT_AT(&outl.z, row, col);
        sumd += exp(v);
    }
    return (exp(x) * (sumd - exp(x)))/sumd;
}

static NN_Activation_Func get_func(NN_Activation *a)
{
    return func;
}

static NN_Activation_Func get_dfunc(NN_Activation *a)
{
    return dfunc;
}

#endif // SOFTMAX_IMPLEMENTATION
#endif // SOFTMAX_H
