#ifndef RELU_H
#define RELU_H

#include <math.h>
#include "activations/activation.h"

typedef struct {
    NN_Activation actv;
} NN_Activation_ReLU;

void nn_activation_relu_init(NN_Activation_ReLU *sig);

#ifdef RELU_IMPLEMENTATION

static float func(float x);
static float dfunc(float x);
static NN_Activation_Func get_func(NN_Activation *a);
static NN_Activation_Func get_dfunc(NN_Activation *a);
static const char *get_name(NN_Activation *a);

const char *RELU_TAG_NAME = "ReLU";

static NN_Activation_Ops actv_ops = {
    .regular = get_func,
    .derived = get_dfunc,
    .get_name = get_name,
};

static NN_Activation actv = {
    .ops = &actv_ops,
};

void nn_activation_relu_init(NN_Activation_ReLU *relu)
{
    relu->actv = actv;
}

static float func(float x)
{
    return fmax(0, x);
}

static const char *get_name(NN_Activation *a)
{
    return RELU_TAG_NAME;
}

static float dfunc(float x)
{
    return x <= 0.0 ? 0.0 : 1.0;
}

static NN_Activation_Func get_func(NN_Activation *a)
{
    return func;
}

static NN_Activation_Func get_dfunc(NN_Activation *a)
{
    return dfunc;
}

#endif // RELU_IMPLEMENTATION
#endif // RELU_H
