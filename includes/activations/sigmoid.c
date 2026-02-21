#ifndef SIGMOID_H
#define SIGMOID_H

#include <math.h>
#include "activations/activation.h"

typedef struct {
    NN_Activation actv;
} NN_Activation_Sigmoid;

void nn_activation_sigmoid_init(NN_Activation_Sigmoid *sig);

#ifdef SIGMOID_IMPLEMENTATION

static float func(float x);
static float dfunc(float x);
static NN_Activation_Func get_func(NN_Activation *a);
static NN_Activation_Func get_dfunc(NN_Activation *a);
static const char *get_name(NN_Activation *a);

const char *SIGMOID_TAG_NAME = "Sigmoid";

static NN_Activation_Ops actv_ops = {
    .regular = get_func,
    .derived = get_dfunc,
    .get_name = get_name,
};

static NN_Activation actv = {
    .ops = &actv_ops,
};

void nn_activation_sigmoid_init(NN_Activation_Sigmoid *sig)
{
    sig->actv = actv;
}

static const char *get_name(NN_Activation *a)
{
    return SIGMOID_TAG_NAME;
}

static float dfunc(float x)
{
    float sigval = func(x);
    return sigval * (1.0f - sigval);
}

static float func(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

static NN_Activation_Func get_func(NN_Activation *a)
{
    return func;
}

static NN_Activation_Func get_dfunc(NN_Activation *a)
{
    return dfunc;
}
#endif // SIGMOID_IMPLEMENTATION
#endif // SIGMOID_H
