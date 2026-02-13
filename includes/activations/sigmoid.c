#ifndef SIGMOID_H
#define SIGMOID_H

#include <math.h>
#include "activations/activation.h"

typedef struct {
    NN_Activation *actv;
} NN_Activation_Sigmoid;

void nn_activation_sigmoid_init(NN_Activation_Sigmoid *sig);
static float sigmoid_func(float x);
static float sigmoid_dfunc(float x);

#ifdef SIGMOID_IMPLEMENTATION

static NN_Activation_Ops actv_ops = {
    .regular = sigmoid_func,
    .derived = sigmoid_dfunc,
};

static NN_Activation actv = {
    .ops = &actv_ops,
};

void nn_activation_sigmoid_init(NN_Activation_Sigmoid *sig)
{
    sig->actv = &actv;
}

static float sigmoid_dfunc(float x)
{
    float sigval = sigmoid_func(x);
    return sigval * (1.0f - sigval);
}

static float sigmoid_func(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

#endif // SIGMOID_IMPLEMENTATION
#endif // SIGMOID_H
