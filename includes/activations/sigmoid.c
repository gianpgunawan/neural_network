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
static const char *sigmoid_get_name(void);

#ifdef SIGMOID_IMPLEMENTATION

const char *SIGMOID_TAG_NAME = "Sigmoid";

static NN_Activation_Ops actv_ops = {
    .regular = sigmoid_func,
    .derived = sigmoid_dfunc,
    .get_name = sigmoid_get_name,
};

static NN_Activation actv = {
    .ops = &actv_ops,
};

void nn_activation_sigmoid_init(NN_Activation_Sigmoid *sig)
{
    sig->actv = &actv;
}

static const char *sigmoid_get_name(void)
{
    return SIGMOID_TAG_NAME;
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
