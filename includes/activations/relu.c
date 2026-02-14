#ifndef RELU_H
#define RELU_H

#include <math.h>
#include "activations/activation.h"

typedef struct {
    NN_Activation *actv;
} NN_Activation_ReLU;

void nn_activation_relu_init(NN_Activation_ReLU *sig);
static float relu_func(float x);
static float relu_dfunc(float x);
static const char *relu_get_name(void);

#ifdef RELU_IMPLEMENTATION

const char *RELU_TAG_NAME = "ReLU";

static NN_Activation_Ops actv_ops = {
    .regular = relu_func,
    .derived = relu_dfunc,
    .get_name = relu_get_name,
};

static NN_Activation actv = {
    .ops = &actv_ops,
};

void nn_activation_relu_init(NN_Activation_ReLU *relu)
{
    relu->actv = &actv;
}

static float relu_func(float x)
{
    return fmax(0, x);
}

static const char *relu_get_name(void)
{
    return RELU_TAG_NAME;
}

static float relu_dfunc(float x)
{
    return x <= 0.0 ? 0.0 : 1.0;
}

#endif // RELU_IMPLEMENTATION

#endif // RELU_H
