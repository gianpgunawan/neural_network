#ifndef ACTIVATION_H
#define ACTIVATION_H

typedef struct {
    float (*regular)(float);
    float (*derived)(float);
} NN_Activation_Ops;

typedef struct {
    NN_Activation_Ops *ops;
} NN_Activation;

#endif // ACTIVATION_H
