#ifndef ACTIVATION_H
#define ACTIVATION_H

typedef struct {
    float (*regular)(float);
    float (*derived)(float);
} Activation_Ops;

typedef struct {
    Activation_Ops *ops;
} Activation;

#endif // ACTIVATION_H
