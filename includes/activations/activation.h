#ifndef ACTIVATION_H
#define ACTIVATION_H

typedef float (*NN_Activation_Func)(float);

typedef struct NN_Activation NN_Activation;
typedef struct NN_Activation_Ops NN_Activation_Ops;

struct NN_Activation_Ops {
    NN_Activation_Func (*regular)(NN_Activation *);
    NN_Activation_Func (*derived)(NN_Activation *);
    const char *(*get_name)(NN_Activation *);
};

struct NN_Activation {
    NN_Activation_Ops *ops;
};

#endif // ACTIVATION_H
