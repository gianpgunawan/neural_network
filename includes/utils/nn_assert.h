#ifndef NN_ASSERT_H
#define NN_ASSERT_H

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#define NN_ASSERT(cond, message) do { \
    if (!(cond)) { \
        fprintf(stderr,message "\nat file " __FILE__ ":%d\n", __LINE__); \
        exit(1); \
    } \
} while(0);

#endif // NN_ASSERT_H
