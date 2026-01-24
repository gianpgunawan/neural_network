#include <string.h>
#include <time.h>
#include <math.h>

#include "implementations.h"

#include "arena.h"
#include "matrix.h"
#include "nn_assert.h"
#include "nn.h"

int main()
{
    srand(time(NULL));
    nn_arena arena = {0};
    size_t arena_sz = 256 * 1000 * 1000; // 256 MBs
    nn_arena_init(&arena, arena_sz);
    nn model = {0};
    size_t arc[] = {2, 2, 1};
    size_t arc_len = sizeof(arc) / sizeof(arc[0]);
    printf("AS: \n");
    nn_init(&model, &arena, arc, arc_len);
    nn_mat_print(&model.as[0]);
    nn_mat_print(&model.as[1]);
    nn_mat_print(&model.as[2]);

    printf("Weights: \n");
    nn_mat_print(&model.ws[1]);
    nn_mat_print(&model.ws[2]);
    return 0;
}
