#ifndef NN_ARENA_H
#define NN_ARENA_H

#include <stddef.h>
#include <stdlib.h>
#include <assert.h>

typedef struct {
    size_t capacity;
    size_t count;
    unsigned char *data;
} nn_arena;

int nn_arena_init(nn_arena *arena, size_t n);
int nn_arena_init_from_block(nn_arena *arena, void *block, size_t size);
void nn_arena_reset(nn_arena *arena);
void *nn_arena_alloc(nn_arena *arena, size_t n);
void nn_arena_free(nn_arena *arena);
void nn_arena_reset_to(nn_arena *arena, size_t checkpoint);

#ifdef NN_ARENA_IMPLEMENTATION

int nn_arena_init(nn_arena *arena, size_t n)
{
    arena->capacity = n;
    arena->count = 0;
    arena->data = malloc(n);
    return arena->data != NULL;
}

int nn_arena_init_from_block(nn_arena *arena, void *block, size_t size)
{
    arena->capacity = size;
    arena->count = 0;
    arena->data = block;
    return arena->data != NULL;
}

void nn_arena_reset(nn_arena *arena)
{
    arena->count = 0;
}

void nn_arena_reset_to(nn_arena *arena, size_t checkpoint)
{
    arena->count = checkpoint;
}

void *nn_arena_alloc(nn_arena *arena, size_t n)
{
    assert(arena->count + n <= arena->capacity);
    void *result = arena->data + arena->count;
    arena->count += n;
    return result;
}

void nn_arena_free(nn_arena *arena)
{
    free(arena->data);
    arena->data = NULL;
    arena->count = 0;
    arena->capacity= 0;
}

#endif // NN_ARENA_IMPLEMENTATION
#endif // NN_ARENA_H
