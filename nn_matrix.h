#ifndef NN_MATRIX_H
#define NN_MATRIX_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t h, w;
    int8_t *elem;
    int32_t zp;
} nn_matrix;

void nn_mtx_init(nn_matrix *mtx, const size_t h, const size_t w, int8_t *elem,
                 int32_t zp);

int8_t *nn_mtx_get(const nn_matrix *mtx, const size_t n, const size_t m);

void nn_mtx_set(nn_matrix *mtx, const size_t n, const size_t m,
                const int8_t v);

void nn_mtx_mul_dim(const nn_matrix *a, const nn_matrix *b, size_t *h,
                    size_t *w);

void nn_mtx_tozero(nn_matrix *mtx);

void nn_mtx_mul(const nn_matrix *a, const nn_matrix *b, nn_matrix *out);

#endif
