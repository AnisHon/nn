#include "nn_matrix.h"
#include "nn_utils.h"
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

// typedef struct {
//     size_t h, w;
//     int8_t *elem;
// } nn_matrix;

void nn_mtx_init(nn_matrix *restrict mtx, const size_t h, const size_t w,
                 int8_t *elem, int32_t zp) {
    mtx->h = h;
    mtx->w = w;
    mtx->elem = elem;
    mtx->zp = zp;
}

int8_t *nn_mtx_get(const nn_matrix *restrict mtx, const size_t n,
                   const size_t m) {
    assert(n < mtx->h && m >= 0 && m < mtx->w);
    return mtx->elem + n * mtx->w + m;
}

void nn_mtx_set(nn_matrix *restrict mtx, const size_t n, const size_t m,
                const int8_t v) {
    assert(n < mtx->h && m < mtx->w);
    *(mtx->elem + n * mtx->w + m) = v;
}

void nn_mtx_mul_dim(const nn_matrix *restrict a, const nn_matrix *b,
                    size_t *restrict h, size_t *restrict w) {
    assert(a->w == b->h);
    *h = a->h;
    *w = b->w;
}

void nn_mtx_tozero(nn_matrix *restrict mtx) {
    for (size_t i = 0; i < mtx->h * mtx->w; i++) {
        mtx->elem[i] = nn_to_i8(0, mtx->zp);
    }
}

void nn_mtx_mul(const nn_matrix *restrict a, const nn_matrix *restrict b,
                nn_matrix *restrict out) {
    assert(a->w == b->h && a->h == out->h && b->w == out->w);
    nn_mtx_tozero(out);
    // 细节 cache friendly
    for (size_t i = 0; i < a->h; i++) {
        for (size_t k = 0; k < a->w; k++) {
            for (size_t j = 0; j < b->w; j++) {
                int8_t ta = *nn_mtx_get(a, i, k);
                int8_t tb = *nn_mtx_get(b, k, j);
                int8_t *tc = nn_mtx_get(out, i, j);

                uint32_t t1 = nn_to_int(ta, a->zp);
                uint32_t t2 = nn_to_int(tb, b->zp);
                uint32_t t3 = t1 * t2;
                *tc = nn_to_i8(t3, out->zp);
            }
        }
    }
}
