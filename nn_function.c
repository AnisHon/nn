#include "nn_function.h"
#include "nn_matrix.h"
#include "nn_stensor.h"
#include "nn_utils.h"
#include "stdint.h"
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

static inline int8_t relu(int8_t num, int8_t zero) {
    int signed_num = nn_to_int(num, zero);
    int8_t res;
    if (signed_num < 0) {
        res = zero;
    } else {
        res = num;
    }
    return res;
}

void nn_relu_image(nn_image *image, nn_image *out) {
    for (size_t i = 0; i < image->channels; i++) {
        nn_relu_matrix(nn_image_get(image, i), nn_image_get(out, i));
    }
}

void nn_relu_matrix(nn_matrix *vector, nn_matrix *out) {
    for (size_t i = 0; i < vector->h; i++) {
        for (size_t j = 0; j < vector->w; j++) {
            int8_t res = relu(*nn_mtx_get(vector, i, j), vector->zp);
            nn_mtx_set(out, i, j, res);
        }
    }
}

void nn_relu(const nn_stensor *ti, nn_stensor *to) {
    switch (ti->tensor_type) {
    case NN_STENSOR_IMG:
        nn_relu_image(ti->tensor.img, to->tensor.img);
        break;
    case NN_STENSOR_MTX:
        nn_relu_matrix(ti->tensor.mtx, to->tensor.mtx);
        break;
    }
}

void nn_flatten(nn_image *image, nn_matrix *out) {
    const size_t total = image->channels * image->h * image->w;
    assert(out->h * out->w == total);
    assert(out->zp == image->zp);
    size_t idx = 0;
    for (size_t c = 0; c < image->channels; c++) {
        nn_matrix *m = nn_image_get(image, c);
        for (size_t i = 0; i < image->h; i++) {
            for (size_t j = 0; j < image->w; j++) {
                out->elem[idx++] = *nn_mtx_get(m, i, j);
            }
        }
    }
}

void nn_avg_pool(nn_image *image, nn_image *out) {
    assert(out->channels == image->channels);
    assert(out->h == 1 && out->w == 1);
    assert(out->zp == image->zp);
    const int32_t area = (int32_t)(image->h * image->w);
    for (size_t c = 0; c < image->channels; c++) {
        int32_t sum = 0;
        nn_matrix *in = nn_image_get(image, c);
        for (size_t i = 0; i < image->h; i++) {
            for (size_t j = 0; j < image->w; j++) {
                sum += nn_to_int(*nn_mtx_get(in, i, j), image->zp);
            }
        }
        const int32_t mean = sum / area;
        nn_mtx_set(nn_image_get(out, c), 0, 0, nn_to_u8(mean, out->zp));
    }
}
