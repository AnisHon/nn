#ifndef NN_CONV2D_H
#define NN_CONV2D_H
#include "nn_image.h"
#include "nn_matrix.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t c_in, c_out, h, w;
    size_t stride_h, stride_w;
    size_t padding_h, padding_w;
    const int32_t *bias;
    int32_t weight_zp;
    int32_t out_zp;
    nn_matrix *mtx;
} nn_conv2d;

void nn_conv2d_init(nn_conv2d *conv2d, int8_t *params, nn_matrix *mtx,
                    const size_t c_in, const size_t c_out, const size_t h,
                    const size_t w, const size_t stride_h,
                    const size_t stride_w, const size_t padding_h,
                    const size_t padding_w, const int32_t *bias,
                    const int32_t weight_zp, const int32_t out_zp);

void nn_conv2d_output_sz(const nn_conv2d *conv2d, const size_t h_in,
                         const size_t w_in, size_t *h_out, size_t *w_out);

void nn_conv2d_forward(const nn_conv2d *conv2d, const nn_image *in,
                       nn_image *out);

#endif
