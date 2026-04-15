#include "nn_conv2d.h"
#include "nn_image.h"
#include "nn_matrix.h"
#include "nn_utils.h"
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

static bool check_valid(const nn_conv2d *restrict conv2d) {
    bool valid = true;
    for (size_t i = 0; i < conv2d->c_in * conv2d->c_out; i++) {
        const nn_matrix *m = conv2d->mtx + i;
        if (m->h != conv2d->h || m->w != conv2d->w) {
            valid = false;
            break;
        }
    }
    return valid;
}

static bool check_forward(const nn_conv2d *restrict conv2d, const nn_image *in,
                          const nn_image *out) {
    size_t h_out, w_out;
    nn_conv2d_output_sz(conv2d, in->h, in->w, &h_out, &w_out);
    if (out->h != h_out || out->w != w_out) {
        return false;
    }
    return true;
}

void nn_conv2d_init(nn_conv2d *restrict conv2d, int8_t *weight,
                    nn_matrix *mtx, const size_t c_in, const size_t c_out,
                    const size_t h, const size_t w, const size_t stride_h,
                    const size_t stride_w, const size_t padding_h,
                    const size_t padding_w, const int32_t *bias,
                    const int32_t weight_zp, const int32_t out_zp) {
    conv2d->mtx = mtx;
    conv2d->c_in = c_in;
    conv2d->c_out = c_out;
    conv2d->h = h;
    conv2d->w = w;
    conv2d->stride_h = stride_h;
    conv2d->stride_w = stride_w;
    conv2d->padding_h = padding_h;
    conv2d->padding_w = padding_w;
    conv2d->weight_zp = weight_zp;
    conv2d->out_zp = out_zp;
    conv2d->bias = bias;

    for (size_t i = 0; i < c_in * c_out; i++) {
        nn_matrix *cur = conv2d->mtx + i;
        int8_t *p = weight + i * h * w;
        nn_mtx_init(cur, h, w, p, weight_zp);
    }

    assert(check_valid(conv2d));
}

nn_matrix *nn_conv2d_get(const nn_conv2d *conv2d, size_t c_in, size_t c_out) {
    return conv2d->mtx + (conv2d->c_out * c_in) + c_out;
}

void nn_conv2d_output_sz(const nn_conv2d *conv2d, const size_t h_in,
                         const size_t w_in, size_t *h_out, size_t *w_out) {
    *h_out = (h_in - conv2d->h + 2 * conv2d->padding_h) / conv2d->stride_h + 1;
    *w_out = (w_in - conv2d->w + 2 * conv2d->padding_w) / conv2d->stride_w + 1;
}

static inline int32_t do_conv(const nn_matrix *conv, const nn_matrix *image,
                              size_t si, size_t sj) {
    int32_t res = 0;
    for (size_t i = 0; i < conv->h; i++) {
        for (size_t j = 0; j < conv->w; j++) {
            const int32_t conv_value =
                nn_q8_sym_to_int(*nn_mtx_get(conv, i, j), conv->zp);
            const int32_t image_value =
                nn_q8_asym_to_int(*nn_mtx_get(image, si + i, sj + j), image->zp);
            res += conv_value * image_value;
        }
    }
    return res;
}

static inline int32_t do_conv_padded(const nn_matrix *conv,
                                     const nn_matrix *image, size_t si,
                                     size_t sj, size_t pad_h, size_t pad_w) {
    int32_t res = 0;
    for (size_t i = 0; i < conv->h; i++) {
        for (size_t j = 0; j < conv->w; j++) {
            const int32_t in_i = (int32_t)si + (int32_t)i - (int32_t)pad_h;
            const int32_t in_j = (int32_t)sj + (int32_t)j - (int32_t)pad_w;
            int32_t image_value = 0;
            if (in_i >= 0 && in_j >= 0 && (size_t)in_i < image->h &&
                (size_t)in_j < image->w) {
                image_value = nn_q8_asym_to_int(
                    *nn_mtx_get(image, (size_t)in_i, (size_t)in_j), image->zp);
            }
            const int32_t conv_value =
                nn_q8_sym_to_int(*nn_mtx_get(conv, i, j), conv->zp);
            res += conv_value * image_value;
        }
    }
    return res;
}

void nn_conv2d_forward(const nn_conv2d *conv2d, const nn_image *in,
                       nn_image *out) {
    if (!check_forward(conv2d, in, out)) {
        size_t h_out, w_out;
        nn_conv2d_output_sz(conv2d, in->h, in->w, &h_out, &w_out);
        fprintf(stderr,
                "conv2d forward size mismatch: in=%zux%zu, conv=(%zux%zu,stride=%zux%zu,pad=%zux%zu), expected out=%zux%zu got %zux%zu\n",
                in->h, in->w, conv2d->h, conv2d->w, conv2d->stride_h,
                conv2d->stride_w, conv2d->padding_h, conv2d->padding_w,
                h_out, w_out, out->h, out->w);
    }
    assert(check_forward(conv2d, in, out));
    assert(conv2d->out_zp == out->zp);

    for (size_t co = 0; co < conv2d->c_out; co++) {
        nn_matrix *out_mtx = nn_image_get(out, co);
        const size_t h_out = out_mtx->h;
        const size_t w_out = out_mtx->w;
        for (size_t i = 0; i < h_out; i++) {
            for (size_t j = 0; j < w_out; j++) {
                int32_t acc = 0;
                for (size_t ci = 0; ci < conv2d->c_in; ci++) {
                    nn_matrix *conv_kernel = nn_conv2d_get(conv2d, ci, co);
                    nn_matrix *in_mtx = nn_image_get(in, ci);
                    const size_t in_i = i * conv2d->stride_h;
                    const size_t in_j = j * conv2d->stride_w;
                    if (conv2d->padding_h == 0 && conv2d->padding_w == 0) {
                        acc += do_conv(conv_kernel, in_mtx, in_i, in_j);
                    } else {
                        acc += do_conv_padded(conv_kernel, in_mtx, in_i, in_j,
                                              conv2d->padding_h,
                                              conv2d->padding_w);
                    }
                }
                if (conv2d->bias != NULL) {
                    acc += conv2d->bias[co];
                }
                nn_mtx_set(out_mtx, i, j, nn_int_to_q8_asym(acc, conv2d->out_zp));
            }
        }
    }
}
