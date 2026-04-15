#include "nn_fc.h"
#include "nn_matrix.h"
#include "nn_utils.h"
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

void nn_fc_init(nn_fc *fc, int8_t *weight, int8_t *bias, size_t d_in,
                size_t d_out, nn_matrix *weight_mtx, nn_matrix *bias_mtx,
                int32_t weight_zp, int32_t out_zp) {
    fc->d_in = d_in;
    fc->d_out = d_out;
    fc->weight = weight_mtx;
    fc->bias = bias_mtx;
    fc->weight_zp = weight_zp;
    fc->out_zp = out_zp;
    nn_mtx_init(weight_mtx, d_in, d_out, weight, weight_zp);
    nn_mtx_init(bias_mtx, d_out, 1, bias, weight_zp);
}

void nn_fc_forward(const nn_fc *fc, const nn_matrix *in, nn_matrix *out) {
    assert(in->h == 1);
    assert(in->w == fc->d_in);
    assert(out->h == 1);
    assert(out->w == fc->d_out);
    assert(out->zp == fc->out_zp);

    for (size_t o = 0; o < fc->d_out; o++) {
        int32_t acc = 0;
        if (fc->bias != NULL) {
            acc = (int32_t)(*nn_mtx_get(fc->bias, 0, o));
        }
        for (size_t i = 0; i < fc->d_in; i++) {
            const int32_t x = nn_to_int(*nn_mtx_get(in, 0, i), in->zp);
            const int32_t w =
                nn_to_int(*nn_mtx_get(fc->weight, o, i), fc->weight_zp);
            acc += x * w;
        }
        nn_mtx_set(out, 0, o, nn_to_i8(acc, fc->out_zp));
    }
}
