#include "nn_utils.h"
#include <math.h>

int nn_to_int(int8_t num, int32_t zp) { return (int32_t)num - zp; }

int8_t nn_clamp_i8(int32_t v) {
    if (v > 127) {
        return 127;
    }
    if (v < -128) {
        return -128;
    }
    return (int8_t)v;
}

int8_t nn_to_u8(int num, int32_t zp) { return nn_clamp_i8((int32_t)num + zp); }

int8_t nn_to_i8(int num, int32_t zp) { return nn_clamp_i8((int32_t)num + zp); }

int8_t nn_qalign_i8(int8_t x_q, float x_scale, int32_t x_zp, float y_scale,
                    int32_t y_zp) {
    // real = (x_q - x_zp) * x_scale
    // y_q  = round(real / y_scale) + y_zp
    const float real = ((float)((int32_t)x_q - x_zp)) * x_scale;
    const int32_t y_q = (int32_t)lrintf(real / y_scale) + y_zp;
    return nn_clamp_i8(y_q);
}

int8_t nn_qadd_i8(int8_t a_q, float a_scale, int32_t a_zp, int8_t b_q,
                  float b_scale, int32_t b_zp, float out_scale,
                  int32_t out_zp) {
    const float a_real = ((float)((int32_t)a_q - a_zp)) * a_scale;
    const float b_real = ((float)((int32_t)b_q - b_zp)) * b_scale;
    const float sum_real = a_real + b_real;
    const int32_t out_q = (int32_t)lrintf(sum_real / out_scale) + out_zp;
    return nn_clamp_i8(out_q);
}

void nn_qadd_i8_vec(int8_t *out, const int8_t *a, float a_scale, int32_t a_zp,
                    const int8_t *b, float b_scale, int32_t b_zp,
                    float out_scale, int32_t out_zp, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        out[i] = nn_qadd_i8(a[i], a_scale, a_zp, b[i], b_scale, b_zp, out_scale,
                            out_zp);
    }
}
