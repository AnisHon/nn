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

int32_t nn_q8_asym_to_int(int8_t raw, int32_t zp) {
    return (int32_t)(uint8_t)raw - zp;
}

int32_t nn_q8_sym_to_int(int8_t raw, int32_t zp) {
    return (int32_t)raw - zp;
}

int8_t nn_int_to_q8_asym(int32_t value, int32_t zp) {
    int32_t q = value + zp;
    if (q < 0) {
        q = 0;
    }
    if (q > 255) {
        q = 255;
    }
    return (int8_t)(uint8_t)q;
}

int8_t nn_qalign_i8(int8_t x_q, float x_scale, int32_t x_zp, float y_scale,
                    int32_t y_zp) {
    const float real = ((float)nn_q8_asym_to_int(x_q, x_zp)) * x_scale;
    const int32_t y_int = (int32_t)lrintf(real / y_scale);
    return nn_int_to_q8_asym(y_int, y_zp);
}

int8_t nn_qadd_i8(int8_t a_q, float a_scale, int32_t a_zp, int8_t b_q,
                  float b_scale, int32_t b_zp, float out_scale,
                  int32_t out_zp) {
    const float a_real = ((float)nn_q8_asym_to_int(a_q, a_zp)) * a_scale;
    const float b_real = ((float)nn_q8_asym_to_int(b_q, b_zp)) * b_scale;
    const float sum_real = a_real + b_real;
    const int32_t out_int = (int32_t)lrintf(sum_real / out_scale);
    return nn_int_to_q8_asym(out_int, out_zp);
}

void nn_qadd_i8_vec(int8_t *out, const int8_t *a, float a_scale, int32_t a_zp,
                    const int8_t *b, float b_scale, int32_t b_zp,
                    float out_scale, int32_t out_zp, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        out[i] = nn_qadd_i8(a[i], a_scale, a_zp, b[i], b_scale, b_zp, out_scale,
                            out_zp);
    }
}
