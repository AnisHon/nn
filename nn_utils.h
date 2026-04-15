#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <stdint.h>

int nn_to_int(int8_t num, int32_t zp);
int8_t nn_to_u8(int num, int32_t zp);
int8_t nn_to_i8(int num, int32_t zp);
int8_t nn_clamp_i8(int32_t v);

// 将量化值从 (x_scale, x_zp) 对齐到 (y_scale, y_zp)
int8_t nn_qalign_i8(int8_t x_q, float x_scale, int32_t x_zp, float y_scale,
                    int32_t y_zp);

// 量化域相加：a+b，输入/输出可有不同 scale/zp
int8_t nn_qadd_i8(int8_t a_q, float a_scale, int32_t a_zp, int8_t b_q,
                  float b_scale, int32_t b_zp, float out_scale, int32_t out_zp);

void nn_qadd_i8_vec(int8_t *out, const int8_t *a, float a_scale, int32_t a_zp,
                    const int8_t *b, float b_scale, int32_t b_zp,
                    float out_scale, int32_t out_zp, int32_t n);

#endif
