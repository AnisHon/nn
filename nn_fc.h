/**
 * 全连接层
 * @author：我
 */
#ifndef NN_FC_H
#define NN_FC_H
#include "nn_matrix.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t d_in, d_out;
    nn_matrix *weight;
    const int32_t *bias;
    int32_t weight_zp;
    int32_t out_zp;
} nn_fc;

void nn_fc_init(nn_fc *fc, int8_t *weight, const int32_t *bias, size_t d_in,
                size_t d_out, nn_matrix *weight_mtx, int32_t weight_zp,
                int32_t out_zp);

void nn_fc_forward(const nn_fc *fc, const nn_matrix *in, nn_matrix *out);

#endif
