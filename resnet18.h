#ifndef RESNET18_H
#define RESNET18_H

#include "nn_conv2d.h"
#include "nn_fc.h"
#include "nn_matrix.h"
//
// #define DEFINE_RES_BLOCK(name, conv_num, chan) \
//     struct { \
//         nn_conv2d conv2ds[(conv_num)]; \
//         nn_matrix mtxes[(conv_num)][(chan)][(chan)]; \
//         int8_t *weights[(conv_num)]; \
//         int8_t *bias[(conv_num)]; \
//     } #name

// #define DEFINE_PROJ_RES_BLOCK(name, conv_num, in_chan, out_chan) \
//     struct { \
//         nn_conv2d conv2ds[(conv_num)]; \
//         nn_matrix mtxes[(conv_num)][(chan_out)][(chan_in)]; \
//         int8_t *weights[(conv_num)]; \
//         int8_t *bias[(conv_num)]; \
//         nn_conv2d *identity; \
//         int8_t *id_weight[(conv_num)]; \
//         int8_t *id_bias[(conv_num)]; \
//     }
// #name

extern struct conv1 {
    nn_conv2d conv2d;
    nn_matrix mtx[64][3];
    int8_t *weights;
    int8_t *bias;
} conv1;

struct res1 {
    struct {
        nn_conv2d conv2ds[2];
        nn_matrix mtxes[2][64][64];
        int8_t *weights[2];
        int8_t *bias[2];
    } res[2];
};

struct res2 {
    struct res2_id {
        nn_conv2d conv2d;
        nn_matrix mtx[128][64];
        int8_t *weights;
        int8_t *bias;
    };

    struct res2_1 {
        nn_conv2d conv2d[2];
        nn_matrix mtx1[128][64];
        nn_matrix mtx2[128][128];
        int8_t *weights[2];
        int8_t *bias[2];
    } res2_1;

    struct res2_2 {
        nn_conv2d conv2ds[2];
        nn_matrix mtxes[2][128][128];
        int8_t *weights[2];
        int8_t *bias[2];
    } res2_2;
};

struct res3 {
    struct res3_id {
        nn_conv2d conv2d;
        nn_matrix mtx[256][128];
        int8_t *weights;
        int8_t *bias;
    };

    struct res3_1 {
        nn_conv2d conv2d[2];
        nn_matrix mtx1[256][128];
        nn_matrix mtx2[256][256];
        int8_t *weights[2];
        int8_t *bias[2];
    } res2_1;

    struct res3_2 {
        nn_conv2d conv2ds[2];
        nn_matrix mtxes[2][256][256];
        int8_t *weights[2];
        int8_t *bias[2];
    } res2_2;
};

struct res4 {
    struct res4_id {
        nn_conv2d conv2d;
        nn_matrix mtx[512][256];
        int8_t *weights;
        int8_t *bias;
    };

    struct res4_1 {
        nn_conv2d conv2d[2];
        nn_matrix mtx1[512][256];
        nn_matrix mtx2[512][512];
        int8_t *weights[2];
        int8_t *bias[2];
    } res2_1;

    struct res4_2 {
        nn_conv2d conv2ds[2];
        nn_matrix mtxes[2][512][512];
        int8_t *weights[2];
        int8_t *bias[2];
    } res2_2;
};

struct fc {
    nn_fc fc;
    int8_t *weight;
    int8_t *bias;
};

void init_res_net();

#endif //! RESNET18_H
