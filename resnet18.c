
#include "nn_conv2d.h"
#include "nn_fc.h"
extern struct conv1 {
    nn_conv2d conv2d;
    nn_matrix mtx[64][3];
} c1;

struct res1 {
    struct {
        nn_conv2d conv2ds[2];
        nn_matrix mtxes[2][64][64];
    } res[2];
} l1;

struct res2 {
    struct res2_id {
        nn_conv2d conv2d;
        nn_matrix mtx[128][64];
    } id;

    struct res2_1 {
        nn_conv2d conv2d[2];
        nn_matrix mtx1[128][64];
        nn_matrix mtx2[128][128];
    } res2_1;

    struct res2_2 {
        nn_conv2d conv2ds[2];
        nn_matrix mtxes[2][128][128];
    } res2_2;
} l2;

struct res3 {
    struct res3_id {
        nn_conv2d conv2d;
        nn_matrix mtx[256][128];
    } id;

    struct res3_1 {
        nn_conv2d conv2d[2];
        nn_matrix mtx1[256][128];
        nn_matrix mtx2[256][256];
    } res2_1;

    struct res3_2 {
        nn_conv2d conv2ds[2];
        nn_matrix mtxes[2][256][256];
    } res2_2;
} l3;

struct res4 {
    struct res4_id {
        nn_conv2d conv2d;
        nn_matrix mtx[512][256];
    } id;

    struct res4_1 {
        nn_conv2d conv2d[2];
        nn_matrix mtx1[512][256];
        nn_matrix mtx2[512][512];
    } res2_1;

    struct res4_2 {
        nn_conv2d conv2ds[2];
        nn_matrix mtxes[2][512][512];
    } res2_2;
} l4;

struct fc {
    nn_fc fc;
};

void init_res_net() {
    nn_conv2d_init(&c1.conv2d, conv1_weight, c1->mtx, 3, 64, 7, 7, 2, 2, 3, 3,
                   conv1_bias, conv1_weight_zp[0], conv1_out_zp);

    // layer 1_0
    nn_conv2d_init(&l1.res[0].conv2ds[0], layer1_0_conv1_weight, &l1->mtx[0],
                   64, 64, 3, 3, 1, 1, 1, 1, layer1_0_conv1_bias,
                   layer1_0_conv1_weight_zp[0], layer1_0_conv1_out_zp);

    nn_conv2d_init(&l1.res[0].conv2ds[1], layer1_0_conv2_weight, &l1->mtx[1],
                   64, 64, 3, 3, 1, 1, 1, 1, layer1_0_conv2_bias,
                   layer1_0_conv2_weight_zp[0], layer1_0_conv2_out_zp);

    // layer 1_1
}
