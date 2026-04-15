#ifndef RESNET18_H
#define RESNET18_H

#include "nn_conv2d.h"
#include "nn_fc.h"
#include "nn_image.h"
#include "nn_layer.h"
#include "nn_matrix.h"
#include "nn_stensor.h"
#include <stdbool.h>
#include <stdint.h>

struct res2_id {
    nn_conv2d conv2d;
    nn_matrix mtx[128][64];
    int8_t *weights;
    const int32_t *bias;
};

struct res3_id {
    nn_conv2d conv2d;
    nn_matrix mtx[256][128];
    int8_t *weights;
    const int32_t *bias;
};

struct res4_id {
    nn_conv2d conv2d;
    nn_matrix mtx[512][256];
    int8_t *weights;
    const int32_t *bias;
};

extern struct conv1 {
    nn_conv2d conv2d;
    nn_matrix mtx[64][3];
    int8_t *weights;
    const int32_t *bias;
} conv1;
extern struct res1 l1;
extern struct res2 l2;
extern struct res3 l3;
extern struct res4 l4;
extern struct fc fc1;

struct res1 {
    struct {
        nn_conv2d conv2ds[2];
        nn_matrix mtxes[2][64][64];
        int8_t *weights[2];
        const int32_t *bias[2];
    } res[2];
};

struct res2_1 {
    nn_conv2d conv2d[2];
    nn_matrix mtx1[128][64];
    nn_matrix mtx2[128][128];
    int8_t *weights[2];
    const int32_t *bias[2];
};

struct res2_2 {
    nn_conv2d conv2ds[2];
    nn_matrix mtxes[2][128][128];
    int8_t *weights[2];
    const int32_t *bias[2];
};

struct res2 {
    struct res2_1 res2_1;
    struct res2_2 res2_2;
};

struct res3_1 {
    nn_conv2d conv2d[2];
    nn_matrix mtx1[256][128];
    nn_matrix mtx2[256][256];
    int8_t *weights[2];
    const int32_t *bias[2];
};

struct res3_2 {
    nn_conv2d conv2ds[2];
    nn_matrix mtxes[2][256][256];
    int8_t *weights[2];
    const int32_t *bias[2];
};

struct res3 {
    struct res3_1 res2_1;
    struct res3_2 res2_2;
};

struct res4_1 {
    nn_conv2d conv2d[2];
    nn_matrix mtx1[512][256];
    nn_matrix mtx2[512][512];
    int8_t *weights[2];
    const int32_t *bias[2];
};

struct res4_2 {
    nn_conv2d conv2ds[2];
    nn_matrix mtxes[2][512][512];
    int8_t *weights[2];
    const int32_t *bias[2];
};

struct res4 {
    struct res4_1 res2_1;
    struct res4_2 res2_2;
};

struct fc {
    nn_fc fc;
    int8_t *weight;
    const int32_t *bias;
};

extern nn_layer g_resnet18_layers[22];
extern nn_stensor g_resnet18_io[2];

void init_res_net(void);
void build_resnet18_graph(void);
bool resnet18_infer(nn_image *input, nn_matrix *output);

#endif
