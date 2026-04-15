#include "resnet18.h"
#include "model_weights.h"
#include "nn_function.h"
#include "nn_utils.h"
#include <stdbool.h>
#include <stdint.h>

struct conv1 conv1;
struct res1 l1;
struct res2 l2;
struct res3 l3;
struct res4 l4;
struct fc fc1;
static struct res2_id l2_id;
static struct res3_id l3_id;
static struct res4_id l4_id;
nn_layer g_resnet18_layers[22];
nn_stensor g_resnet18_io[2];
static bool g_graph_ready = false;

static const float S0 = 0.025440f, S1 = 0.030989f, S2 = 0.091684f,
                   S3 = 0.017408f, S4 = 0.083024f, S5 = 0.025067f,
                   S6 = 0.070420f, S7 = 0.017507f, S8 = 0.021187f,
                   S9 = 0.061251f, S10 = 0.025773f, S11 = 0.071879f,
                   S12 = 0.018311f, S13 = 0.022495f, S14 = 0.065245f,
                   S15 = 0.019454f, S16 = 0.056956f, S17 = 0.015274f,
                   S18 = 0.019256f, S19 = 0.176500f;

static void setup_image(nn_image *img, nn_matrix *m, int8_t *b, size_t c,
                        size_t h, size_t w, int32_t zp) {
    size_t hw = h * w;
    for (size_t i = 0; i < c; ++i) nn_mtx_init(&m[i], h, w, b + i * hw, zp);
    nn_image_init(img, c, h, w, m, zp);
}

static void add_relu(nn_image *dst, const nn_image *a, float as, const nn_image *b,
                     float bs) {
    nn_qadd_i8_vec(dst->mtx[0].elem, a->mtx[0].elem, as, a->zp, b->mtx[0].elem,
                   bs, b->zp, as, dst->zp,
                   (int32_t)(dst->channels * dst->h * dst->w));
    nn_relu_image(dst, dst);
}

static void block(nn_conv2d *c1, float s1, nn_conv2d *c2, float s2,
                  nn_conv2d *down, float sd, nn_image *src, float *srcs,
                  nn_image *t0, nn_image *t1, nn_image *skip) {
    (void)s1;
    nn_conv2d_forward(c1, src, t0);
    nn_relu_image(t0, t0);
    nn_conv2d_forward(c2, t0, t1);
    if (down) {
        nn_conv2d_forward(down, src, skip);
        add_relu(t1, t1, s2, skip, sd);
    } else {
        add_relu(t1, t1, s2, src, *srcs);
    }
    *srcs = s2;
}

void init_res_net(void) {
#define IC(dst,w,mtx,ci,co,kh,kw,sh,sw,ph,pw,bias,wzp,ozp) \
    nn_conv2d_init(&(dst), (int8_t *)(w), (mtx), (ci), (co), (kh), (kw), (sh), \
                   (sw), (ph), (pw), (bias), (wzp), (ozp))
    IC(conv1.conv2d, conv1_weight, &conv1.mtx[0][0], 3, 64, 3, 3, 1, 1, 1, 1,
       conv1_bias, conv1_weight_zp[0], conv1_out_zp);
    IC(l1.res[0].conv2ds[0], layer1_0_conv1_weight, &l1.res[0].mtxes[0][0][0],
       64, 64, 3, 3, 1, 1, 1, 1, layer1_0_conv1_bias,
       layer1_0_conv1_weight_zp[0], layer1_0_conv1_out_zp);
    IC(l1.res[0].conv2ds[1], layer1_0_conv2_weight, &l1.res[0].mtxes[1][0][0],
       64, 64, 3, 3, 1, 1, 1, 1, layer1_0_conv2_bias,
       layer1_0_conv2_weight_zp[0], layer1_0_conv2_out_zp);
    IC(l1.res[1].conv2ds[0], layer1_1_conv1_weight, &l1.res[1].mtxes[0][0][0],
       64, 64, 3, 3, 1, 1, 1, 1, layer1_1_conv1_bias,
       layer1_1_conv1_weight_zp[0], layer1_1_conv1_out_zp);
    IC(l1.res[1].conv2ds[1], layer1_1_conv2_weight, &l1.res[1].mtxes[1][0][0],
       64, 64, 3, 3, 1, 1, 1, 1, layer1_1_conv2_bias,
       layer1_1_conv2_weight_zp[0], layer1_1_conv2_out_zp);
    IC(l2.res2_1.conv2d[0], layer2_0_conv1_weight, &l2.res2_1.mtx1[0][0], 64,
       128, 3, 3, 2, 2, 1, 1, layer2_0_conv1_bias, layer2_0_conv1_weight_zp[0],
       layer2_0_conv1_out_zp);
    IC(l2.res2_1.conv2d[1], layer2_0_conv2_weight, &l2.res2_1.mtx2[0][0], 128,
       128, 3, 3, 1, 1, 1, 1, layer2_0_conv2_bias, layer2_0_conv2_weight_zp[0],
       layer2_0_conv2_out_zp);
    IC(l2_id.conv2d, layer2_0_downsample_0_weight, &l2_id.mtx[0][0], 64, 128,
       1, 1, 2, 2, 0, 0, layer2_0_downsample_0_bias,
       layer2_0_downsample_0_weight_zp[0], layer2_0_downsample_0_out_zp);
    IC(l2.res2_2.conv2ds[0], layer2_1_conv1_weight, &l2.res2_2.mtxes[0][0][0],
       128, 128, 3, 3, 1, 1, 1, 1, layer2_1_conv1_bias,
       layer2_1_conv1_weight_zp[0], layer2_1_conv1_out_zp);
    IC(l2.res2_2.conv2ds[1], layer2_1_conv2_weight, &l2.res2_2.mtxes[1][0][0],
       128, 128, 3, 3, 1, 1, 1, 1, layer2_1_conv2_bias,
       layer2_1_conv2_weight_zp[0], layer2_1_conv2_out_zp);
    IC(l3.res2_1.conv2d[0], layer3_0_conv1_weight, &l3.res2_1.mtx1[0][0], 128,
       256, 3, 3, 2, 2, 1, 1, layer3_0_conv1_bias, layer3_0_conv1_weight_zp[0],
       layer3_0_conv1_out_zp);
    IC(l3.res2_1.conv2d[1], layer3_0_conv2_weight, &l3.res2_1.mtx2[0][0], 256,
       256, 3, 3, 1, 1, 1, 1, layer3_0_conv2_bias, layer3_0_conv2_weight_zp[0],
       layer3_0_conv2_out_zp);
    IC(l3_id.conv2d, layer3_0_downsample_0_weight, &l3_id.mtx[0][0], 128, 256,
       1, 1, 2, 2, 0, 0, layer3_0_downsample_0_bias,
       layer3_0_downsample_0_weight_zp[0], layer3_0_downsample_0_out_zp);
    IC(l3.res2_2.conv2ds[0], layer3_1_conv1_weight, &l3.res2_2.mtxes[0][0][0],
       256, 256, 3, 3, 1, 1, 1, 1, layer3_1_conv1_bias,
       layer3_1_conv1_weight_zp[0], layer3_1_conv1_out_zp);
    IC(l3.res2_2.conv2ds[1], layer3_1_conv2_weight, &l3.res2_2.mtxes[1][0][0],
       256, 256, 3, 3, 1, 1, 1, 1, layer3_1_conv2_bias,
       layer3_1_conv2_weight_zp[0], layer3_1_conv2_out_zp);
    IC(l4.res2_1.conv2d[0], layer4_0_conv1_weight, &l4.res2_1.mtx1[0][0], 256,
       512, 3, 3, 2, 2, 1, 1, layer4_0_conv1_bias, layer4_0_conv1_weight_zp[0],
       layer4_0_conv1_out_zp);
    IC(l4.res2_1.conv2d[1], layer4_0_conv2_weight, &l4.res2_1.mtx2[0][0], 512,
       512, 3, 3, 1, 1, 1, 1, layer4_0_conv2_bias, layer4_0_conv2_weight_zp[0],
       layer4_0_conv2_out_zp);
    IC(l4_id.conv2d, layer4_0_downsample_0_weight, &l4_id.mtx[0][0], 256, 512,
       1, 1, 2, 2, 0, 0, layer4_0_downsample_0_bias,
       layer4_0_downsample_0_weight_zp[0], layer4_0_downsample_0_out_zp);
    IC(l4.res2_2.conv2ds[0], layer4_1_conv1_weight, &l4.res2_2.mtxes[0][0][0],
       512, 512, 3, 3, 1, 1, 1, 1, layer4_1_conv1_bias,
       layer4_1_conv1_weight_zp[0], layer4_1_conv1_out_zp);
    IC(l4.res2_2.conv2ds[1], layer4_1_conv2_weight, &l4.res2_2.mtxes[1][0][0],
       512, 512, 3, 3, 1, 1, 1, 1, layer4_1_conv2_bias,
       layer4_1_conv2_weight_zp[0], layer4_1_conv2_out_zp);
#undef IC
    static nn_matrix fcw;
    nn_fc_init(&fc1.fc, (int8_t *)fc_weight, fc_bias, 512, 200, &fcw,
               fc_weight_zp[0], fc_out_zp);
    g_graph_ready = true;
}

void build_resnet18_graph(void) {}

bool resnet18_infer(nn_image *input, nn_matrix *output) {
    if (!input || !output) return false;
    if (input->channels != 3 || input->h != 64 || input->w != 64) return false;
    if (output->h != 1 || output->w != 200 || output->zp != fc_out_zp)
        return false;
    if (!g_graph_ready) init_res_net();

    static int8_t b0[512 * 64 * 64], b1[512 * 64 * 64], b2[512 * 64 * 64];
    static int8_t bs[512 * 32 * 32], poolb[512], flatb[512];
    static nn_matrix m0[512], m1[512], m2[512], ms[512], pm[512], flatm;
    static nn_image i0, i1, i2, is, pool;
    float cs = S0;

    setup_image(&i0, m0, b0, 64, 64, 64, conv1_out_zp);
    nn_conv2d_forward(&conv1.conv2d, input, &i0);
    nn_relu_image(&i0, &i0);
    nn_image *cur = &i0;

    setup_image(&i1, m1, b1, 64, 64, 64, layer1_0_conv1_out_zp);
    setup_image(&i2, m2, b2, 64, 64, 64, layer1_0_conv2_out_zp);
    block(&l1.res[0].conv2ds[0], S1, &l1.res[0].conv2ds[1], S2, 0, 0, cur, &cs,
          &i1, &i2, &is);
    cur = &i2;

    setup_image(&i0, m0, b0, 64, 64, 64, layer1_1_conv1_out_zp);
    setup_image(&i1, m1, b1, 64, 64, 64, layer1_1_conv2_out_zp);
    block(&l1.res[1].conv2ds[0], S3, &l1.res[1].conv2ds[1], S4, 0, 0, cur, &cs,
          &i0, &i1, &is);
    cur = &i1;

    setup_image(&i2, m2, b2, 128, 32, 32, layer2_0_conv1_out_zp);
    setup_image(&i0, m0, b0, 128, 32, 32, layer2_0_conv2_out_zp);
    setup_image(&is, ms, bs, 128, 32, 32, layer2_0_downsample_0_out_zp);
    block(&l2.res2_1.conv2d[0], S5, &l2.res2_1.conv2d[1], S6, &l2_id.conv2d, S7,
          cur, &cs, &i2, &i0, &is);
    cur = &i0;

    setup_image(&i1, m1, b1, 128, 32, 32, layer2_1_conv1_out_zp);
    setup_image(&i2, m2, b2, 128, 32, 32, layer2_1_conv2_out_zp);
    block(&l2.res2_2.conv2ds[0], S8, &l2.res2_2.conv2ds[1], S9, 0, 0, cur, &cs,
          &i1, &i2, &is);
    cur = &i2;

    setup_image(&i0, m0, b0, 256, 16, 16, layer3_0_conv1_out_zp);
    setup_image(&i1, m1, b1, 256, 16, 16, layer3_0_conv2_out_zp);
    setup_image(&is, ms, bs, 256, 16, 16, layer3_0_downsample_0_out_zp);
    block(&l3.res2_1.conv2d[0], S10, &l3.res2_1.conv2d[1], S11, &l3_id.conv2d,
          S12, cur, &cs, &i0, &i1, &is);
    cur = &i1;

    setup_image(&i2, m2, b2, 256, 16, 16, layer3_1_conv1_out_zp);
    setup_image(&i0, m0, b0, 256, 16, 16, layer3_1_conv2_out_zp);
    block(&l3.res2_2.conv2ds[0], S13, &l3.res2_2.conv2ds[1], S14, 0, 0, cur, &cs,
          &i2, &i0, &is);
    cur = &i0;

    setup_image(&i1, m1, b1, 512, 8, 8, layer4_0_conv1_out_zp);
    setup_image(&i2, m2, b2, 512, 8, 8, layer4_0_conv2_out_zp);
    setup_image(&is, ms, bs, 512, 8, 8, layer4_0_downsample_0_out_zp);
    block(&l4.res2_1.conv2d[0], S15, &l4.res2_1.conv2d[1], S16, &l4_id.conv2d,
          S17, cur, &cs, &i1, &i2, &is);
    cur = &i2;

    setup_image(&i0, m0, b0, 512, 8, 8, layer4_1_conv1_out_zp);
    setup_image(&i1, m1, b1, 512, 8, 8, layer4_1_conv2_out_zp);
    block(&l4.res2_2.conv2ds[0], S18, &l4.res2_2.conv2ds[1], S19, 0, 0, cur, &cs,
          &i0, &i1, &is);
    cur = &i1;

    setup_image(&pool, pm, poolb, 512, 1, 1, cur->zp);
    nn_avg_pool(cur, &pool);
    nn_mtx_init(&flatm, 1, 512, flatb, pool.zp);
    nn_flatten(&pool, &flatm);
    nn_fc_forward(&fc1.fc, &flatm, output);
    g_resnet18_io[0] = (nn_stensor){.tensor.mtx = &flatm, .tensor_type = NN_STENSOR_MTX};
    g_resnet18_io[1] = (nn_stensor){.tensor.mtx = output, .tensor_type = NN_STENSOR_MTX};
    return true;
}
