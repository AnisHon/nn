#include "resnet18.h"
#include "model_weights.h"
#include "nn_function.h"
#include "nn_utils.h"
#include <assert.h>
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

void init_res_net(void) {
    // conv1
    nn_conv2d_init(&conv1.conv2d, (int8_t *)conv1_weight, &conv1.mtx[0][0], 3,
                   64, 3, 3, 1, 1, 1, 1, conv1_bias, conv1_weight_zp[0],
                   conv1_out_zp);
    conv1.weights = (int8_t *)conv1_weight;
    conv1.bias = (int8_t *)conv1_bias;

    // layer1.0
    nn_conv2d_init(&l1.res[0].conv2ds[0], (int8_t *)layer1_0_conv1_weight,
                   &l1.res[0].mtxes[0][0][0], 64, 64, 3, 3, 1, 1, 1, 1,
                   layer1_0_conv1_bias, layer1_0_conv1_weight_zp[0],
                   layer1_0_conv1_out_zp);
    l1.res[0].weights[0] = (int8_t *)layer1_0_conv1_weight;
    l1.res[0].bias[0] = (int8_t *)layer1_0_conv1_bias;

    nn_conv2d_init(&l1.res[0].conv2ds[1], (int8_t *)layer1_0_conv2_weight,
                   &l1.res[0].mtxes[1][0][0], 64, 64, 3, 3, 1, 1, 1, 1,
                   layer1_0_conv2_bias, layer1_0_conv2_weight_zp[0],
                   layer1_0_conv2_out_zp);
    l1.res[0].weights[1] = (int8_t *)layer1_0_conv2_weight;
    l1.res[0].bias[1] = (int8_t *)layer1_0_conv2_bias;

    // layer1.1
    nn_conv2d_init(&l1.res[1].conv2ds[0], (int8_t *)layer1_1_conv1_weight,
                   &l1.res[1].mtxes[0][0][0], 64, 64, 3, 3, 1, 1, 1, 1,
                   layer1_1_conv1_bias, layer1_1_conv1_weight_zp[0],
                   layer1_1_conv1_out_zp);
    l1.res[1].weights[0] = (int8_t *)layer1_1_conv1_weight;
    l1.res[1].bias[0] = (int8_t *)layer1_1_conv1_bias;

    nn_conv2d_init(&l1.res[1].conv2ds[1], (int8_t *)layer1_1_conv2_weight,
                   &l1.res[1].mtxes[1][0][0], 64, 64, 3, 3, 1, 1, 1, 1,
                   layer1_1_conv2_bias, layer1_1_conv2_weight_zp[0],
                   layer1_1_conv2_out_zp);
    l1.res[1].weights[1] = (int8_t *)layer1_1_conv2_weight;
    l1.res[1].bias[1] = (int8_t *)layer1_1_conv2_bias;

    // layer2.0 + downsample
    nn_conv2d_init(&l2.res2_1.conv2d[0], (int8_t *)layer2_0_conv1_weight,
                   &l2.res2_1.mtx1[0][0], 64, 128, 3, 3, 2, 2, 1, 1,
                   layer2_0_conv1_bias, layer2_0_conv1_weight_zp[0],
                   layer2_0_conv1_out_zp);
    l2.res2_1.weights[0] = (int8_t *)layer2_0_conv1_weight;
    l2.res2_1.bias[0] = (int8_t *)layer2_0_conv1_bias;

    nn_conv2d_init(&l2.res2_1.conv2d[1], (int8_t *)layer2_0_conv2_weight,
                   &l2.res2_1.mtx2[0][0], 128, 128, 3, 3, 1, 1, 1, 1,
                   layer2_0_conv2_bias, layer2_0_conv2_weight_zp[0],
                   layer2_0_conv2_out_zp);
    l2.res2_1.weights[1] = (int8_t *)layer2_0_conv2_weight;
    l2.res2_1.bias[1] = (int8_t *)layer2_0_conv2_bias;

    nn_conv2d_init(
        &l2_id.conv2d, (int8_t *)layer2_0_downsample_0_weight, &l2_id.mtx[0][0],
        64, 128, 1, 1, 2, 2, 0, 0, layer2_0_downsample_0_bias,
        layer2_0_downsample_0_weight_zp[0], layer2_0_downsample_0_out_zp);
    l2_id.weights = (int8_t *)layer2_0_downsample_0_weight;
    l2_id.bias = (int8_t *)layer2_0_downsample_0_bias;

    // layer2.1
    nn_conv2d_init(&l2.res2_2.conv2ds[0], (int8_t *)layer2_1_conv1_weight,
                   &l2.res2_2.mtxes[0][0][0], 128, 128, 3, 3, 1, 1, 1, 1,
                   layer2_1_conv1_bias, layer2_1_conv1_weight_zp[0],
                   layer2_1_conv1_out_zp);
    l2.res2_2.weights[0] = (int8_t *)layer2_1_conv1_weight;
    l2.res2_2.bias[0] = (int8_t *)layer2_1_conv1_bias;

    nn_conv2d_init(&l2.res2_2.conv2ds[1], (int8_t *)layer2_1_conv2_weight,
                   &l2.res2_2.mtxes[1][0][0], 128, 128, 3, 3, 1, 1, 1, 1,
                   layer2_1_conv2_bias, layer2_1_conv2_weight_zp[0],
                   layer2_1_conv2_out_zp);
    l2.res2_2.weights[1] = (int8_t *)layer2_1_conv2_weight;
    l2.res2_2.bias[1] = (int8_t *)layer2_1_conv2_bias;

    // layer3.0 + downsample
    nn_conv2d_init(&l3.res2_1.conv2d[0], (int8_t *)layer3_0_conv1_weight,
                   &l3.res2_1.mtx1[0][0], 128, 256, 3, 3, 2, 2, 1, 1,
                   layer3_0_conv1_bias, layer3_0_conv1_weight_zp[0],
                   layer3_0_conv1_out_zp);
    l3.res2_1.weights[0] = (int8_t *)layer3_0_conv1_weight;
    l3.res2_1.bias[0] = (int8_t *)layer3_0_conv1_bias;

    nn_conv2d_init(&l3.res2_1.conv2d[1], (int8_t *)layer3_0_conv2_weight,
                   &l3.res2_1.mtx2[0][0], 256, 256, 3, 3, 1, 1, 1, 1,
                   layer3_0_conv2_bias, layer3_0_conv2_weight_zp[0],
                   layer3_0_conv2_out_zp);
    l3.res2_1.weights[1] = (int8_t *)layer3_0_conv2_weight;
    l3.res2_1.bias[1] = (int8_t *)layer3_0_conv2_bias;

    nn_conv2d_init(
        &l3_id.conv2d, (int8_t *)layer3_0_downsample_0_weight, &l3_id.mtx[0][0],
        128, 256, 1, 1, 2, 2, 0, 0, layer3_0_downsample_0_bias,
        layer3_0_downsample_0_weight_zp[0], layer3_0_downsample_0_out_zp);
    l3_id.weights = (int8_t *)layer3_0_downsample_0_weight;
    l3_id.bias = (int8_t *)layer3_0_downsample_0_bias;

    // layer3.1
    nn_conv2d_init(&l3.res2_2.conv2ds[0], (int8_t *)layer3_1_conv1_weight,
                   &l3.res2_2.mtxes[0][0][0], 256, 256, 3, 3, 1, 1, 1, 1,
                   layer3_1_conv1_bias, layer3_1_conv1_weight_zp[0],
                   layer3_1_conv1_out_zp);
    l3.res2_2.weights[0] = (int8_t *)layer3_1_conv1_weight;
    l3.res2_2.bias[0] = (int8_t *)layer3_1_conv1_bias;

    nn_conv2d_init(&l3.res2_2.conv2ds[1], (int8_t *)layer3_1_conv2_weight,
                   &l3.res2_2.mtxes[1][0][0], 256, 256, 3, 3, 1, 1, 1, 1,
                   layer3_1_conv2_bias, layer3_1_conv2_weight_zp[0],
                   layer3_1_conv2_out_zp);
    l3.res2_2.weights[1] = (int8_t *)layer3_1_conv2_weight;
    l3.res2_2.bias[1] = (int8_t *)layer3_1_conv2_bias;

    // layer4.0 + downsample
    nn_conv2d_init(&l4.res2_1.conv2d[0], (int8_t *)layer4_0_conv1_weight,
                   &l4.res2_1.mtx1[0][0], 256, 512, 3, 3, 2, 2, 1, 1,
                   layer4_0_conv1_bias, layer4_0_conv1_weight_zp[0],
                   layer4_0_conv1_out_zp);
    l4.res2_1.weights[0] = (int8_t *)layer4_0_conv1_weight;
    l4.res2_1.bias[0] = (int8_t *)layer4_0_conv1_bias;

    nn_conv2d_init(&l4.res2_1.conv2d[1], (int8_t *)layer4_0_conv2_weight,
                   &l4.res2_1.mtx2[0][0], 512, 512, 3, 3, 1, 1, 1, 1,
                   layer4_0_conv2_bias, layer4_0_conv2_weight_zp[0],
                   layer4_0_conv2_out_zp);
    l4.res2_1.weights[1] = (int8_t *)layer4_0_conv2_weight;
    l4.res2_1.bias[1] = (int8_t *)layer4_0_conv2_bias;

    nn_conv2d_init(
        &l4_id.conv2d, (int8_t *)layer4_0_downsample_0_weight, &l4_id.mtx[0][0],
        256, 512, 1, 1, 2, 2, 0, 0, layer4_0_downsample_0_bias,
        layer4_0_downsample_0_weight_zp[0], layer4_0_downsample_0_out_zp);
    l4_id.weights = (int8_t *)layer4_0_downsample_0_weight;
    l4_id.bias = (int8_t *)layer4_0_downsample_0_bias;

    // layer4.1
    nn_conv2d_init(&l4.res2_2.conv2ds[0], (int8_t *)layer4_1_conv1_weight,
                   &l4.res2_2.mtxes[0][0][0], 512, 512, 3, 3, 1, 1, 1, 1,
                   layer4_1_conv1_bias, layer4_1_conv1_weight_zp[0],
                   layer4_1_conv1_out_zp);
    l4.res2_2.weights[0] = (int8_t *)layer4_1_conv1_weight;
    l4.res2_2.bias[0] = (int8_t *)layer4_1_conv1_bias;

    nn_conv2d_init(&l4.res2_2.conv2ds[1], (int8_t *)layer4_1_conv2_weight,
                   &l4.res2_2.mtxes[1][0][0], 512, 512, 3, 3, 1, 1, 1, 1,
                   layer4_1_conv2_bias, layer4_1_conv2_weight_zp[0],
                   layer4_1_conv2_out_zp);
    l4.res2_2.weights[1] = (int8_t *)layer4_1_conv2_weight;
    l4.res2_2.bias[1] = (int8_t *)layer4_1_conv2_bias;

    // fc
    static nn_matrix fc_weight_mtx;
    static nn_matrix fc_bias_mtx;
    nn_fc_init(&fc1.fc, (int8_t *)fc_weight, (int8_t *)fc_bias, 512, 200,
               &fc_weight_mtx, &fc_bias_mtx, fc_weight_zp[0], fc_out_zp);
    fc1.weight = (int8_t *)fc_weight;
    fc1.bias = (int8_t *)fc_bias;
}

void build_resnet18_graph(void) {
    // convs
    g_resnet18_layers[0] =
        (nn_layer){.layer.conv2d = &conv1.conv2d, .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[1] = (nn_layer){.layer.conv2d = &l1.res[0].conv2ds[0],
                                      .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[2] = (nn_layer){.layer.conv2d = &l1.res[0].conv2ds[1],
                                      .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[3] = (nn_layer){.layer.conv2d = &l1.res[1].conv2ds[0],
                                      .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[4] = (nn_layer){.layer.conv2d = &l1.res[1].conv2ds[1],
                                      .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[5] = (nn_layer){.layer.conv2d = &l2.res2_1.conv2d[0],
                                      .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[6] = (nn_layer){.layer.conv2d = &l2.res2_1.conv2d[1],
                                      .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[7] =
        (nn_layer){.layer.conv2d = &l2_id.conv2d, .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[8] = (nn_layer){.layer.conv2d = &l2.res2_2.conv2ds[0],
                                      .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[9] = (nn_layer){.layer.conv2d = &l2.res2_2.conv2ds[1],
                                      .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[10] = (nn_layer){.layer.conv2d = &l3.res2_1.conv2d[0],
                                       .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[11] = (nn_layer){.layer.conv2d = &l3.res2_1.conv2d[1],
                                       .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[12] =
        (nn_layer){.layer.conv2d = &l3_id.conv2d, .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[13] = (nn_layer){.layer.conv2d = &l3.res2_2.conv2ds[0],
                                       .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[14] = (nn_layer){.layer.conv2d = &l3.res2_2.conv2ds[1],
                                       .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[15] = (nn_layer){.layer.conv2d = &l4.res2_1.conv2d[0],
                                       .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[16] = (nn_layer){.layer.conv2d = &l4.res2_1.conv2d[1],
                                       .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[17] =
        (nn_layer){.layer.conv2d = &l4_id.conv2d, .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[18] = (nn_layer){.layer.conv2d = &l4.res2_2.conv2ds[0],
                                       .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[19] = (nn_layer){.layer.conv2d = &l4.res2_2.conv2ds[1],
                                       .layer_type = NN_LAYER_CONV};
    g_resnet18_layers[20] = (nn_layer){.layer_type = NN_LAYER_FLATTEN};
    g_resnet18_layers[21] =
        (nn_layer){.layer.fc = &fc1.fc, .layer_type = NN_LAYER_FC};

    g_graph_ready = true;
}

static void setup_image(nn_image *img, nn_matrix *mats, int8_t *buf, size_t c,
                        size_t h, size_t w, int32_t zp) {
    const size_t hw = h * w;
    for (size_t i = 0; i < c; i++) {
        nn_mtx_init(&mats[i], h, w, buf + i * hw, zp);
    }
    nn_image_init(img, c, h, w, mats, zp);
}

static void run_conv_layer(const nn_layer *layer, nn_image *in, nn_image *out) {
    nn_stensor ti = {.tensor.img = in, .tensor_type = NN_STENSOR_IMG};
    nn_stensor to = {.tensor.img = out, .tensor_type = NN_STENSOR_IMG};
    nn_forward(layer, &ti, &to);
}

static void add_residual_relu(nn_image *dst, const nn_image *main_in,
                              float main_scale, const nn_image *skip_in,
                              float skip_scale) {
    const int32_t n = (int32_t)(dst->channels * dst->h * dst->w);
    nn_qadd_i8_vec(dst->mtx[0].elem, main_in->mtx[0].elem, main_scale,
                   main_in->zp, skip_in->mtx[0].elem, skip_scale, skip_in->zp,
                   main_scale, dst->zp, n);
    nn_relu_image(dst, dst);
}

static void run_block(const nn_layer *conv1, const nn_layer *conv2,
                      const nn_layer *downsample_or_null, float conv1_out_scale,
                      float conv2_out_scale, float downsample_out_scale,
                      nn_image *cur_img, float *cur_scale, nn_image *buf_a,
                      nn_image *buf_b, nn_image *buf_skip) {
    run_conv_layer(conv1, cur_img, buf_a);
    nn_relu_image(buf_a, buf_a);
    run_conv_layer(conv2, buf_a, buf_b);

    if (downsample_or_null != NULL) {
        run_conv_layer(downsample_or_null, cur_img, buf_skip);
        add_residual_relu(buf_b, buf_b, conv2_out_scale, buf_skip,
                          downsample_out_scale);
    } else {
        add_residual_relu(buf_b, buf_b, conv2_out_scale, cur_img, *cur_scale);
    }
    *cur_scale = conv2_out_scale;
}

bool resnet18_infer(nn_image *input, nn_matrix *output) {
    if (input == NULL || output == NULL) {
        return false;
    }
    if (output->h != 1 || output->w != 200) {
        return false;
    }
    if (output->zp != fc_out_zp) {
        return false;
    }

    if (!g_graph_ready) {
        init_res_net();
        build_resnet18_graph();
    }

    static int8_t buf0[64 * 224 * 224];
    static int8_t buf1[64 * 224 * 224];
    static int8_t buf_skip[64 * 224 * 224];
    static nn_matrix mats0[512];
    static nn_matrix mats1[512];
    static nn_matrix mats_skip[512];
    static int8_t pooled_buf[512];
    static nn_matrix pooled_mats[512];
    static int8_t flat_buf[512];
    static nn_matrix flat_mtx;
    static nn_image img0, img1, img_skip, pooled_img;

    // stem
    setup_image(&img0, mats0, buf0, 64, input->h, input->w, conv1_out_zp);
    run_conv_layer(&g_resnet18_layers[0], input, &img0);
    nn_relu_image(&img0, &img0);

    nn_image *cur = &img0;
    nn_image *a = &img1;
    nn_image *b = &img0;
    nn_image *s = &img_skip;
    float cur_scale = conv1_out_scale;

    // layer1.0
    setup_image(a, mats1, buf1, 64, cur->h, cur->w, layer1_0_conv1_out_zp);
    setup_image(b, mats0, buf0, 64, cur->h, cur->w, layer1_0_conv2_out_zp);
    run_block(&g_resnet18_layers[1], &g_resnet18_layers[2], NULL,
              layer1_0_conv1_out_scale, layer1_0_conv2_out_scale, 0.0f, cur,
              &cur_scale, a, b, s);
    cur = b;

    // layer1.1
    setup_image(a, mats1, buf1, 64, cur->h, cur->w, layer1_1_conv1_out_zp);
    setup_image(b, mats0, buf0, 64, cur->h, cur->w, layer1_1_conv2_out_zp);
    run_block(&g_resnet18_layers[3], &g_resnet18_layers[4], NULL,
              layer1_1_conv1_out_scale, layer1_1_conv2_out_scale, 0.0f, cur,
              &cur_scale, a, b, s);
    cur = b;

    // layer2.0
    setup_image(a, mats1, buf1, 128, cur->h / 2, cur->w / 2,
                layer2_0_conv1_out_zp);
    setup_image(b, mats0, buf0, 128, cur->h / 2, cur->w / 2,
                layer2_0_conv2_out_zp);
    setup_image(s, mats_skip, buf_skip, 128, cur->h / 2, cur->w / 2,
                layer2_0_downsample_0_out_zp);
    run_block(&g_resnet18_layers[5], &g_resnet18_layers[6],
              &g_resnet18_layers[7], layer2_0_conv1_out_scale,
              layer2_0_conv2_out_scale, layer2_0_downsample_0_out_scale, cur,
              &cur_scale, a, b, s);
    cur = b;

    // layer2.1
    setup_image(a, mats1, buf1, 128, cur->h, cur->w, layer2_1_conv1_out_zp);
    setup_image(b, mats0, buf0, 128, cur->h, cur->w, layer2_1_conv2_out_zp);
    run_block(&g_resnet18_layers[8], &g_resnet18_layers[9], NULL,
              layer2_1_conv1_out_scale, layer2_1_conv2_out_scale, 0.0f, cur,
              &cur_scale, a, b, s);
    cur = b;

    // layer3.0
    setup_image(a, mats1, buf1, 256, cur->h / 2, cur->w / 2,
                layer3_0_conv1_out_zp);
    setup_image(b, mats0, buf0, 256, cur->h / 2, cur->w / 2,
                layer3_0_conv2_out_zp);
    setup_image(s, mats_skip, buf_skip, 256, cur->h / 2, cur->w / 2,
                layer3_0_downsample_0_out_zp);
    run_block(&g_resnet18_layers[10], &g_resnet18_layers[11],
              &g_resnet18_layers[12], layer3_0_conv1_out_scale,
              layer3_0_conv2_out_scale, layer3_0_downsample_0_out_scale, cur,
              &cur_scale, a, b, s);
    cur = b;

    // layer3.1
    setup_image(a, mats1, buf1, 256, cur->h, cur->w, layer3_1_conv1_out_zp);
    setup_image(b, mats0, buf0, 256, cur->h, cur->w, layer3_1_conv2_out_zp);
    run_block(&g_resnet18_layers[13], &g_resnet18_layers[14], NULL,
              layer3_1_conv1_out_scale, layer3_1_conv2_out_scale, 0.0f, cur,
              &cur_scale, a, b, s);
    cur = b;

    // layer4.0
    setup_image(a, mats1, buf1, 512, cur->h / 2, cur->w / 2,
                layer4_0_conv1_out_zp);
    setup_image(b, mats0, buf0, 512, cur->h / 2, cur->w / 2,
                layer4_0_conv2_out_zp);
    setup_image(s, mats_skip, buf_skip, 512, cur->h / 2, cur->w / 2,
                layer4_0_downsample_0_out_zp);
    run_block(&g_resnet18_layers[15], &g_resnet18_layers[16],
              &g_resnet18_layers[17], layer4_0_conv1_out_scale,
              layer4_0_conv2_out_scale, layer4_0_downsample_0_out_scale, cur,
              &cur_scale, a, b, s);
    cur = b;

    // layer4.1
    setup_image(a, mats1, buf1, 512, cur->h, cur->w, layer4_1_conv1_out_zp);
    setup_image(b, mats0, buf0, 512, cur->h, cur->w, layer4_1_conv2_out_zp);
    run_block(&g_resnet18_layers[18], &g_resnet18_layers[19], NULL,
              layer4_1_conv1_out_scale, layer4_1_conv2_out_scale, 0.0f, cur,
              &cur_scale, a, b, s);
    cur = b;

    // GAP -> flatten -> FC
    setup_image(&pooled_img, pooled_mats, pooled_buf, 512, 1, 1, cur->zp);
    nn_avg_pool(cur, &pooled_img);
    nn_mtx_init(&flat_mtx, 1, 512, flat_buf, pooled_img.zp);
    nn_flatten(&pooled_img, &flat_mtx);

    nn_stensor flat_t = {.tensor.mtx = &flat_mtx,
                         .tensor_type = NN_STENSOR_MTX};
    nn_stensor out_t = {.tensor.mtx = output, .tensor_type = NN_STENSOR_MTX};
    nn_forward(&g_resnet18_layers[21], &flat_t, &out_t);
    g_resnet18_io[0] = flat_t;
    g_resnet18_io[1] = out_t;
    return true;
}
