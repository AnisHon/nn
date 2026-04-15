#include "nn_conv2d.h"
#include "nn_fc.h"
#include "nn_function.h"
#include "nn_image.h"
#include "nn_matrix.h"
#include "nn_tests.h"
#include "nn_utils.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static int test_utils(void) {
    assert(nn_to_int((int8_t)10, 10) == 0);
    assert(nn_to_u8(0, 10) == (int8_t)10);
    assert(nn_to_u8(200, 0) == (int8_t)127);
    assert(nn_to_u8(-200, 0) == (int8_t)-128);
    assert(nn_q8_asym_to_int((int8_t)0, 0) == 0);
    assert(nn_q8_asym_to_int((int8_t)-1, 255) == 0);
    assert(nn_q8_sym_to_int((int8_t)-3, 0) == -3);
    assert(nn_int_to_q8_asym(0, 128) == (int8_t)-128);
    assert(nn_qalign_i8((int8_t)-126, 0.5f, 128, 0.25f, 128) == (int8_t)-124);
    assert(nn_qadd_i8((int8_t)-126, 0.5f, 128, (int8_t)-124, 0.25f, 128,
                      0.25f, 128) == (int8_t)-120);
    return 0;
}

static int test_matrix_basic(void) {
    static int8_t a_buf[6] = {1, 2, 3, 4, 5, 6};
    nn_matrix a;
    nn_mtx_init(&a, 2, 3, a_buf, 7);
    assert(*nn_mtx_get(&a, 0, 0) == 1);
    assert(*nn_mtx_get(&a, 1, 2) == 6);
    nn_mtx_set(&a, 1, 1, (int8_t)99);
    assert(*nn_mtx_get(&a, 1, 1) == 99);
    return 0;
}

static int test_relu_flatten_avgpool(void) {
    static int8_t img_buf[4] = {(int8_t)126, (int8_t)127, (int8_t)-128, (int8_t)-127};
    static nn_matrix img_mtx_arr[1];
    nn_mtx_init(&img_mtx_arr[0], 2, 2, img_buf, 128);
    nn_image img;
    nn_image_init(&img, 1, 2, 2, img_mtx_arr, 128);

    static int8_t relu_buf[4] = {0};
    static nn_matrix relu_mtx_arr[1];
    nn_mtx_init(&relu_mtx_arr[0], 2, 2, relu_buf, 128);
    nn_image relu_img;
    nn_image_init(&relu_img, 1, 2, 2, relu_mtx_arr, 128);
    nn_relu_image(&img, &relu_img);
    assert(relu_buf[0] == (int8_t)-128);
    assert(relu_buf[1] == (int8_t)-128);
    assert(relu_buf[2] == (int8_t)-128);
    assert(relu_buf[3] == (int8_t)-127);

    static int8_t flat_buf[4] = {0};
    nn_matrix flat;
    nn_mtx_init(&flat, 1, 4, flat_buf, 128);
    nn_flatten(&img, &flat);
    assert(flat_buf[0] == (int8_t)126 && flat_buf[3] == (int8_t)-127);

    static int8_t pool_buf[1] = {0};
    static nn_matrix pool_mtx_arr[1];
    nn_mtx_init(&pool_mtx_arr[0], 1, 1, pool_buf, 128);
    nn_image pool_img;
    nn_image_init(&pool_img, 1, 1, 1, pool_mtx_arr, 128);
    nn_avg_pool(&img, &pool_img);
    assert(pool_buf[0] == (int8_t)-128);
    return 0;
}

static int test_conv2d_quantized(void) {
    static int8_t in_buf[9] = {(int8_t)-128, (int8_t)-127, (int8_t)-126,
                               (int8_t)-125, (int8_t)-124, (int8_t)-123,
                               (int8_t)-122, (int8_t)-121, (int8_t)-120};
    static nn_matrix in_mtx_arr[1];
    nn_mtx_init(&in_mtx_arr[0], 3, 3, in_buf, 128);
    nn_image in_img;
    nn_image_init(&in_img, 1, 3, 3, in_mtx_arr, 128);

    static int8_t k_buf[4] = {1, 0, 0, 1};
    static nn_matrix k_mtx_arr[1];
    nn_mtx_init(&k_mtx_arr[0], 2, 2, k_buf, 0);

    static int8_t out_buf[4] = {0};
    static nn_matrix out_mtx_arr[1];
    nn_mtx_init(&out_mtx_arr[0], 2, 2, out_buf, 3);
    nn_image out_img;
    nn_image_init(&out_img, 1, 2, 2, out_mtx_arr, 3);

    nn_conv2d conv;
    static const int32_t bias[1] = {0};
    nn_conv2d_init(&conv, k_buf, k_mtx_arr, 1, 1, 2, 2, 1, 1, 0, 0, bias, 0, 3);
    nn_conv2d_forward(&conv, &in_img, &out_img);

    assert((uint8_t)out_buf[0] == 7);
    assert((uint8_t)out_buf[1] == 9);
    assert((uint8_t)out_buf[2] == 13);
    assert((uint8_t)out_buf[3] == 15);
    return 0;
}

static int test_fc_basic(void) {
    static int8_t in_buf[4] = {(int8_t)-128, (int8_t)-127, (int8_t)-126, (int8_t)-125};
    nn_matrix in;
    nn_mtx_init(&in, 1, 4, in_buf, 128);

    static int8_t w_buf[8] = {1, 1, 1, 1, 1, 0, 0, 0};
    nn_matrix w;
    nn_mtx_init(&w, 2, 4, w_buf, 0);

    static int8_t out_buf[2] = {0};
    nn_matrix out;
    nn_mtx_init(&out, 1, 2, out_buf, 3);

    static const int32_t bias[2] = {0, 0};
    nn_fc fc;
    nn_fc_init(&fc, w_buf, bias, 4, 2, &w, 0, 3);
    nn_fc_forward(&fc, &in, &out);

    assert((uint8_t)out_buf[0] == 9);
    assert((uint8_t)out_buf[1] == 3);
    return 0;
}

int nn_run_all_tests(void) {
    int rc = 0;
    rc |= test_utils();
    rc |= test_matrix_basic();
    rc |= test_relu_flatten_avgpool();
    rc |= test_conv2d_quantized();
    rc |= test_fc_basic();
    printf("All tests passed.\n");
    return rc;
}
