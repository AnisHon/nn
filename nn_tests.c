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
    assert(nn_to_int((int8_t)0, 10) == -10);
    assert(nn_to_u8(0, 10) == (int8_t)10);
    assert(nn_to_u8(200, 0) == (int8_t)127);
    assert(nn_to_u8(-200, 0) == (int8_t)-128);

    // align: (x_scale=0.5,zp=10) -> (y_scale=0.25,zp=0)
    // x_q=12 => real=(12-10)*0.5=1.0 => y_q=round(1.0/0.25)+0=4
    assert(nn_qalign_i8((int8_t)12, 0.5f, 10, 0.25f, 0) == (int8_t)4);

    // add with different quant params:
    // a_q=12 (a_scale=0.5,a_zp=10) => a_real=1.0
    // b_q=4  (b_scale=0.25,b_zp=0) => b_real=1.0
    // sum=2.0, out_scale=0.25,out_zp=0 => out_q=8
    assert(nn_qadd_i8((int8_t)12, 0.5f, 10, (int8_t)4, 0.25f, 0, 0.25f, 0) ==
           (int8_t)8);

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
    // image: C=1, H=2, W=2, zp=10
    static int8_t img_buf[4] = {8, 9, 10, 11}; // int domain: -2,-1,0,1
    static nn_matrix img_mtx_arr[1];
    nn_mtx_init(&img_mtx_arr[0], 2, 2, img_buf, 10);
    nn_image img;
    nn_image_init(&img, 1, 2, 2, img_mtx_arr, 10);

    // relu out
    static int8_t relu_buf[4] = {0};
    static nn_matrix relu_mtx_arr[1];
    nn_mtx_init(&relu_mtx_arr[0], 2, 2, relu_buf, 10);
    nn_image relu_img;
    nn_image_init(&relu_img, 1, 2, 2, relu_mtx_arr, 10);
    nn_relu_image(&img, &relu_img);
    // negatives should become zp (10)
    assert(relu_buf[0] == 10);
    assert(relu_buf[1] == 10);
    assert(relu_buf[2] == 10);
    assert(relu_buf[3] == 11);

    // flatten to 1x4
    static int8_t flat_buf[4] = {0};
    nn_matrix flat;
    nn_mtx_init(&flat, 1, 4, flat_buf, 10);
    nn_flatten(&img, &flat);
    assert(flat_buf[0] == 8 && flat_buf[1] == 9 && flat_buf[2] == 10 &&
           flat_buf[3] == 11);

    // avg pool to 1x1
    static int8_t pool_buf[1] = {0};
    static nn_matrix pool_mtx_arr[1];
    nn_mtx_init(&pool_mtx_arr[0], 1, 1, pool_buf, 10);
    nn_image pool_img;
    nn_image_init(&pool_img, 1, 1, 1, pool_mtx_arr, 10);
    nn_avg_pool(&img, &pool_img);
    // mean int domain = (-2-1+0+1)/4 = -0.5 -> integer division in code: -2/4 = 0? actually sum=-2 => -2/4 = 0 in C (towards 0)
    // so expected mean = 0 => quant = zp
    assert(pool_buf[0] == 10);
    return 0;
}

static int test_conv2d_zp_mismatch(void) {
    // 1x3x3 input, zp=10
    // values in quant: [10..18] => int domain [0..8]
    static int8_t in_buf[9] = {
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,
    };
    static nn_matrix in_mtx_arr[1];
    nn_mtx_init(&in_mtx_arr[0], 3, 3, in_buf, 10);
    nn_image in_img;
    nn_image_init(&in_img, 1, 3, 3, in_mtx_arr, 10);

    // 1 out channel, 1 in channel, 2x2 kernel, weight zp=0
    static int8_t k_buf[4] = {1, 0,
                              0, 1}; // int domain same
    static nn_matrix k_mtx_arr[1];
    nn_mtx_init(&k_mtx_arr[0], 2, 2, k_buf, 0);

    // output 2x2, out zp=3
    static int8_t out_buf[4] = {0};
    static nn_matrix out_mtx_arr[1];
    nn_mtx_init(&out_mtx_arr[0], 2, 2, out_buf, 3);
    nn_image out_img;
    nn_image_init(&out_img, 1, 2, 2, out_mtx_arr, 3);

    // conv2d init with mismatched conv2d->zp (should NOT matter now)
    nn_conv2d conv;
    static const int32_t bias[1] = {0};
    nn_conv2d_init(&conv, k_mtx_arr, 1, 1, 2, 2, 1, 1, 0, 0, bias, 0);

    nn_conv2d_forward(&conv, &in_img, &out_img);

    // expected acc = x00 + x11 in int domain (because identity corners)
    // top-left patch: [0,1;3,4] => 0+4=4 => q=4+out_zp=7
    // top-right patch: [1,2;4,5] => 1+5=6 => q=9
    // bottom-left patch: [3,4;6,7] => 3+7=10 => q=13
    // bottom-right patch: [4,5;7,8] => 4+8=12 => q=15
    assert(out_buf[0] == 7);
    assert(out_buf[1] == 9);
    assert(out_buf[2] == 13);
    assert(out_buf[3] == 15);
    return 0;
}

static int test_fc_basic(void) {
    // in: 1x4, zp=10, int domain [0,1,2,3]
    static int8_t in_buf[4] = {10, 11, 12, 13};
    nn_matrix in;
    nn_mtx_init(&in, 1, 4, in_buf, 10);

    // weight: 2x4, wzp=0
    static int8_t w_buf[8] = {
        1, 1, 1, 1, // sum
        1, 0, 0, 0  // pick x0
    };
    nn_matrix w;
    nn_mtx_init(&w, 2, 4, w_buf, 0);

    // bias as matrix (1x2) stored int8 here to match current fc impl
    static int8_t b_buf[2] = {0, 0};
    nn_matrix b;
    nn_mtx_init(&b, 1, 2, b_buf, 0);

    // out: 1x2, out zp=3
    static int8_t out_buf[2] = {0};
    nn_matrix out;
    nn_mtx_init(&out, 1, 2, out_buf, 3);

    nn_fc fc;
    nn_fc_init(&fc, 4, 2, &w, &b, 3);
    nn_fc_forward(&fc, &in, &out);

    // y0 int = 0+1+2+3 = 6 => q=9
    // y1 int = x0 = 0 => q=3
    assert(out_buf[0] == 9);
    assert(out_buf[1] == 3);
    return 0;
}

int nn_run_all_tests(void) {
    int rc = 0;
    rc |= test_utils();
    rc |= test_matrix_basic();
    rc |= test_relu_flatten_avgpool();
    rc |= test_conv2d_zp_mismatch();
    rc |= test_fc_basic();
    printf("All tests passed.\n");
    return rc;
}

