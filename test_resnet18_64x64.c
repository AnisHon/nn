#include "resnet18.h"
#include "nn_image.h"
#include "nn_matrix.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define INPUT_H 64
#define INPUT_W 64
#define INPUT_C 3
#define OUTPUT_CLASSES 200
#define INPUT_ZP 128
#define OUTPUT_ZP 48

int main(void) {
    printf("Testing ResNet18 input validation...\n");

    {
        static int8_t invalid_input_buf[3 * 224 * 224];
        static nn_matrix invalid_input_mats[3];
        for (int c = 0; c < 3; c++) {
            nn_mtx_init(&invalid_input_mats[c], 224, 224,
                        invalid_input_buf + c * 224 * 224, INPUT_ZP);
        }
        nn_image invalid_input;
        nn_image_init(&invalid_input, 3, 224, 224, invalid_input_mats, INPUT_ZP);
        static int8_t output_buf[OUTPUT_CLASSES];
        static nn_matrix output_mat;
        nn_mtx_init(&output_mat, 1, OUTPUT_CLASSES, output_buf, OUTPUT_ZP);
        if (resnet18_infer(&invalid_input, &output_mat)) {
            fprintf(stderr, "FAILED: should reject 224x224 input\n");
            return 1;
        }
        printf("ok: rejected invalid input size\n");
    }

    {
        static int8_t input_buf[INPUT_C * INPUT_H * INPUT_W];
        static nn_matrix input_mats[INPUT_C];
        memset(input_buf, (int8_t)-128, sizeof(input_buf));
        for (int c = 0; c < INPUT_C; c++) {
            nn_mtx_init(&input_mats[c], INPUT_H, INPUT_W,
                        input_buf + c * INPUT_H * INPUT_W, INPUT_ZP);
        }
        nn_image input;
        nn_image_init(&input, INPUT_C, INPUT_H, INPUT_W, input_mats, INPUT_ZP);
        static int8_t output_buf[OUTPUT_CLASSES];
        static nn_matrix output_mat;
        nn_mtx_init(&output_mat, 1, OUTPUT_CLASSES, output_buf, OUTPUT_ZP);
        if (!resnet18_infer(&input, &output_mat)) {
            fprintf(stderr, "FAILED: valid inference failed\n");
            return 1;
        }
        printf("ok: valid 64x64 inference passed\n");
        printf("first 10 logits: ");
        for (int i = 0; i < 10; i++) {
            printf("%u ", (unsigned)(uint8_t)output_buf[i]);
        }
        printf("\n");
    }

    {
        static int8_t input_buf[INPUT_C * INPUT_H * INPUT_W];
        static nn_matrix input_mats[INPUT_C];
        memset(input_buf, (int8_t)-128, sizeof(input_buf));
        for (int c = 0; c < INPUT_C; c++) {
            nn_mtx_init(&input_mats[c], INPUT_H, INPUT_W,
                        input_buf + c * INPUT_H * INPUT_W, INPUT_ZP);
        }
        nn_image input;
        nn_image_init(&input, INPUT_C, INPUT_H, INPUT_W, input_mats, INPUT_ZP);
        static int8_t output_buf[OUTPUT_CLASSES];
        static nn_matrix output_mat;
        nn_mtx_init(&output_mat, 1, OUTPUT_CLASSES, output_buf, 0);
        if (resnet18_infer(&input, &output_mat)) {
            fprintf(stderr, "FAILED: should reject wrong output zp\n");
            return 1;
        }
        printf("ok: rejected wrong output zero-point\n");
    }

    return 0;
}
