#include "resnet18.h"
#include "nn_image.h"
#include "nn_matrix.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>
#include <setjmp.h>

#define INPUT_H 64
#define INPUT_W 64
#define INPUT_C 3
#define OUTPUT_CLASSES 200
#define INPUT_ZP 128
#define OUTPUT_ZP 48

struct error_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

static char wnids[OUTPUT_CLASSES][32];
static int wnid_count = 0;
static int8_t image_buf[INPUT_C * INPUT_H * INPUT_W];
static nn_matrix input_mats[INPUT_C];
static nn_image input_img;
static int8_t output_buf[OUTPUT_CLASSES];
static nn_matrix output_mat;

static void jpeg_error_exit(j_common_ptr cinfo) {
    struct error_mgr *err = (struct error_mgr *)cinfo->err;
    (*cinfo->err->output_message)(cinfo);
    longjmp(err->setjmp_buffer, 1);
}

static int load_wnids(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char line[128];
    wnid_count = 0;
    while (wnid_count < OUTPUT_CLASSES && fgets(line, sizeof(line), fp)) {
        size_t n = strcspn(line, "\r\n");
        line[n] = '\0';
        if (!line[0]) continue;
        if (snprintf(wnids[wnid_count], sizeof(wnids[wnid_count]), "%s", line) >=
            (int)sizeof(wnids[wnid_count])) {
            fclose(fp);
            return -1;
        }
        wnid_count++;
    }
    fclose(fp);
    return wnid_count == OUTPUT_CLASSES ? 0 : -1;
}

static int wnid_to_label(const char *wnid) {
    for (int i = 0; i < wnid_count; ++i) {
        if (strcmp(wnids[i], wnid) == 0) return i;
    }
    return -1;
}

static int load_jpeg_chw_q8(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return -1;

    struct jpeg_decompress_struct cinfo;
    struct error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(file);
        return -1;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    if (cinfo.output_width != INPUT_W || cinfo.output_height != INPUT_H ||
        cinfo.output_components != 3) {
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(file);
        return -1;
    }

    const size_t row_stride = INPUT_W * 3;
    JSAMPLE *row = (JSAMPLE *)malloc(row_stride);
    if (!row) {
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(file);
        return -1;
    }

    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW rowptr = row;
        jpeg_read_scanlines(&cinfo, &rowptr, 1);
        const size_t y = cinfo.output_scanline - 1;
        for (size_t x = 0; x < INPUT_W; ++x) {
            const size_t src = x * 3;
            image_buf[0 * INPUT_H * INPUT_W + y * INPUT_W + x] =
                (int8_t)(uint8_t)row[src + 0];
            image_buf[1 * INPUT_H * INPUT_W + y * INPUT_W + x] =
                (int8_t)(uint8_t)row[src + 1];
            image_buf[2 * INPUT_H * INPUT_W + y * INPUT_W + x] =
                (int8_t)(uint8_t)row[src + 2];
        }
    }

    free(row);
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(file);
    return 0;
}

static int argmax_u8(const int8_t *data, size_t n) {
    int best = 0;
    uint8_t bestv = (uint8_t)data[0];
    for (size_t i = 1; i < n; ++i) {
        uint8_t v = (uint8_t)data[i];
        if (v > bestv) {
            bestv = v;
            best = (int)i;
        }
    }
    return best;
}

static int infer_one(const char *jpeg_path) {
    if (load_jpeg_chw_q8(jpeg_path) != 0) return -1;
    for (int c = 0; c < INPUT_C; ++c) {
        nn_mtx_init(&input_mats[c], INPUT_H, INPUT_W,
                    image_buf + c * INPUT_H * INPUT_W, INPUT_ZP);
    }
    nn_image_init(&input_img, INPUT_C, INPUT_H, INPUT_W, input_mats, INPUT_ZP);
    nn_mtx_init(&output_mat, 1, OUTPUT_CLASSES, output_buf, OUTPUT_ZP);
    return resnet18_infer(&input_img, &output_mat) ? argmax_u8(output_buf, OUTPUT_CLASSES)
                                                   : -1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr,
                "Usage: %s <val_annotations.txt> <val_images_dir> <wnids.txt>\n",
                argv[0]);
        return 1;
    }

    if (load_wnids(argv[3]) != 0) {
        fprintf(stderr, "failed to load wnids: %s\n", argv[3]);
        return 1;
    }

    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        perror("fopen");
        return 1;
    }

    printf("starting Tiny-ImageNet val...\n");
    fflush(stdout);

    char line[1024];
    size_t total = 0, correct = 0, failed = 0;
    while (fgets(line, sizeof(line), fp)) {
        char file_name[256], wnid[32];
        int x0, y0, x1, y1;
        const size_t seen = total + failed + 1;
        printf("\rprocessing sample %zu...", seen);
        fflush(stdout);
        if (sscanf(line, "%255s\t%31s\t%d\t%d\t%d\t%d", file_name, wnid, &x0,
                   &y0, &x1, &y1) < 2) {
            continue;
        }
        int label = wnid_to_label(wnid);
        if (label < 0) {
            failed++;
            continue;
        }
        char image_path[1024];
        if (snprintf(image_path, sizeof(image_path), "%s/%s", argv[2], file_name) >=
            (int)sizeof(image_path)) {
            failed++;
            continue;
        }
        int pred = infer_one(image_path);
        if (pred < 0) {
            failed++;
            printf("\rprocessed=%zu failed=%zu", total, failed);
            fflush(stdout);
            continue;
        }
        total++;
        if (pred == label) correct++;
        printf("\rprocessed=%zu acc=%.2f%% failed=%zu", total,
               100.0 * (double)correct / (double)total, failed);
        fflush(stdout);
    }
    fclose(fp);

    if (!total) {
        fprintf(stderr, "no samples processed\n");
        return 1;
    }

    printf("\nTiny-ImageNet val accuracy: %zu / %zu = %.2f%%\n", correct, total,
           100.0 * (double)correct / (double)total);
    fflush(stdout);
    if (failed) printf("failed samples: %zu\n", failed);
    return 0;
}
