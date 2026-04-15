#ifndef NN_IMAGE_H
#define NN_IMAGE_H

#include "nn_matrix.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t h, w;
    size_t channels;
    int32_t zp;
    nn_matrix *mtx;
} nn_image;

void nn_image_init(nn_image *image, const size_t channels, const size_t h,
                   const size_t w, nn_matrix *mtx, int32_t zp);

nn_matrix *nn_image_get(const nn_image *image, const size_t channel);

#endif
