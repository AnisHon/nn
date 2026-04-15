#include "nn_image.h"
#include "nn_matrix.h"
#include "stdbool.h"
#include <assert.h>
#include <stdint.h>

static bool check_image(nn_image *image) {
    bool flag = true;
    for (size_t i = 0; i < image->channels; i++) {
        nn_matrix *mtx = nn_image_get(image, i);
        bool is_invalid =
            mtx->h != image->h || mtx->w != image->w || mtx->zp != image->zp;
        if (is_invalid) {
            flag = false;
            break;
        }
    }
    return flag;
}

void nn_image_init(nn_image *image, const size_t channels, const size_t h,
                   const size_t w, nn_matrix *mtx, int32_t zp) {
    image->h = h;
    image->w = w;
    image->channels = channels;
    image->zp = zp;
    image->mtx = mtx;
    assert(check_image(image));
}

nn_matrix *nn_image_get(const nn_image *image, const size_t channel) {
    assert(channel < image->channels);
    return &image->mtx[channel];
}
