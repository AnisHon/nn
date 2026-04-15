#ifndef NN_LAYER_H
#define NN_LAYER_H
#include "nn_conv2d.h"
#include "nn_fc.h"
#include "nn_image.h"
#include "nn_matrix.h"
#include "nn_stensor.h"
enum nn_layer_type {
    NN_LAYER_FC,
    NN_LAYER_CONV,
    NN_LAYER_RELU,
    NN_LAYER_AVG_POOL,
    NN_LAYER_FLATTEN,
};

typedef struct {
    union {
        nn_fc *fc;
        nn_conv2d *conv2d;
    } layer;
    enum nn_layer_type layer_type;
} nn_layer;

void nn_forward(const nn_layer *layer, const nn_stensor *ti, nn_stensor *to);

#endif
