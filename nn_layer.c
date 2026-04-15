#include "nn_layer.h"
#include "nn_conv2d.h"
#include "nn_fc.h"
#include "nn_function.h"
#include <assert.h>

void nn_forward(const nn_layer *layer, const nn_stensor *ti, nn_stensor *to) {
    switch (layer->layer_type) {
    case NN_LAYER_CONV:
        assert(ti->tensor_type == NN_STENSOR_IMG &&
               to->tensor_type == NN_STENSOR_IMG);
        nn_conv2d_forward(layer->layer.conv2d, ti->tensor.img, to->tensor.img);
        break;
    case NN_LAYER_FC:
        assert(ti->tensor_type == NN_STENSOR_MTX &&
               to->tensor_type == NN_STENSOR_MTX);
        nn_fc_forward(layer->layer.fc, ti->tensor.mtx, to->tensor.mtx);
        break;
    case NN_LAYER_RELU:
        nn_relu(ti, to);
        break;
    case NN_LAYER_AVG_POOL:
        assert(ti->tensor_type == NN_STENSOR_IMG &&
               to->tensor_type == NN_STENSOR_IMG);
        nn_avg_pool(ti->tensor.img, to->tensor.img);
        break;
    case NN_LAYER_FLATTEN:
        assert(ti->tensor_type == NN_STENSOR_IMG &&
               to->tensor_type == NN_STENSOR_MTX);
        nn_flatten(ti->tensor.img, to->tensor.mtx);
        break;
    default:
        // just in case
        assert(false);
    }
}
