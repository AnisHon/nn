#ifndef ACTIVATE_FUNC_H
#define ACTIVATE_FUNC_H

#include "nn_image.h"
#include "nn_matrix.h"
#include "nn_stensor.h"

void nn_relu_image(nn_image *image, nn_image *out);

void nn_relu_matrix(nn_matrix *vector, nn_matrix *out);

void nn_relu(const nn_stensor *ti, nn_stensor *to);

void nn_flatten(nn_image *image, nn_matrix *out);

void nn_avg_pool(nn_image *image, nn_image *out);

#endif
