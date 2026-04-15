#ifndef NN_STENSOR_H
#define NN_STENSOR_H

#include "nn_image.h"
#include "nn_matrix.h"

enum nn_stensor_type { NN_STENSOR_MTX, NN_STENSOR_IMG };

typedef struct {
    union {
        nn_matrix *mtx;
        nn_image *img;
    } tensor;
    enum nn_stensor_type tensor_type;
} nn_stensor;

#endif //! NN_STENSOR_H
