#include <stdio.h>
#include "nn_tests.h"

int main(void) {
    (void)printf("Running nn tests...\n");
    return nn_run_all_tests();
}
