#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void helloCUDA() {
    printf("Hello CUDA from GPU!\n");
}

int main() {
    printf("Hello CUDA from CPU!\n");

    helloCUDA<<<2, 5>>>();

    cudaError_t e = cudaGetLastError();
    e = cudaDeviceSynchronize();

    return 0;
}
