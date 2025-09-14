#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#define NUM_DATA 1024

__global__ void vecAdd(int* _a, int* _b, int* _c) {
    int tID = threadIdx.x;
    _c[tID] = _a[tID] + _b[tID];
}

void vecAdd_cpu(int* _a, int* _b, int* _c) {
    for (int i = 0; i < NUM_DATA; i++) {
        _c[i] = _a[i] + _b[i];
    }
}

int main(void) {
    int* a, * b, * c, * hc; // Vectors on the host
    int* da, * db, * dc; // Vectors on the device

    int memSize = sizeof(int) * NUM_DATA;
    printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

    // Memory allocation on the host side
    a = new int[NUM_DATA]; memset(a, 0, memSize);
    b = new int[NUM_DATA]; memset(b, 0, memSize);
    c = new int[NUM_DATA]; memset(c, 0, memSize);
    hc = new int[NUM_DATA]; memset(hc, 0, memSize);

    for (int i = 0; i < NUM_DATA; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    vecAdd_cpu(a, b, hc);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = stop_cpu - start_cpu;
    printf("CPU execution time: %.6f ms\n", cpu_time.count());

    // Memory allocation on the device side
    cudaMalloc(&da, memSize); cudaMemset(da, 0 ,memSize);
    cudaMalloc(&db, memSize); cudaMemset(db, 0, memSize);
    cudaMalloc(&dc, memSize); cudaMemset(dc, 0, memSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Copy data from host to device
    cudaMemcpy(da, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, memSize, cudaMemcpyHostToDevice);

    // Kernel invocation
    vecAdd<<<1, NUM_DATA>>>(da, db, dc);

    // Copy data from device to host
    cudaMemcpy(c, dc, memSize, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Total GPU processing time (with memory transfers): %.6f ms\n", gpu_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Release device memory
    cudaFree(da); cudaFree(db); cudaFree(dc);

    // Check the result
    bool result = true;
    for (int i = 0; i < NUM_DATA; i++) {
        if (hc[i] != c[i]) {
            result = false;
            printf("[%d] The result is not matched!: host = %d, device = %d\n", i, hc[i], c[i]);
        }
    }

    if (result) {
        printf("GPU calculation is correct!\n");
    }

    printf("\n==== Performance Comparison ====\n");
    printf("CPU time: %.6f ms\n", cpu_time.count());
    printf("GPU time (with mem transfers): %.6f ms\n", gpu_time);
    if (cpu_time.count() > gpu_time)
        printf("=> GPU is faster by %.6f ms\n", cpu_time.count() - gpu_time);
    else
        printf("=> CPU is faster by %.6f ms\n", gpu_time - cpu_time.count());
    
    /* 
    CPU: AMD Ryzen 7 7800X3D
    GPU: GeForce RTX 4070Ti Super
    
    Sample Output:
    1024 elements, memSize = 4096 bytes
    CPU execution time: 0.001779 ms
    Total GPU processing time (with memory transfers): 1.145088 ms
    GPU calculation is correct!

    ==== Performance Comparison ====
    CPU time: 0.001779 ms
    GPU time (with mem transfers): 1.145088 ms
    => CPU is faster by 1.143309 ms
    */
    

    // Release host memory
    delete[] a; delete[] b; delete[] c; delete[] hc;

    return 0;
}
