#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// GPU kernel
__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CPU version
// void matrixMultiplyCPU(float *A, float *B, float *C, int N) {
//     for (int row = 0; row < N; row++) {
//         for (int col = 0; col < N; col++) {
//             float sum = 0.0f;
//             for (int k = 0; k < N; k++) {
//                 sum += A[row * N + k] * B[k * N + col];
//             }
//             C[row * N + col] = sum;
//         }
//     }
// }

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    size_t size = N * N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
    }

    // // CPU timing
    // clock_t start = clock();
    // matrixMultiplyCPU(h_A, h_B, h_C, N);
    // clock_t end = clock();
    // printf("CPU time: %f s\n", (double)(end - start) / CLOCKS_PER_SEC);

    // GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t gstart, gend;
    cudaEventCreate(&gstart);
    cudaEventCreate(&gend);

    cudaEventRecord(gstart);
    matrixMultiplyGPU<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(gend);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, gstart, gend);
    printf("GPU time: %f ms\n", gpuTime);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}