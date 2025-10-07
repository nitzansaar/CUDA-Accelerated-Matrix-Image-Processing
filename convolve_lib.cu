#include <cuda_runtime.h>
#include <stdio.h>

__constant__ float d_filter[49];

__global__ void convolution2D(const unsigned char *input,
                              unsigned char *output,
                              int M, int N)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M || c >= M) return;
    int radius = N / 2;
    float sum = 0.0f;

    for (int fr = -radius; fr <= radius; fr++) {
        for (int fc = -radius; fc <= radius; fc++) {
            int ir = r + fr;
            int ic = c + fc;
            if (ir >= 0 && ir < M && ic >= 0 && ic < M) {
                float pixel = (float)input[ir * M + ic];
                float weight = d_filter[(fr + radius) * N + (fc + radius)];
                sum += pixel * weight;
            }
        }
    }
    sum = fminf(fmaxf(sum, 0.0f), 255.0f);
    output[r * M + c] = (unsigned char)(sum + 0.5f);
}

extern "C" void cuda_convolve(unsigned char *h_input,
                              unsigned char *h_output,
                              float *h_filter,
                              int M, int N)
{
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, M*M);
    cudaMalloc(&d_output, M*M);
    cudaMemcpy(d_input, h_input, M*M, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_filter, h_filter, N*N*sizeof(float));

    dim3 block(16,16);
    dim3 grid((M+block.x-1)/block.x,(M+block.y-1)/block.y);

    convolution2D<<<grid,block>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, M*M, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}
