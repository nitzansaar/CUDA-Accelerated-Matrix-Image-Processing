#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

// --------------------------------------------------
// Constant-memory filter (up to 49 elements, i.e., 7×7)
// --------------------------------------------------
__constant__ float d_filter[49];

// --------------------------------------------------
// CUDA kernel with shared memory and memory coalescing
// --------------------------------------------------
__global__ void convolution2D(const unsigned char *input,
                              unsigned char *output,
                              int M, int N)
{
    // Shared memory tile with halo region for filter
    extern __shared__ unsigned char s_tile[];
    
    int radius = N / 2;
    int tile_width = blockDim.x + 2 * radius;
    int tile_height = blockDim.y + 2 * radius;
    
    // Global thread indices
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Local thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Load tile into shared memory with coalesced access
    // Threads load consecutive elements row-by-row for optimal coalescing
    int threads_per_block = blockDim.x * blockDim.y;
    int thread_id = ty * blockDim.x + tx;
    int total_elements = tile_width * tile_height;
    int elements_per_thread = (total_elements + threads_per_block - 1) / threads_per_block;
    
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = thread_id + i * threads_per_block;
        if (idx < total_elements) {
            // Calculate position in tile (row-major order)
            int local_row = idx / tile_width;
            int local_col = idx % tile_width;
            
            // Calculate global coordinates (accounting for halo)
            int global_row = blockIdx.y * blockDim.y + local_row - radius;
            int global_col = blockIdx.x * blockDim.x + local_col - radius;
            
            // Load with boundary checking - consecutive threads access consecutive memory
            if (global_row >= 0 && global_row < M && global_col >= 0 && global_col < M) {
                s_tile[local_row * tile_width + local_col] = input[global_row * M + global_col];
            } else {
                s_tile[local_row * tile_width + local_col] = 0;
            }
        }
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // Check if this thread computes a valid output pixel
    if (gx >= M || gy >= M) return;
    
    // Compute convolution using shared memory
    float sum = 0.0f;
    
    // Local position in shared memory tile (accounting for halo)
    int local_r = ty + radius;
    int local_c = tx + radius;
    
    for (int fr = -radius; fr <= radius; fr++) {
        for (int fc = -radius; fc <= radius; fc++) {
            int local_ir = local_r + fr;
            int local_ic = local_c + fc;
            
            // Access shared memory (no bounds check needed as halo is loaded)
            float pixel = (float)s_tile[local_ir * tile_width + local_ic];
            float weight = d_filter[(fr + radius) * N + (fc + radius)];
            sum += pixel * weight;
        }
    }
    
    // Clamp and write output with coalesced access
    if (sum < 0)   sum = 0;
    if (sum > 255) sum = 255;
    output[gy * M + gx] = (unsigned char)(sum + 0.5f);
}

// --------------------------------------------------
// Host-side PGM utilities (same as CPU version)
// --------------------------------------------------
unsigned char *readPGM(const char *filename, int M) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) { perror("Cannot open image"); exit(1); }

    char magic[3]; int w,h,maxval;
    fscanf(fp, "%2s", magic);
    if (strcmp(magic,"P5")!=0){ fprintf(stderr,"Not a P5 PGM\n"); exit(1);}
    int ch; while ((ch=fgetc(fp))=='#') while (fgetc(fp)!='\n'); ungetc(ch,fp);
    fscanf(fp,"%d %d %d",&w,&h,&maxval); fgetc(fp);

    if (w!=M||h!=M) fprintf(stderr,"Warning: image size mismatch (%dx%d)\n",w,h);
    unsigned char *img=(unsigned char*)malloc(M*M);
    fread(img,1,M*M,fp); fclose(fp);
    return img;
}

void writePGM(const char *fname, const unsigned char *img, int M){
    FILE *fp=fopen(fname,"wb");
    fprintf(fp,"P5\n%d %d\n255\n",M,M);
    fwrite(img,1,M*M,fp);
    fclose(fp);
}

// --------------------------------------------------
// Filter generation (same as CPU version)
// --------------------------------------------------
void makeFilter(float *filter, int N, const char *mode){
    int total=N*N;
    if(strcmp(mode,"blur")==0){
        for(int i=0;i<total;i++) filter[i]=1.0f/total;
    }else if(strcmp(mode,"edge")==0){
        for(int i=0;i<total;i++) filter[i]=-1.0f;
        filter[(N/2)*N+(N/2)]=(float)(total-1);
    }else{
        fprintf(stderr,"Unknown mode '%s'\n",mode); exit(1);
    }
}

// --------------------------------------------------
// Main
// --------------------------------------------------
int main(int argc,char**argv){
    if(argc<5){
        printf("Usage: %s <input.pgm> <M> <N> <mode>\n",argv[0]);
        return 1;
    }

    const char* fname=argv[1];
    int M=atoi(argv[2]);
    int N=atoi(argv[3]);
    const char* mode=argv[4];

    // Host data
    unsigned char *h_input=readPGM(fname,M);
    unsigned char *h_output=(unsigned char*)malloc(M*M);
    float h_filter[49];   // up to 7×7
    makeFilter(h_filter,N,mode);

    // Device data
    unsigned char *d_input,*d_output;
    cudaMalloc(&d_input,M*M);
    cudaMalloc(&d_output,M*M);

    cudaMemcpy(d_input,h_input,M*M,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_filter,h_filter,N*N*sizeof(float));

    dim3 block(16,16);
    dim3 grid((M+block.x-1)/block.x,(M+block.y-1)/block.y);
    
    // Calculate shared memory size (tile + halo region)
    int radius = N / 2;
    int tile_width = block.x + 2 * radius;
    int tile_height = block.y + 2 * radius;
    size_t shared_mem_size = tile_width * tile_height * sizeof(unsigned char);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start
    cudaEventRecord(start);

    // Launch kernel with shared memory
    convolution2D<<<grid, block, shared_mem_size>>>(d_input, d_output, M, N);

    // Record stop and synchronize
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Compute elapsed time (milliseconds)
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU convolution: %s, %dx%d image, %dx%d filter → %.6fs\n",
        mode, M, M, N, N, milliseconds / 1000.0f);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaMemcpy(h_output,d_output,M*M,cudaMemcpyDeviceToHost);

    char outname[256];
    snprintf(outname,sizeof(outname),"output_%s_%dx%d.pgm",mode,N,N);
    writePGM(outname,h_output,M);
    printf("Output written to %s\n",outname);

    cudaFree(d_input); cudaFree(d_output);
    free(h_input); free(h_output);
    return 0;
}
