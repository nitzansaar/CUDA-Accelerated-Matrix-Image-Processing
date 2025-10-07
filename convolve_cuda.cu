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
// CUDA kernel
// --------------------------------------------------
__global__ void convolution2D(const unsigned char *input,
                              unsigned char *output,
                              int M, int N)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = N / 2;

    if (r >= M || c >= M) return;

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

    // clamp
    if (sum < 0)   sum = 0;
    if (sum > 255) sum = 255;
    output[r * M + c] = (unsigned char)(sum + 0.5f);
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

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start
    cudaEventRecord(start);

    // Launch kernel
    convolution2D<<<grid, block>>>(d_input, d_output, M, N);

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
