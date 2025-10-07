#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

//
// General-purpose grayscale convolution
// Supports arbitrary N×N blur or edge-detection filters.
//

void convolutionCPU(const unsigned int *image,
                    const float *filter,
                    unsigned int *output,
                    int M, int N)
{
    int radius = N / 2;
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < M; c++) {
            float sum = 0.0f;
            for (int fr = -radius; fr <= radius; fr++) {
                for (int fc = -radius; fc <= radius; fc++) {
                    int ir = r + fr;
                    int ic = c + fc;
                    if (ir >= 0 && ir < M && ic >= 0 && ic < M) {
                        unsigned int pixel = image[ir * M + ic];
                        float weight = filter[(fr + radius) * N + (fc + radius)];
                        sum += pixel * weight;
                    }
                }
            }
            // clamp
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            output[r * M + c] = (unsigned int)(sum + 0.5f);
        }
    }
}

// -----------------------------
// Simple PGM (P5) reader/writer
// -----------------------------
unsigned int *readPGM(const char *filename, int M)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) { perror("Cannot open image file"); exit(1); }

    char magic[3];
    int width, height, maxval;
    fscanf(fp, "%2s", magic);
    if (strcmp(magic, "P5") != 0) { fprintf(stderr, "Not a P5 PGM\n"); exit(1); }

    // skip comments
    int ch;
    while ((ch = fgetc(fp)) == '#') while (fgetc(fp) != '\n');
    ungetc(ch, fp);
    fscanf(fp, "%d %d %d", &width, &height, &maxval);
    fgetc(fp); // consume whitespace

    if (width != M || height != M)
        fprintf(stderr, "Warning: image size mismatch (%dx%d vs %dx%d)\n",
                width, height, M, M);

    unsigned int *img = malloc(M * M * sizeof(unsigned int));
    for (int i = 0; i < M * M; i++) {
        unsigned char px; fread(&px, 1, 1, fp);
        img[i] = px;
    }
    fclose(fp);
    return img;
}

void writePGM(const char *filename, const unsigned int *img, int M)
{
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P5\n%d %d\n255\n", M, M);
    for (int i = 0; i < M * M; i++) {
        unsigned char px = (unsigned char)img[i];
        fwrite(&px, 1, 1, fp);
    }
    fclose(fp);
}

// -----------------------------
// Filter generators
// -----------------------------
void makeFilter(float *filter, int N, const char *mode)
{
    int total = N * N;

    if (strcmp(mode, "blur") == 0) {
        for (int i = 0; i < total; i++) filter[i] = 1.0f / total;
    }
    else if (strcmp(mode, "edge") == 0) {
        for (int i = 0; i < total; i++) filter[i] = -1.0f;
        filter[(N / 2) * N + (N / 2)] = (float)(total - 1); // center positive
    }
    else {
        fprintf(stderr, "Unknown mode '%s'. Use 'blur' or 'edge'.\n", mode);
        exit(1);
    }
}

// -----------------------------
// Main
// -----------------------------
int main(int argc, char **argv)
{
    if (argc < 5) {
        printf("Usage: %s <input.pgm> <M> <N> <mode>\n", argv[0]);
        printf("Example: %s lena_512.pgm 512 3 edge\n", argv[0]);
        printf("         %s lena_512.pgm 512 5 blur\n", argv[0]);
        return 1;
    }

    const char *fname = argv[1];
    int M = atoi(argv[2]);
    int N = atoi(argv[3]);
    const char *mode = argv[4];

    unsigned int *image  = readPGM(fname, M);
    unsigned int *output = malloc(M * M * sizeof(unsigned int));
    float *filter        = malloc(N * N * sizeof(float));

    makeFilter(filter, N, mode);

    clock_t start = clock();
    convolutionCPU(image, filter, output, M, N);
    clock_t end = clock();

    double t = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU convolution: %s, %dx%d image, %dx%d filter → %.4fs\n",
           mode, M, M, N, N, t);

    char outname[256];
    snprintf(outname, sizeof(outname), "output_%s_%dx%d.pgm", mode, N, N);
    writePGM(outname, output, M);
    printf("Output written to %s\n", outname);

    free(image); free(output); free(filter);
    return 0;
}
