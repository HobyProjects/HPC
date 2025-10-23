#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#include "lodepng.h"

#define PI 3.1415926535897932384626433832795

inline void gpu_assert(cudaError_t code, const char* file, int line, bool abort=true)
{
  if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
  }
}
        
#define CUDA_CHECK(err) { gpu_assert((err), __FILE__, __LINE__); }

__device__ float coefficient(int x, int y, float sigma)
{
    float coeff = 1.0f / (2.0f * PI * sigma * sigma);
    float exponent = -(x * x + y * y) / (2.0f * sigma * sigma);
    return coeff * expf(exponent);
}

__global__ void box_blur_effect(unsigned char* input, unsigned char* output, int width, int height, int maxRadius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float cx = width / 2.0f;
    float cy = height / 2.0f;
    float dx = x - cx;
    float dy = y - cy;
    float dist = sqrtf(dx * dx + dy * dy);
    float maxDist = sqrtf(cx * cx + cy * cy);

    int radius = (int)(maxRadius * (dist / maxDist));
    if (radius < 1) {
        int idx = 4 * (y * width + x);
        output[idx + 0] = input[idx + 0];
        output[idx + 1] = input[idx + 1];
        output[idx + 2] = input[idx + 2];
        output[idx + 3] = input[idx + 3];
        return;
    }

    float sigma = radius / 2.0f;
    float4 sum = make_float4(0, 0, 0, 0);
    float totalWeight = 0.0f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            int nIdx = 4 * (ny * width + nx);

            float w = coefficient(kx, ky, sigma);
            sum.x += input[nIdx + 0] * w;
            sum.y += input[nIdx + 1] * w;
            sum.z += input[nIdx + 2] * w;
            sum.w += input[nIdx + 3] * w;
            totalWeight += w;
        }
    }

    int outIdx = 4 * (y * width + x);
    output[outIdx + 0] = (unsigned char)(sum.x / totalWeight);
    output[outIdx + 1] = (unsigned char)(sum.y / totalWeight);
    output[outIdx + 2] = (unsigned char)(sum.z / totalWeight);
    output[outIdx + 3] = (unsigned char)(sum.w / totalWeight);
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        printf("Usage: %s <input.png> <output.png> [maxRadius]\n", argv[0]);
        return 1;
    }

    const char* inputFilename = argv[1];
    const char* outputFilename = argv[2];
    int maxRadius = (argc > 3) ? atoi(argv[3]) : 30;

    unsigned char* image = nullptr;
    unsigned width, height;
    unsigned error = lodepng_decode32_file(&image, &width, &height, inputFilename);
    if (error) {
        printf("Decoder error %u: %s\n", error, lodepng_error_text(error));
        return 1;
    }

    size_t imageSize = width * height * 4;
    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    CUDA_CHECK(cudaMemcpy(d_input, image, imageSize, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    printf("Applying box blur (max radius = %d)...\n", maxRadius);
    box_blur_effect<<<grid, block>>>(d_input, d_output, width, height, maxRadius);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned char* output = (unsigned char*)malloc(imageSize);
    CUDA_CHECK(cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost));

    unsigned encodeError = lodepng_encode32_file(outputFilename, output, width, height);
    if (encodeError)
        printf("Encoder error %u: %s\n", encodeError, lodepng_error_text(encodeError));
    else
        printf("Saved blurred image to %s\n", outputFilename);

    cudaFree(d_input);
    cudaFree(d_output);
    free(image);
    free(output);

    return 0;
}
