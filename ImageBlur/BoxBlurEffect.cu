#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

#include "lodepng.h" 

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line){
    if (code != cudaSuccess){
        fprintf(stderr, "CUDA Error %s %d: %s\n", file, line, cudaGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ int clampi(int v, int lo, int hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void boxBlurKernel(const unsigned char* __restrict__ in,
                              unsigned char* __restrict__ out,
                              int width, int height, int halfKernelSize){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int kernelSize = 2 * halfKernelSize + 1;
    const int startX = x - halfKernelSize;
    const int startY = y - halfKernelSize;

    int sumR = 0, sumG = 0, sumB = 0, sumA = 0;
    int count = 0;

    for (int j = 0; j < kernelSize; ++j){
        int yy = clampi(startY + j, 0, height - 1);
        for (int i = 0; i < kernelSize; ++i){
            int xx = clampi(startX + i, 0, width - 1);
            int idx = (yy * width + xx) * 4;
            sumR += in[idx + 0];
            sumG += in[idx + 1];
            sumB += in[idx + 2];
            sumA += in[idx + 3];
            ++count;
        }
    }

    int outIdx = (y * width + x) * 4;
    out[outIdx + 0] = static_cast<unsigned char>(sumR / count);
    out[outIdx + 1] = static_cast<unsigned char>(sumG / count);
    out[outIdx + 2] = static_cast<unsigned char>(sumB / count);
    out[outIdx + 3] = static_cast<unsigned char>(sumA / count);
}

/*
__global__ void boxBlurHorizontal(const unsigned char* __restrict__ in,
                                  unsigned char* __restrict__ tmp,
                                  int width, int height, int halfKernelSize){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int kernelSize = 2 * halfKernelSize + 1;
    const int startX = x - halfKernelSize;

    int sumR=0,sumG=0,sumB=0,sumA=0; int count=0;
    for (int i = 0; i < kernelSize; ++i){
        int xx = clampi(startX + i, 0, width - 1);
        int idx = (y * width + xx) * 4;
        sumR += in[idx+0]; sumG += in[idx+1]; sumB += in[idx+2]; sumA += in[idx+3];
        ++count;
    }
    int outIdx = (y * width + x) * 4;
    tmp[outIdx+0] = sumR / count; tmp[outIdx+1] = sumG / count; tmp[outIdx+2] = sumB / count; tmp[outIdx+3] = sumA / count;
}

__global__ void boxBlurVertical(const unsigned char* __restrict__ tmp,
                                unsigned char* __restrict__ out,
                                int width, int height, int halfKernelSize){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int kernelSize = 2 * halfKernelSize + 1;
    const int startY = y - halfKernelSize;

    int sumR=0,sumG=0,sumB=0,sumA=0; int count=0;
    for (int j = 0; j < kernelSize; ++j){
        int yy = clampi(startY + j, 0, height - 1);
        int idx = (yy * width + x) * 4;
        sumR += tmp[idx+0]; sumG += tmp[idx+1]; sumB += tmp[idx+2]; sumA += tmp[idx+3];
        ++count;
    }
    int outIdx = (y * width + x) * 4;
    out[outIdx+0] = sumR / count; out[outIdx+1] = sumG / count; out[outIdx+2] = sumB / count; out[outIdx+3] = sumA / count;
}
*/

int main(int argc, char** argv){
    if (argc < 4){
        fprintf(stderr, "Usage: %s <input.png> <output.png> <halfKernelSize>\n", argv[0]);
        return EXIT_FAILURE;
    }

    std::string inPath  = argv[1];
    std::string outPath = argv[2];
    int halfKernelSize = std::max(0, atoi(argv[3]));

    std::vector<unsigned char> image;
    unsigned width = 0, height = 0;
    unsigned err = lodepng::decode(image, width, height, inPath);
    if (err){
        fprintf(stderr, "LodePNG decode error %u: %s\n", err, lodepng_error_text(err));
        return EXIT_FAILURE;
    }
    if (image.size() != width * height * 4){
        fprintf(stderr, "Unexpected decoded size.\n");
        return EXIT_FAILURE;
    }

    const size_t numBytes = image.size();

    unsigned char *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, numBytes));
    CUDA_CHECK(cudaMalloc(&d_out, numBytes));
    CUDA_CHECK(cudaMemcpy(d_in, image.data(), numBytes, cudaMemcpyHostToDevice));

    dim3 block(16,16);
    dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);

    boxBlurKernel<<<grid, block>>>(d_in, d_out, (int)width, (int)height, halfKernelSize);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned char> outHost(numBytes);
    CUDA_CHECK(cudaMemcpy(outHost.data(), d_out, numBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    std::vector<unsigned char> png;
    err = lodepng::encode(png, outHost.data(), width, height, LCT_RGBA, 8);
    if (err){
        fprintf(stderr, "LodePNG encode error %u: %s\n", err, lodepng_error_text(err));
        return EXIT_FAILURE;
    }
    err = lodepng_save_file(png.data(), png.size(), outPath.c_str());
    if (err){
        fprintf(stderr, "LodePNG save error %u: %s\n", err, lodepng_error_text(err));
        return EXIT_FAILURE;
    }

    printf("wrote %s (%ux%u, halfKernelSize=%d => kernel=%dx%d)\n",
           outPath.c_str(), width, height, halfKernelSize,
           2*halfKernelSize+1, 2*halfKernelSize+1);
    return EXIT_SUCCESS;
}
