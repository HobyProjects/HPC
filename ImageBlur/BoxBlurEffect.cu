#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include "lodepng.h"

#define CUDA_CHECK(expr) do {                             \
  cudaError_t _err = (expr);                              \
  if (_err != cudaSuccess) {                              \
    std::cerr << "CUDA error: " << cudaGetErrorString(_err)\
              << " at " << __FILE__ << ":" << __LINE__    \
              << std::endl;                               \
    std::exit(1);                                         \
  }                                                       \
} while (0)

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Naive box blur: average over (2*radius+1)^2 window.
// Input/Output are RGBA8 interleaved.
__global__ void boxBlurRGBA(
    const uint8_t* __restrict__ in,
    uint8_t* __restrict__ out,
    int width, int height, int radius)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int channels = 4;
  int2 wh = make_int2(width, height);

  int rsum = 0, gsum = 0, bsum = 0, asum = 0;
  int count = 0;

  int y0 = max(0, y - radius);
  int y1 = min(height - 1, y + radius);
  int x0 = max(0, x - radius);
  int x1 = min(width  - 1, x + radius);

  for (int yy = y0; yy <= y1; ++yy) {
    int row = yy * wh.x;
    for (int xx = x0; xx <= x1; ++xx) {
      const uint8_t* p = &in[(row + xx) * channels];
      rsum += p[0];
      gsum += p[1];
      bsum += p[2];
      asum += p[3];
      ++count;
    }
  }

  uint8_t* q = &out[(y * width + x) * channels];
  q[0] = static_cast<uint8_t>(rsum / count);
  q[1] = static_cast<uint8_t>(gsum / count);
  q[2] = static_cast<uint8_t>(bsum / count);
  q[3] = static_cast<uint8_t>(asum / count);
}

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " input.png output.png [radius]\n";
    return 1;
  }
  const std::string inPath  = argv[1];
  const std::string outPath = argv[2];
  int radius = (argc == 4) ? std::max(0, std::atoi(argv[3])) : 2;

  // --- Load PNG to RGBA8 with lodepng ---
  std::vector<unsigned char> hostRGBA;
  unsigned w = 0, h = 0;
  unsigned err = lodepng::decode(hostRGBA, w, h, inPath); // RGBA8
  if (err) {
    std::cerr << "Decode error " << err << ": " << lodepng_error_text(err) << "\n";
    return 1;
  }
  if (w == 0 || h == 0) {
    std::cerr << "Empty image?\n";
    return 1;
  }

  const size_t numPixels = static_cast<size_t>(w) * static_cast<size_t>(h);
  const size_t numBytes  = numPixels * 4;

  // --- Allocate device buffers ---
  uint8_t *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in,  numBytes));
  CUDA_CHECK(cudaMalloc(&d_out, numBytes));

  // --- Upload to device ---
  CUDA_CHECK(cudaMemcpy(d_in, hostRGBA.data(), numBytes, cudaMemcpyHostToDevice));

  // --- Launch blur kernel ---
  dim3 block(16, 16);
  dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
  boxBlurRGBA<<<grid, block>>>(d_in, d_out, static_cast<int>(w), static_cast<int>(h), radius);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- Download result ---
  std::vector<unsigned char> hostOut(numBytes);
  CUDA_CHECK(cudaMemcpy(hostOut.data(), d_out, numBytes, cudaMemcpyDeviceToHost));

  // --- Save PNG with lodepng ---
  std::vector<unsigned char> png;
  err = lodepng::encode(png, hostOut, w, h); // auto color -> RGBA8
  if (err) {
    std::cerr << "Encode error " << err << ": " << lodepng_error_text(err) << "\n";
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 1;
  }
  unsigned saveErr = lodepng_save_file(png.data(), png.size(), outPath.c_str());
  if (saveErr) {
    std::cerr << "File save error " << saveErr << " (check path/permissions)\n";
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 1;
  }

  // --- Cleanup ---
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  std::cout << "Blurred " << inPath << " -> " << outPath
            << " (" << w << "x" << h << "), radius=" << radius << "\n";
  return 0;
}
