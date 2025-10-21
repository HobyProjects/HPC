// BoxBlurEffect.cu
// CUDA 12.2+ compatible box blur for RGBA8 PNGs using LodePNG (C++ wrapper)

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "lodepng.h"  // C++ wrapper is enabled when compiled as C++ (see build cmd)

/* ---------- CUDA error guard ---------- */
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t _err = (call);                                               \
    if (_err != cudaSuccess) {                                               \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                \
                << " (" << static_cast<int>(_err) << ") at " << __FILE__     \
                << ":" << __LINE__ << std::endl;                             \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

/* ---------- Device helpers ---------- */
__device__ __forceinline__ int clampi(int v, int lo, int hi) {
  // CUDA provides min/max in the global namespace for device code
  return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void boxBlurRGBA(const uint8_t* __restrict__ in,
                            uint8_t* __restrict__ out,
                            int width, int height, int radius) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int channels = 4;
  int rsum = 0, gsum = 0, bsum = 0, asum = 0;
  int count = 0;

  const int y0 = clampi(y - radius, 0, height - 1);
  const int y1 = clampi(y + radius, 0, height - 1);
  const int x0 = clampi(x - radius, 0, width  - 1);
  const int x1 = clampi(x + radius, 0, width  - 1);

  for (int yy = y0; yy <= y1; ++yy) {
    const int row = yy * width;
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

static void usage(const char* exe) {
  std::cerr << "Usage: " << exe << " <input.png> <output.png> [radius]\n";
}

/* ---------- Host main ---------- */
int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) { usage(argv[0]); return 1; }
  const std::string inPath  = argv[1];
  const std::string outPath = argv[2];
  const int radius = (argc == 4) ? std::max(0, std::stoi(argv[3])) : 3;

  // --- Decode PNG to RGBA8 (host) ---
  std::vector<unsigned char> hostRGBA;
  unsigned w = 0, h = 0;
  unsigned err = lodepng::decode(hostRGBA, w, h, inPath);  // RGBA8 out
  if (err) {
    std::cerr << "Decode error " << err << ": " << lodepng_error_text(err) << "\n";
    return 1;
  }
  if (w == 0 || h == 0) {
    std::cerr << "Invalid image dimensions.\n";
    return 1;
  }

  const size_t numPixels = static_cast<size_t>(w) * static_cast<size_t>(h);
  const size_t numBytes  = numPixels * 4u;

  // --- Allocate device buffers ---
  uint8_t *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in,  numBytes));
  CUDA_CHECK(cudaMalloc(&d_out, numBytes));

  // --- Upload image ---
  CUDA_CHECK(cudaMemcpy(d_in, hostRGBA.data(), numBytes, cudaMemcpyHostToDevice));

  // --- Launch kernel ---
  dim3 block(16, 16);
  dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
  boxBlurRGBA<<<grid, block>>>(d_in, d_out, static_cast<int>(w), static_cast<int>(h), radius);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- Download result ---
  std::vector<unsigned char> hostOut(numBytes);
  CUDA_CHECK(cudaMemcpy(hostOut.data(), d_out, numBytes, cudaMemcpyDeviceToHost));

  // --- Encode and save PNG (host) ---
  std::vector<unsigned char> png;
  err = lodepng::encode(png, hostOut, w, h); // auto choose PNG color; input is RGBA8
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
