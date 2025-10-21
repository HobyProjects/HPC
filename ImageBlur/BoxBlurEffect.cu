// GaussianBlurEffect.cu
// CUDA 12.2+ compatible separable Gaussian blur for RGBA8 PNGs using LodePNG (C++ wrapper)
//
// Build (example):
//   nvcc -std=c++17 -O3 -use_fast_math -arch=sm_75 GaussianBlurEffect.cu lodepng.cpp -o gaussian_blur
// Adjust -arch=sm_XX for your GPU (e.g., sm_86 for RTX 30xx, sm_89 for RTX 40xx).
//
// Usage:
//   ./gaussian_blur input.png output.png  sigma [radius]
// If radius is omitted, it is computed as ceil(3*sigma).
//
// Notes:
// - This is a separable Gaussian: horizontal pass then vertical pass.
// - Alpha is processed the same as RGB so premultiplied-alpha inputs blur correctly.
// - The code intentionally prefers clarity over micro-optimizations.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "lodepng.h"  // C++ wrapper is enabled when compiled as C++ (see build cmd)

/* ---------- CUDA error guard ---------- */
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t _err = (call);                                               \
    if (_err != cudaSuccess) {                                               \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                \
                << " (" << static_cast<int>(_err) << ") at " << __FILE__     \
                << ":" << __LINE__ << "\\n";                                 \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

/* ---------- Utilities ---------- */
__device__ __forceinline__ int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

/* ---------- Separable Gaussian: horizontal then vertical ---------- */

// Horizontal pass: reads src, writes tmp (same pitch), applying 1D kernel along X.
// Pixels are RGBA (uchar4). Kernel weights are in global memory (d_kernel) of length (2*radius+1).
__global__ void gaussianHorizontal(const uchar4* __restrict__ src,
                                   uchar4* __restrict__ tmp,
                                   int width, int height, int pitch,  // pitch in bytes
                                   const float* __restrict__ d_kernel,
                                   int radius) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= width || y >= height) return;

  const int stride = pitch / sizeof(uchar4);
  const uchar4* row = src + y * stride;

  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  int ksize = 2 * radius + 1;
  for (int k = -radius; k <= radius; ++k) {
    int xx = clampi(x + k, 0, width - 1);
    uchar4 px = row[xx];
    float w = d_kernel[k + radius];
    acc.x += w * px.x;
    acc.y += w * px.y;
    acc.z += w * px.z;
    acc.w += w * px.w;
  }

  uchar4 out;
  out.x = static_cast<unsigned char>(fminf(fmaxf(acc.x, 0.f), 255.f));
  out.y = static_cast<unsigned char>(fminf(fmaxf(acc.y, 0.f), 255.f));
  out.z = static_cast<unsigned char>(fminf(fmaxf(acc.z, 0.f), 255.f));
  out.w = static_cast<unsigned char>(fminf(fmaxf(acc.w, 0.f), 255.f));

  tmp[y * stride + x] = out;
}

// Vertical pass: reads tmp, writes dst, applying 1D kernel along Y.
__global__ void gaussianVertical(const uchar4* __restrict__ tmp,
                                 uchar4* __restrict__ dst,
                                 int width, int height, int pitch,
                                 const float* __restrict__ d_kernel,
                                 int radius) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= width || y >= height) return;

  const int stride = pitch / sizeof(uchar4);

  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  int ksize = 2 * radius + 1;
  for (int k = -radius; k <= radius; ++k) {
    int yy = clampi(y + k, 0, height - 1);
    const uchar4 px = tmp[yy * stride + x];
    float w = d_kernel[k + radius];
    acc.x += w * px.x;
    acc.y += w * px.y;
    acc.z += w * px.z;
    acc.w += w * px.w;
  }

  uchar4 out;
  out.x = static_cast<unsigned char>(fminf(fmaxf(acc.x, 0.f), 255.f));
  out.y = static_cast<unsigned char>(fminf(fmaxf(acc.y, 0.f), 255.f));
  out.z = static_cast<unsigned char>(fminf(fmaxf(acc.z, 0.f), 255.f));
  out.w = static_cast<unsigned char>(fminf(fmaxf(acc.w, 0.f), 255.f));

  dst[y * stride + x] = out;
}

/* ---------- Host helpers ---------- */

static std::vector<float> makeGaussianKernel(float sigma, int radius) {
  if (radius <= 0) {
    radius = static_cast<int>(std::ceil(3.0 * sigma));
  }
  const int ksize = 2 * radius + 1;
  std::vector<float> k(ksize);
  const float twoSigma2 = 2.0f * sigma * sigma;
  float sum = 0.f;
  for (int i = -radius; i <= radius; ++i) {
    float w = std::exp(-(i * i) / twoSigma2);
    k[i + radius] = w;
    sum += w;
  }
  // normalize
  for (int i = 0; i < ksize; ++i) k[i] /= sum;
  return k;
}

static int decodePNG(const std::string& path, std::vector<unsigned char>& out,
                     unsigned& w, unsigned& h) {
  unsigned err = lodepng::decode(out, w, h, path);
  if (err) {
    std::cerr << "PNG decode error " << err << ": "
              << lodepng_error_text(err) << "\\n";
    return 1;
  }
  if (out.size() != w * h * 4) {
    std::cerr << "Unexpected PNG format (not RGBA8)\\n";
    return 2;
  }
  return 0;
}

static int encodePNG(const std::string& path, const std::vector<unsigned char>& data,
                     unsigned w, unsigned h) {
  unsigned err = lodepng::encode(path, data, w, h);
  if (err) {
    std::cerr << "PNG encode error " << err << ": "
              << lodepng_error_text(err) << "\\n";
    return 1;
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " input.png output.png sigma [radius]\\n";
    return 1;
  }
  const std::string inPath = argv[1];
  const std::string outPath = argv[2];
  const float sigma = std::stof(argv[3]);
  int radius = 0;
  if (argc >= 5) radius = std::stoi(argv[4]);

  if (sigma <= 0.f) {
    std::cerr << "Sigma must be > 0\\n";
    return 1;
  }

  std::vector<unsigned char> host; unsigned w=0, h=0;
  if (int e = decodePNG(inPath, host, w, h)) return e;

  // Device buffers (pitched 2D to be cache-friendly).
  uchar4* d_in = nullptr;
  uchar4* d_tmp = nullptr;
  uchar4* d_out = nullptr;
  size_t pitch = 0;
  CUDA_CHECK(cudaMallocPitch(&d_in, &pitch, w * sizeof(uchar4), h));
  CUDA_CHECK(cudaMallocPitch(&d_tmp, &pitch, w * sizeof(uchar4), h));
  CUDA_CHECK(cudaMallocPitch(&d_out, &pitch, w * sizeof(uchar4), h));

  // Copy input to device
  CUDA_CHECK(cudaMemcpy2D(d_in, pitch, host.data(), w * sizeof(uchar4),
                          w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

  // Build Gaussian kernel and upload
  auto h_kernel = makeGaussianKernel(sigma, radius);
  radius = (int)h_kernel.size() / 2; // ensure consistent with computed radius
  float* d_kernel = nullptr;
  CUDA_CHECK(cudaMalloc(&d_kernel, h_kernel.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(),
                        h_kernel.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Launch configuration
  dim3 block(32, 8);
  dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

  // Horizontal then vertical
  gaussianHorizontal<<<grid, block>>>(d_in, d_tmp, (int)w, (int)h, (int)pitch, d_kernel, radius);
  CUDA_CHECK(cudaGetLastError());
  gaussianVertical<<<grid, block>>>(d_tmp, d_out, (int)w, (int)h, (int)pitch, d_kernel, radius);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy back
  std::vector<unsigned char> out(host.size());
  CUDA_CHECK(cudaMemcpy2D(out.data(), w * sizeof(uchar4), d_out, pitch,
                          w * sizeof(uchar4), h, cudaMemcpyDeviceToHost));

  // Save
  if (int e = encodePNG(outPath, out, w, h)) {
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_kernel));
    return e;
  }

  // Cleanup
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_tmp));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_kernel));

  std::cout << "Gaussian-blurred " << inPath << " -> " << outPath
            << " (" << w << "x" << h << "), sigma=" << sigma
            << ", radius=" << radius << "\\n";
  return 0;
}
