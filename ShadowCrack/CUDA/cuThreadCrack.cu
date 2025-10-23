// Shadow Crack using CUDA (CUDA 12.2+ compatible)

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <inttypes.h>
#include <unistd.h>
#include <errno.h>
#include <crypt.h>

// --- CUDA headers (required on CUDA 12.x+)
#include <cuda_runtime.h>

#ifndef __CUDACC__
#error "Compile with nvcc (CUDA compiler)"
#endif

// Require CUDA Runtime 12.2 or newer
#ifndef CUDART_VERSION
#error "CUDART_VERSION not defined — are you compiling with nvcc?"
#endif
#if CUDART_VERSION < 12020
#error "This program requires CUDA 12.2 or newer."
#endif

// ---------------- Config ----------------
#define DEFAULT_TPB            256
#define DEFAULT_MAXLEN         6
#define MAX_CAND_LEN           32
#define DEFAULT_BATCH_SIZE     (1u<<20)  // 1,048,576
#define DISPLAY_INTERVAL_NS    200000000L

// ---------------- Charset ----------------
static const char HOST_CHARSET[] =
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789";

static const int HOST_CHARSET_LEN = (int)(sizeof(HOST_CHARSET) - 1);

// Device constants
__constant__ char DEV_CHARSET[ sizeof(HOST_CHARSET) ];
__constant__ int  DEV_CHARSET_LEN;

// ---------------- Utilities ----------------
static inline uint64_t pow_u64_host(uint64_t base, int exp) {
    uint64_t r = 1;
    for (int i = 0; i < exp; ++i) {
        if (r > UINT64_MAX / base) return UINT64_MAX; // overflow clamp
        r *= base;
    }
    return r;
}

static inline double timespec_diff_s(const struct timespec *a, const struct timespec *b) {
    return (double)(a->tv_sec - b->tv_sec) + (double)(a->tv_nsec - b->tv_nsec) / 1e9;
}

static inline void cuda_or_die(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] %s failed: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

// ---------------- Device code ----------------
__device__ __forceinline__ void idx_to_candidate_u64(uint64_t idx, int len, char* out) {
    // Base-N index -> string over DEV_CHARSET
    for (int pos = len - 1; pos >= 0; --pos) {
        int sel = (int)(idx % (uint64_t)DEV_CHARSET_LEN);
        out[pos] = DEV_CHARSET[sel];
        idx /= (uint64_t)DEV_CHARSET_LEN;
    }
}

__global__ void generate_candidates_kernel(
    uint64_t start_index,
    uint32_t count,
    int len,
    char* __restrict__ out,
    uint32_t stride)
{
    uint64_t gtid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (gtid >= (uint64_t)count) return;

    uint64_t idx = start_index + gtid;
    char* dst = out + (size_t)gtid * (size_t)stride;
    idx_to_candidate_u64(idx, len, dst);
    dst[len] = '\0';
}

// ---------------- Main ----------------
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <password> [threads_per_block] [max_len] [batch_size]\n", argv[0]);
        return 1;
    }

    const char* password = argv[1];
    int threads_per_block = (argc >= 3) ? atoi(argv[2]) : DEFAULT_TPB;
    int maxlen           = (argc >= 4) ? atoi(argv[3]) : DEFAULT_MAXLEN;
    uint32_t batch_size  = (argc >= 5) ? (uint32_t)strtoul(argv[4], NULL, 10) : DEFAULT_BATCH_SIZE;

    if (threads_per_block < 32) threads_per_block = DEFAULT_TPB;
    if (maxlen < 1) maxlen = DEFAULT_MAXLEN;
    if (maxlen > MAX_CAND_LEN) maxlen = MAX_CAND_LEN;
    if (batch_size == 0u) batch_size = DEFAULT_BATCH_SIZE;

    // Hash the provided password with SHA-256 ($5$) to get target
    const char* const salt_prefix = "$5$"; // glibc crypt: $5$ = SHA-256
    char* target_hash = crypt(password, salt_prefix);
    if (target_hash == NULL) {
        perror("crypt");
        return 1;
    }

    printf("---------------------------------------------------------------------\n");
    printf("                     Shadow Crack using CUDA                         \n");
    printf("---------------------------------------------------------------------\n");
    printf("Target hash: %s\n", target_hash);
    printf("TPB: %d  maxlen: %d  | batch_size: %u\n", threads_per_block, maxlen, batch_size);
    printf("Charset size: %d\n", HOST_CHARSET_LEN);
    printf("CUDA runtime version: %d\n", CUDART_VERSION);
    printf("---------------------------------------------------------------------\n");

    // Copy constants to device
    cuda_or_die(cudaMemcpyToSymbol(DEV_CHARSET,     HOST_CHARSET,     sizeof(HOST_CHARSET), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol(DEV_CHARSET)");
    cuda_or_die(cudaMemcpyToSymbol(DEV_CHARSET_LEN, &HOST_CHARSET_LEN, sizeof(int),          0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol(DEV_CHARSET_LEN)");

    const uint32_t max_stride = (uint32_t)(MAX_CAND_LEN + 1);
    const size_t   host_bytes = (size_t)batch_size * (size_t)max_stride;

    // Host buffer (pageable is fine; you can switch to pinned for more speed)
    char* h_buf = (char*) malloc(host_bytes);
    if (!h_buf) {
        fprintf(stderr, "malloc host buffer failed (size=%zu)\n", host_bytes);
        return 1;
    }

    // Device buffer
    char* d_buf = NULL;
    cuda_or_die(cudaMalloc((void**)&d_buf, host_bytes), "cudaMalloc(d_buf)");

    uint64_t attempts = 0;
    int found = 0;
    char found_password[MAX_CAND_LEN + 1] = {0};
    char last_combo[MAX_CAND_LEN + 1] = {0};

    struct timespec t_start{}, t_last_print{}, t_now{};
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    t_last_print = t_start;

    // ↓↓↓ declare these BEFORE any possible 'goto cleanup'
    double total_elapsed = 0.0;
    double avg_rate = 0.0;

    struct crypt_data cdata;
    memset(&cdata, 0, sizeof(cdata));

    for (int len = 1; len <= maxlen && !found; ++len) {
        uint64_t total = pow_u64_host((uint64_t)HOST_CHARSET_LEN, len);
        if (total == UINT64_MAX) {
            fprintf(stderr, "Search space overflow for length %d\n", len);
            continue;
        }

        const uint32_t stride = (uint32_t)(len + 1);
        uint64_t start = 0;

        while (start < total && !found) {
            const uint64_t remaining = total - start;
            const uint32_t this_count = (remaining >= (uint64_t)batch_size)
                                        ? batch_size
                                        : (uint32_t)remaining;

            const unsigned int tpb = (unsigned int)threads_per_block;
            const unsigned int blocks = (unsigned int)((this_count + tpb - 1u) / tpb);

            generate_candidates_kernel<<<blocks, tpb>>>(start, this_count, len, d_buf, stride);

            cudaError_t kerr = cudaDeviceSynchronize();
            if (kerr != cudaSuccess) {
                fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(kerr));
                goto cleanup;  // now legal: we’re not skipping any new initializations
            }

            const size_t copy_bytes = (size_t)this_count * (size_t)stride;
            cuda_or_die(cudaMemcpy(h_buf, d_buf, copy_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H)");

            for (uint32_t i = 0; i < this_count; ++i) {
                char* cand = h_buf + (size_t)i * (size_t)stride;
                cand[len] = '\0';

                strncpy(last_combo, cand, MAX_CAND_LEN);
                last_combo[MAX_CAND_LEN] = '\0';

                char* res = crypt_r(cand, salt_prefix, &cdata);
                ++attempts;

                if (res && strcmp(res, target_hash) == 0) {
                    strncpy(found_password, cand, MAX_CAND_LEN);
                    found_password[MAX_CAND_LEN] = '\0';
                    found = 1;
                    break;
                }

                clock_gettime(CLOCK_MONOTONIC, &t_now);
                long dnsec = (t_now.tv_sec - t_last_print.tv_sec) * 1000000000L
                           + (t_now.tv_nsec - t_last_print.tv_nsec);
                if (dnsec >= DISPLAY_INTERVAL_NS) {
                    double elapsed = timespec_diff_s(&t_now, &t_start);
                    double rate = (elapsed > 0.0) ? ((double)attempts / elapsed) : 0.0;
                    printf("\rTried: %" PRIu64 "  Last: %s  Rate: %.0f/sec  Elapsed: %.1fs  ",
                           attempts,
                           last_combo[0] ? last_combo : "-",
                           rate, elapsed);
                    fflush(stdout);
                    t_last_print = t_now;
                }
            }

            start += (uint64_t)this_count;
        }
    }

    // compute after all jumps are done
    clock_gettime(CLOCK_MONOTONIC, &t_now);
    total_elapsed = timespec_diff_s(&t_now, &t_start);
    avg_rate = (total_elapsed > 0.0) ? ((double)attempts / total_elapsed) : 0.0;

    printf("\n\n---------------------------------------------------------------------\n");
    if (found) {
        printf("Password Found: %s\n", found_password);
    } else {
        printf("No match found up to length %d\n", maxlen);
    }
    printf("Total attempts: %" PRIu64 "  Time: %.2fs  Avg rate: %.0f/sec\n",
           attempts, total_elapsed, avg_rate);
    printf("---------------------------------------------------------------------\n");

cleanup:
    if (d_buf) cudaFree(d_buf);
    free(h_buf);
    return 0;
}
