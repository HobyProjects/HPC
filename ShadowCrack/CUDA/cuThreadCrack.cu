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
#include <crypt.h>
#include <errno.h>

#ifndef __CUDACC__
#error "Compile with nvcc (CUDA compiler)"
#endif

#define DEFAULT_TPB        256
#define DEFAULT_MAXLEN     6
#define MAX_CAND_LEN       32
#define DEFAULT_BATCH_SIZE (1<<20)  
#define DISPLAY_INTERVAL_NS 200000000L 

static const char HOST_CHARSET[] =
  "abcdefghijklmnopqrstuvwxyz"
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  "0123456789";

static const int HOST_CHARSET_LEN = (int)(sizeof(HOST_CHARSET)-1);

__constant__ char DEV_CHARSET[ sizeof(HOST_CHARSET) ];
__constant__ int  DEV_CHARSET_LEN;
__device__ __forceinline__ void idx_to_candidate_u64(uint64_t idx, int len, char* out) {
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
    char* out,      
    uint32_t stride) 
{
    uint64_t gtid = (uint64_t)blockIdx.x * blockDim.x + (uint64_t)threadIdx.x;
    if (gtid >= count) return;

    uint64_t idx = start_index + gtid;
    char* dst = out + (size_t)gtid * (size_t)stride;
    idx_to_candidate_u64(idx, len, dst);
    dst[len] = '\0';
}

static uint64_t pow_u64_host(uint64_t base, int exp) {
    uint64_t r = 1;
    for (int i = 0; i < exp; ++i) {
        if (r > UINT64_MAX / base) return UINT64_MAX;
        r *= base;
    }
    return r;
}

static double timespec_diff_s(const struct timespec *a, const struct timespec *b) {
    return (double)(a->tv_sec - b->tv_sec) + (double)(a->tv_nsec - b->tv_nsec) / 1e9;
}


int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <password> [threads_per_block] [max_len] [batch_size]\n", argv[0]);
        return 1;
    }

    const char* password = argv[1];
    int threads_per_block = (argc >= 3) ? atoi(argv[2]) : DEFAULT_TPB;
    int maxlen = (argc >= 4) ? atoi(argv[3]) : DEFAULT_MAXLEN;
    uint32_t batch_size = (argc >= 5) ? (uint32_t)strtoul(argv[4], NULL, 10) : DEFAULT_BATCH_SIZE;

    if (threads_per_block < 32) threads_per_block = DEFAULT_TPB;
    if (maxlen < 1) maxlen = DEFAULT_MAXLEN;
    if (maxlen > MAX_CAND_LEN) maxlen = MAX_CAND_LEN;
    if (batch_size == 0) batch_size = DEFAULT_BATCH_SIZE;

    const char* salt_prefix = "$5$AbfFA234fFCfgh4&92$"; 
    char* target_hash = crypt(password, salt_prefix);
    if (target_hash == NULL) {
        perror("crypt");
        return 1;
    }

    printf("Target hash: %s\n", target_hash);
    printf("TPB: %d  maxlen: %d  batch_size: %u\n", threads_per_block, maxlen, batch_size);
    printf("Charset size: %d\n", HOST_CHARSET_LEN);
    printf("Starting GPU-assisted brute-force (Ctrl+C to abort)...\n\n");

    cudaError_t cerr;
    cerr = cudaMemcpyToSymbol(DEV_CHARSET, HOST_CHARSET, sizeof(HOST_CHARSET));
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol DEV_CHARSET failed: %s\n", cudaGetErrorString(cerr));
        return 1;
    }
    cerr = cudaMemcpyToSymbol(DEV_CHARSET_LEN, &HOST_CHARSET_LEN, sizeof(int));
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol DEV_CHARSET_LEN failed: %s\n", cudaGetErrorString(cerr));
        return 1;
    }

    const uint32_t max_stride = (uint32_t)(MAX_CAND_LEN + 1);
    size_t host_bytes = (size_t)batch_size * (size_t)max_stride;
    char* h_buf = (char*) malloc(host_bytes);
    if (!h_buf) {
        fprintf(stderr, "malloc host buffer failed\n");
        return 1;
    }
    char* d_buf = NULL;
    cerr = cudaMalloc((void**)&d_buf, host_bytes);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_buf failed: %s\n", cudaGetErrorString(cerr));
        free(h_buf);
        return 1;
    }

    uint64_t attempts = 0;
    int found = 0;
    char found_password[MAX_CAND_LEN+1];
    found_password[0] = '\0';
    char last_combo[MAX_CAND_LEN+1];
    last_combo[0] = '\0';

    struct timespec t_start, t_last_print, t_now;
    double total_elapsed = 0.0;
    double avg_rate = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &t_start);
    t_last_print = t_start;

    struct crypt_data cdata;
    memset(&cdata, 0, sizeof(cdata));

    for (int len = 1; len <= maxlen && !found; ++len) {
        uint64_t total = pow_u64_host((uint64_t)HOST_CHARSET_LEN, len);
        if (total == UINT64_MAX) {
            fprintf(stderr, "Search space overflow for length %d\n", len);
            continue;
        }

        uint32_t stride = (uint32_t)(len + 1);
        uint64_t start = 0;

        while (start < total && !found) {
            uint32_t this_count = (uint32_t)((start + batch_size <= total) ? batch_size : (total - start));
            uint32_t blocks = (this_count + (uint32_t)threads_per_block - 1) / (uint32_t)threads_per_block;

            generate_candidates_kernel<<<(size_t)blocks, (size_t)threads_per_block>>>(start, this_count, len, d_buf, max_stride);
            cerr = cudaGetLastError();
            if (cerr != cudaSuccess) {
                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cerr));
                goto cleanup;
            }

            size_t copy_bytes = (size_t)this_count * (size_t)max_stride;
            cerr = cudaMemcpy(h_buf, d_buf, copy_bytes, cudaMemcpyDeviceToHost);
            if (cerr != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(cerr));
                goto cleanup;
            }

            uint32_t i;
            for (i = 0; i < this_count; ++i) {
                char* cand = h_buf + (size_t)i * (size_t)max_stride;
                cand[len] = '\0';
                strncpy(last_combo, cand, MAX_CAND_LEN);
                last_combo[MAX_CAND_LEN] = '\0';

                char* res = crypt_r(cand, salt_prefix, &cdata);
                attempts++;

                if (res && strcmp(res, target_hash) == 0) {
                    strncpy(found_password, cand, MAX_CAND_LEN);
                    found_password[MAX_CAND_LEN] = '\0';
                    found = 1;
                    break;
                }

                clock_gettime(CLOCK_MONOTONIC, &t_now);
                long dnsec = (t_now.tv_sec - t_last_print.tv_sec) * 1000000000L + (t_now.tv_nsec - t_last_print.tv_nsec);
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

            start += this_count;
        }
    }

    printf("\n\n");
    if (found) {
        printf("Password Found: %s\n", found_password);
    } else {
        printf("No match found up to length %d\n", maxlen);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_now);
    total_elapsed = timespec_diff_s(&t_now, &t_start);
    avg_rate = (total_elapsed > 0.0) ? ((double)attempts / total_elapsed) : 0.0;
    printf("Total attempts: %" PRIu64 "  Time: %.2fs  Avg rate: %.0f/sec\n", attempts, total_elapsed, avg_rate);

cleanup:
    cudaFree(d_buf);
    free(h_buf);
    return 0;
}
