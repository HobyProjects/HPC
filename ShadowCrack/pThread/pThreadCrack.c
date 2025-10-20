#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <crypt.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdatomic.h>

#define DEFAULT_THREADS 4
#define DEFAULT_MAXLEN 6
#define MAX_CAND_LEN 32
#define DISPLAY_INTERVAL_NS 200000000L 

static const char *CHARSET = "abcdefghijklmnopqrstuvwxyz"
                             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                             "0123456789";
static const size_t CHARSET_LEN = 26+26+10;

typedef struct {
    int id;
    int nthreads;
    int maxlen;
    const char *salt;          
    const char *target_hash;    
} thread_arg_t;

atomic_uint_fast64_t attempts = 0;
atomic_int found_flag = 0;
char found_password[MAX_CAND_LEN+1] = {0};

pthread_mutex_t last_combo_mutex = PTHREAD_MUTEX_INITIALIZER;
char last_combo[MAX_CAND_LEN+1] = {0};

static inline uint64_t pow_u64(size_t base, int exp) {
    uint64_t r = 1;
    for (int i = 0; i < exp; ++i) {
        if (r > UINT64_MAX / base) return UINT64_MAX;
        r *= base;
    }
    return r;
}

static void index_to_candidate(uint64_t index, int len, char *out) {
    out[len] = '\0';
    for (int pos = len - 1; pos >= 0; --pos) {
        out[pos] = CHARSET[index % CHARSET_LEN];
        index /= CHARSET_LEN;
    }
}

void *worker_thread(void *varg) {
    thread_arg_t *arg = (thread_arg_t*)varg;

    struct crypt_data cdata;
    memset(&cdata, 0, sizeof(cdata));

    char candidate[MAX_CAND_LEN+1];

    for (int len = 1; len <= arg->maxlen && !atomic_load(&found_flag); ++len) {
        uint64_t total = pow_u64(CHARSET_LEN, len);
        if (total == UINT64_MAX) {
            continue;
        }

        for (uint64_t idx = arg->id; idx < total && !atomic_load(&found_flag); idx += arg->nthreads) {
            index_to_candidate(idx, len, candidate);
            pthread_mutex_lock(&last_combo_mutex);
            strncpy(last_combo, candidate, sizeof(last_combo)-1);
            last_combo[sizeof(last_combo)-1] = '\0';
            pthread_mutex_unlock(&last_combo_mutex);

            char *res = crypt_r(candidate, arg->salt, &cdata);
            if (!res) continue;

            atomic_fetch_add(&attempts, 1);
            if (strcmp(res, arg->target_hash) == 0) {
                if (!atomic_exchange(&found_flag, 1)) {
                    strncpy(found_password, candidate, sizeof(found_password)-1);
                    found_password[sizeof(found_password)-1] = '\0';
                }
                break;
            }

            if (atomic_load(&found_flag)) break;
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <password> [num_threads] [max_len]\n", argv[0]);
        return 1;
    }

    const char *password = argv[1];
    int nthreads = (argc >= 3) ? atoi(argv[2]) : DEFAULT_THREADS;
    int maxlen = (argc >= 4) ? atoi(argv[3]) : DEFAULT_MAXLEN;

    if (nthreads < 1) nthreads = DEFAULT_THREADS;
    if (maxlen < 1) maxlen = DEFAULT_MAXLEN;
    if (maxlen > MAX_CAND_LEN) maxlen = MAX_CAND_LEN;

    const char *salt_prefix = "$5$sHaDoW_CraCKsalt$"; 
    char *target_hash = crypt(password, salt_prefix);
    if (!target_hash) {
        perror("crypt");
        return 1;
    }

    printf("Target hash: %s\n", target_hash);
    printf("Threads: %d, max candidate length: %d\n", nthreads, maxlen);
    printf("Charset size: %zu\n", CHARSET_LEN);
    printf("Starting brute-force (press Ctrl+C to abort)...\n\n");

    pthread_t *tids = malloc(sizeof(pthread_t) * nthreads);
    thread_arg_t *targs = malloc(sizeof(thread_arg_t) * nthreads);
    for (int i = 0; i < nthreads; ++i) {
        targs[i].id = i;
        targs[i].nthreads = nthreads;
        targs[i].maxlen = maxlen;
        targs[i].salt = salt_prefix;
        targs[i].target_hash = target_hash;
    }

    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int i = 0; i < nthreads; ++i) {
        if (pthread_create(&tids[i], NULL, worker_thread, &targs[i]) != 0) {
            perror("pthread_create");
            return 1;
        }
    }

    uint64_t last_attempts = 0;
    while (!atomic_load(&found_flag)) {
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = DISPLAY_INTERVAL_NS;
        nanosleep(&ts, NULL);

        uint64_t cur_attempts = atomic_load(&attempts);

        struct timespec t_now;
        clock_gettime(CLOCK_MONOTONIC, &t_now);
        double elapsed = (t_now.tv_sec - t_start.tv_sec) + (t_now.tv_nsec - t_start.tv_nsec)/1e9;

        double rate = (elapsed > 0.0) ? ((double)cur_attempts / elapsed) : 0.0;

        char last[sizeof(last_combo)];
        pthread_mutex_lock(&last_combo_mutex);
        strncpy(last, last_combo, sizeof(last)-1);
        last[sizeof(last)-1] = '\0';
        pthread_mutex_unlock(&last_combo_mutex);

        if (rate > 0.0) {
            printf("\rTried: %" PRIu64 "  Last: %s  Rate: %.0f/sec  Elapsed: %.1fs  ",
                   cur_attempts, last[0] ? last : "-", rate, elapsed);
        } else {
            printf("\rTried: %" PRIu64 "  Last: %s  Rate: -/sec  Elapsed: %.1fs  ",
                   cur_attempts, last[0] ? last : "-", elapsed);
        }
        fflush(stdout);

        last_attempts = cur_attempts;
    }

    printf("\n\n");
    if (found_password[0]) {
        printf("Password Found : %s\n", found_password);
    } else {
        printf("A thread set found flag, but password buffer empty (race). Check.\n");
    }

    for (int i = 0; i < nthreads; ++i) {
        pthread_join(tids[i], NULL);
    }

    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double total_elapsed = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec)/1e9;
    uint64_t total_attempts = atomic_load(&attempts);
    printf("Total attempts: %" PRIu64 "  Time: %.2fs  Avg rate: %.0f/sec\n",
           total_attempts, total_elapsed, total_elapsed > 0.0 ? total_attempts / total_elapsed : 0.0);

    free(tids);
    free(targs);
    return 0;
}
