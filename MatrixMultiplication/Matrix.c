#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define MAT_VALID 1
#define MAT_INVALID -1
#define MAT_INVALID_PTR NULL
#define MAT_MAX_THREADS 4

typedef struct {
    int mr, mc;
    float **mat_ptr;
} mat;

typedef struct {
    mat** list;
    int count;
} mat_list;

void  mat_help();
mat*  mat_create(int r, int c);
void  mat_delete(mat* m);
void  mat_delete_list(mat_list* ml);

int   mat_is_valid(const mat* m);
int   mat_validate(const mat* lh, const mat* rh);
int   mat_max_dimension_of_pair(const mat* A, const mat* B);

mat*  mat_mul_parallel(const mat* lh, const mat* rh, int thread_count);

void  mat_write_matrix_to_stream(FILE* out, const mat* m);
int   mat_parse_thread_count(const char* str);

int      mat_verify_file(const char* file);
mat*     mat_file_read(const char* filename);
mat_list mat_file_read_all(const char* filename);

static void skip_delims(FILE* f);
static void* mat_init_buffers(size_t n, size_t sz);
static int   mat_clamp_threads(int requested, int rows);

// =====================================================

int main(int argc, char** argv) {
    if (argc == 1) {
        mat_help();
        return -1;
    }

    const char* file_name = NULL;
    if (argc >= 2 && argv[1]) {
        file_name = argv[1];
        if (file_name && !mat_verify_file(file_name)) {
            printf("[ERROR]: Unable to locate or verify %s\n", argv[1]);
            return -2;
        }
    }

    int requested_threads = 0;
    if (argc >= 3 && argv[2]) {
        requested_threads = mat_parse_thread_count(argv[2]);
        if (requested_threads < 0) {
            printf("[WARN]: Invalid thread count. Using system/OMP default.\n");
            requested_threads = 0;
        }
    }

    printf("----------------------------------------------\n");
    printf(" FILE NAME    : %s\n", file_name);
    printf(" THREAD COUNT : %d\n", requested_threads);
    printf("----------------------------------------------\n");

    mat_list ml = mat_file_read_all(file_name);
    if (!ml.list || ml.count == 0) {
        fprintf(stderr, "[ERROR]: No matrices were read from '%s'. Check the file format.\n", file_name);
        return -3;
    }

    if (ml.count % 2 != 0) {
        fprintf(stderr, "[WARN]: Odd number of matrices (%d). The last one will be ignored.\n", ml.count);
    }

    char out_name[1024];
    snprintf(out_name, sizeof(out_name), "./results.txt");
    FILE* out = fopen(out_name, "w");
    if (!out) {
        fprintf(stderr, "[ERROR]: Could not open output file '%s' for writing.\n", out_name);
        mat_delete_list(&ml);
        return -4;
    }

    int results_written = 0;
    for (int i = 0; i + 1 < ml.count; i += 2) {
        const mat* A = ml.list[i];
        const mat* B = ml.list[i + 1];

        printf("[INFO]: Pair %d --> A(%d x %d) * B(%d x %d)\n",
               (i / 2) + 1, A->mr, A->mc, B->mr, B->mc);

        if (mat_validate(A, B) != MAT_VALID) {
            fprintf(stderr, "[ERROR]: Cannot multiply this pair. Skipping.\n");
            fprintf(out, "# Skipped pair %d: incompatible dimensions (%d x %d) * (%d x %d)\n\n",
                    (i / 2) + 1, A->mr, A->mc, B->mr, B->mc);
            continue;
        }

        int C_rows = A->mr;
        int biggest_dim = mat_max_dimension_of_pair(A, B);

        int T = requested_threads;
        if (T <= 0) T = mat_clamp_threads(0, C_rows);
        if (T > biggest_dim) T = biggest_dim;
        if (T > C_rows)      T = C_rows;
        if (T < 1)           T = 1;

        mat* C = mat_mul_parallel(A, B, T);
        if (!C) {
            fprintf(stderr, "[ERROR]: Multiplication failed for pair %d\n", (i / 2) + 1);
            fprintf(out, "# Error computing result for pair %d\n\n", (i / 2) + 1);
            continue;
        }

        fprintf(out, "# Result %d: A(%d x %d) * B(%d x %d) using %d thread(s)\n",
                (i / 2) + 1, A->mr, A->mc, B->mr, B->mc, T);
        mat_write_matrix_to_stream(out, C);
        ++results_written;

        mat_delete(C);
    }

    fclose(out);
    printf("[INFO]: Wrote %d result matrix/matrices to: %s\n", results_written, out_name);

    mat_delete_list(&ml);
    return 0;
}

void mat_help() {
    printf("----------------------------------------------\n");
    printf(" Usage: ./Matrix <file-name> <thread-count>\n");
    printf("----------------------------------------------\n");
    printf(" <file-name>     : Path to your matrix file\n");
    printf(" <thread-count>  : Number of threads to use (<= biggest matrix dimension)\n");
    printf("----------------------------------------------\n");
    printf(" Example: ./Matrix MatData.txt 4\n");
    printf("----------------------------------------------\n");
}

int mat_parse_thread_count(const char* str) {
    int value = 0;
    if (str && sscanf(str, "%d", &value) == 1) return value;
    return -1;
}

int mat_max_dimension_of_pair(const mat* A, const mat* B) {
    int m = A->mr;
    if (A->mc > m) m = A->mc;
    if (B->mr > m) m = B->mr;
    if (B->mc > m) m = B->mc;
    return m;
}

void mat_write_matrix_to_stream(FILE* out, const mat* m) {
    fprintf(out, "%d,%d\n", m->mr, m->mc);
    for (int i = 0; i < m->mr; ++i) {
        for (int j = 0; j < m->mc; ++j) fprintf(out, "%.2f ", m->mat_ptr[i][j]);
        fprintf(out, "\n");
    }
    fprintf(out, "\n");
}

mat* mat_create(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "[ERROR]: Invalid matrix size %d x %d\n", rows, cols);
        return MAT_INVALID_PTR;
    }

    mat* m = (mat*)malloc(sizeof(mat));
    if (!m) {
        fprintf(stderr, "[ERROR]: malloc(mat) failed\n");
        return MAT_INVALID_PTR;
    }

    m->mr = rows;
    m->mc = cols;

    m->mat_ptr = (float**)mat_init_buffers((size_t)rows, sizeof(float*));
    if (!m->mat_ptr) {
        fprintf(stderr, "[ERROR]: calloc( row pointers ) failed\n");
        free(m);
        return MAT_INVALID_PTR;
    }

    for (int i = 0; i < rows; ++i) {
        m->mat_ptr[i] = (float*)mat_init_buffers((size_t)cols, sizeof(float));
        if (!m->mat_ptr[i]) {
            fprintf(stderr, "[ERROR]: calloc(row %d) failed\n", i);
            for (int r = 0; r < i; ++r) free(m->mat_ptr[r]);
            free(m->mat_ptr);
            free(m);
            return MAT_INVALID_PTR;
        }
    }

    return m;
}

void mat_delete(mat* m) {
    if (!m) return;
    if (m->mat_ptr) {
        for (int i = 0; i < m->mr; ++i) free(m->mat_ptr[i]);
        free(m->mat_ptr);
    }
    free(m);
}

void mat_delete_list(mat_list* ml) {
    if (!ml || !ml->list) return;
    for (int i = 0; i < ml->count; ++i) {
        if (ml->list[i]) mat_delete(ml->list[i]);
    }
    free(ml->list);
    ml->list = NULL;
    ml->count = 0;
}

int mat_validate(const mat* lh, const mat* rh) {
    if (!lh || !rh) {
        fprintf(stderr, "[ERROR]: Null matrix pointer!\n");
        return MAT_INVALID;
    }
    if (lh->mc != rh->mr) {
        fprintf(stderr, "[ERROR]: Incompatible matrix dimensions: (%d x %d) * (%d x %d)\n",
                lh->mr, lh->mc, rh->mr, rh->mc);
        return MAT_INVALID;
    }
    return MAT_VALID;
}

int mat_is_valid(const mat* m) {
    if (!m || m->mat_ptr == MAT_INVALID_PTR) {
        fprintf(stderr, "[ERROR]: Null matrix pointer!\n");
        return MAT_INVALID;
    }
    if (m->mr <= 0 || m->mc <= 0) {
        fprintf(stderr, "[ERROR]: Invalid matrix dimensions: (%d x %d)\n", m->mr, m->mc);
        return MAT_INVALID;
    }
    return MAT_VALID;
}

// ---------- OpenMP multiply ----------
mat* mat_mul_parallel(const mat* lh, const mat* rh, int thread_count) {
    if (mat_validate(lh, rh) != MAT_VALID) return MAT_INVALID_PTR;

    mat* C = mat_create(lh->mr, rh->mc);
    if (!C) return MAT_INVALID_PTR;

    int T = mat_clamp_threads(thread_count, C->mr);
#ifdef _OPENMP
    omp_set_num_threads(T);
#endif

#pragma omp parallel for schedule(static) num_threads(T)
    for (int i = 0; i < C->mr; ++i) {
        float* Ci = C->mat_ptr[i];
        for (int j = 0; j < C->mc; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < lh->mc; ++k) {
                sum += lh->mat_ptr[i][k] * rh->mat_ptr[k][j];
            }
            Ci[j] = sum;
        }
    }

    return C;
}

int mat_verify_file(const char* file) {
    if (!file) {
        printf("No file name provided!!");
        return MAT_INVALID;
    }
    FILE* fp = fopen(file, "r");
    if (!fp) {
        fprintf(stderr, "[ERROR]: Could not open file '%s' for reading\n", file);
        return MAT_INVALID;
    }
    fclose(fp);
    return MAT_VALID;
}

static void skip_delims(FILE* f) {
    int ch;
    for (;;) {
        ch = fgetc(f);
        if (ch == EOF) break;
        if (ch == ',' || isspace((unsigned char)ch)) continue;
        ungetc(ch, f);
        break;
    }
}

mat* mat_file_read(const char* filename) {
    if (!filename) {
        fprintf(stderr, "[ERROR]: Null filename!\n");
        return MAT_INVALID_PTR;
    }
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[ERROR]: Could not open file '%s' for reading\n", filename);
        return MAT_INVALID_PTR;
    }

    int rows = 0, cols = 0;
    char line[4096];
    int got_header = 0;

    while (fgets(line, sizeof(line), fp)) {
        int only_ws = 1;
        for (char* p = line; *p; ++p) {
            if (!isspace((unsigned char)*p)) { only_ws = 0; break; }
        }
        if (only_ws) continue;
        if (sscanf(line, " %d , %d", &rows, &cols) == 2) {
            got_header = 1;
            break;
        }
    }

    if (!got_header || rows <= 0 || cols <= 0) {
        fprintf(stderr, "[ERROR]: Invalid or missing header in '%s'\n", filename);
        fclose(fp);
        return MAT_INVALID_PTR;
    }

    mat* m = mat_create(rows, cols);
    if (!m) { fclose(fp); return MAT_INVALID_PTR; }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float val;
            skip_delims(fp);
            if (fscanf(fp, " %f", &val) != 1) {
                fprintf(stderr, "[ERROR]: Not enough data at element (%d,%d)\n", i, j);
                mat_delete(m);
                fclose(fp);
                return MAT_INVALID_PTR;
            }
            m->mat_ptr[i][j] = val;
            skip_delims(fp);
        }
    }

    fclose(fp);
    return m;
}

mat_list mat_file_read_all(const char* filename) {
    mat_list out = { NULL, 0 };

    if (!filename) {
        fprintf(stderr, "[ERROR]: Null filename!\n");
        return out;
    }
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[ERROR]: Could not open file '%s'\n", filename);
        return out;
    }

    int capacity = 4;
    out.list = (mat**)malloc(sizeof(mat*) * capacity);
    if (!out.list) { fclose(fp); return out; }

    char line[4096];
    int rows, cols;

    while (fgets(line, sizeof(line), fp)) {
        int only_ws = 1;
        for (char* p = line; *p; ++p) {
            if (!isspace((unsigned char)*p)) { only_ws = 0; break; }
        }
        if (only_ws) continue;

        if (sscanf(line, " %d , %d", &rows, &cols) != 2 || rows <= 0 || cols <= 0) {
            fprintf(stderr, "[WARN]: Skipping invalid header line: %s", line);
            continue;
        }

        mat* m = mat_create(rows, cols);
        if (!m) continue;

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float val;
                if (fscanf(fp, " %f", &val) != 1) {
                    fprintf(stderr, "[ERROR]: Failed reading value (%d,%d)\n", i, j);
                    mat_delete(m);
                    fclose(fp);
                    return out;
                }
                m->mat_ptr[i][j] = val;

                int ch;
                while ((ch = fgetc(fp)) != EOF) {
                    if (ch == ',' || isspace((unsigned char)ch)) continue;
                    ungetc(ch, fp);
                    break;
                }
            }
        }

        if (out.count >= capacity) {
            capacity *= 2;
            mat** tmp = (mat**)realloc(out.list, sizeof(mat*) * capacity);
            if (!tmp) {
                fprintf(stderr, "[ERROR]: realloc failed\n");
                mat_delete(m);
                break;
            }
            out.list = tmp;
        }

        out.list[out.count++] = m;
    }

    fclose(fp);
    return out;
}

static void* mat_init_buffers(size_t n, size_t sz) {
    return calloc(n, sz);
}

static int mat_clamp_threads(int requested, int rows) {
    int t = requested;

    if (t <= 0) {
        int ncpu = 0;
#ifdef _OPENMP
        ncpu = omp_get_max_threads();
#else
    #if defined(_WIN32) || defined(_WIN64)
        SYSTEM_INFO si; GetSystemInfo(&si);
        ncpu = (int)si.dwNumberOfProcessors;
    #elif defined(_SC_NPROCESSORS_ONLN)
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        ncpu = (n > 0) ? (int)n : 0;
    #elif defined(_SC_NPROCESSORS_CONF)
        long n = sysconf(_SC_NPROCESSORS_CONF);
        ncpu = (n > 0) ? (int)n : 0;
    #else
        ncpu = MAT_MAX_THREADS;
    #endif
#endif
        if (ncpu <= 0) ncpu = MAT_MAX_THREADS;
        t = ncpu;
    }

    if (t > rows) t = rows;
    if (t < 1)    t = 1;
    return t;
}