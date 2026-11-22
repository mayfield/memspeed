#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include <time.h>
#include <assert.h>
#include <signal.h>
#include <pthread.h>
#include <fcntl.h>
#include <sys/param.h>
#include <sys/mman.h>
#ifdef __linux__
# include <sys/prctl.h>
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
# include <immintrin.h>
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
# include <arm_neon.h>
#endif


#define TB (1UL * 1024 * 1024 * 1024 * 1024)
#define GB (1UL * 1024 * 1024 * 1024)
#define MB (1UL * 1024 * 1024)


typedef void (*mem_write_test)(void *ptr, size_t size, size_t iter);

typedef struct thread_options {
    int id;
    mem_write_test test;
    void *mem;
    size_t offset;
    size_t size;
    size_t iterations;
    size_t *ready;
    size_t *done;
    pthread_cond_t *ready_cond;
    pthread_mutex_t *ready_mut;
    pthread_cond_t *start_cond;
    pthread_mutex_t *start_mut;
    pthread_cond_t *prog_cond;
    pthread_mutex_t *prog_mut;
} thread_options_t;

typedef struct draw_state {
    size_t ticks;
    size_t last_sz;
    double last_time;
} draw_state_t;

typedef struct cpus_topology {
    int *cpus;
    int count;
} cpus_topology_t;


static size_t g_page_size = 0;
static double g_start_time = -1;
static size_t g_transferred = 0;
static size_t g_thread_count = 1;
static bool g_verbose = false;
static int fd = -1;


#define ZERO_OR_EXIT(call) \
    do { \
        if ((call) != 0) { \
            fprintf(stderr, "Non-zero Call error: %s\n", strerror(errno)); \
            exit(1); \
        } \
    } while (0)


// A static buffer allocated on the stack is used for this..
// The returned pointer should not be shared, and only up to 10
// uses in a single function can be used concurrently per thread.
static char *human_size(size_t bytes) {
    static _Thread_local size_t buf_idx = 0;
    static _Thread_local char bufs[10][128];
    const size_t i = buf_idx;
    buf_idx = (buf_idx + 1) % 10;
    char *fmt;
    double v;
    if (bytes >= TB * 2) {
        fmt = (bytes % TB) ? "%.2f TB" : "%.0f TB";
        v = ((double) bytes) / TB;
    } else if (bytes >= GB * 2) {
        fmt = (bytes % GB) ? "%.2f GB" : "%.0f GB";
        v = ((double) bytes) / GB;
    } else if (bytes >= MB * 2) {
        fmt = (bytes % MB) ? "%.2f MB" : "%.0f MB";
        v = ((double) bytes) / MB;
    } else if (bytes >= 1024 * 2) {
        fmt = (bytes % 1024) ? "%.2f KB" : "%.0f KB";
        v = bytes / 1024.0;
    } else {
        fmt = "%.0f B";
        v = bytes;
    }
    snprintf(bufs[i], sizeof(bufs[i]), fmt, v);
    return bufs[i];
}


static double get_time() {
    struct timespec tspec;
    if (clock_gettime(CLOCK_MONOTONIC, &tspec) != 0) {
        fprintf(stderr, "Get time error");
        exit(1);
    }
    return tspec.tv_sec + tspec.tv_nsec / 1e9;
}


static uint64_t str_to_pos_u64(char* raw) {
    errno = 0;
    char *end;
    uint64_t num = strtoull(raw, &end, 10);
    if (errno) {
        fprintf(stderr, "Bad number: %s (%s)\n", raw, strerror(errno));
        exit(1);
    } else if (end == raw) {
        fprintf(stderr, "Not a number: %s\n", raw);
        exit(1);
    }
    return num;
}


#ifdef __x86_64__
static void mem_write_test_x86asm_nt(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    size_t len = size / sizeof(uint64_t);
    uint64_t *mem = ptr;
    __asm__ __volatile__(
        "movq %[mem], %%rdx\n\t"
        "movq %[len], %%rcx\n\t"
    "1:\n\t"
        "movnti %[v], (%%rdx)\n\t"
        "addq $8, %%rdx\n\t"
        "dec %%rcx\n\t"
        "jnz 1b\n\t"
        "sfence\n\t"
        :
        : [mem] "r" (mem),
          [len] "r" (len),
          [v] "r" (v)
        : "rcx", "rdx", "memory"
    );
}


static void mem_write_test_x86asm(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    size_t len = size / sizeof(uint64_t);
    uint64_t *mem = ptr;
    __asm__ __volatile__(
        "movq %[mem], %%rdx\n\t"
        "movq %[len], %%rcx\n\t"
    "1:\n\t"
        "movq %[v], (%%rdx)\n\t"
        "addq $8, %%rdx\n\t"
        "dec %%rcx\n\t"
        "jnz 1b\n\t"
        :
        : [mem] "r" (mem),
          [len] "r" (len),
          [v] "r" (v)
        : "rcx", "rdx", "memory"
    );
}


static void mem_write_test_x86asm_nt_x8(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    size_t len = size / sizeof(uint64_t) / 8;
    uint64_t *mem = ptr;
    __asm__ __volatile__(
        "movq %[mem], %%rdx\n\t"
        "movq %[len], %%rcx\n\t"
    "1:\n\t"
        "movnti %[v], (%%rdx)\n\t"
        "movnti %[v], 8(%%rdx)\n\t"
        "movnti %[v], 16(%%rdx)\n\t"
        "movnti %[v], 24(%%rdx)\n\t"
        "movnti %[v], 32(%%rdx)\n\t"
        "movnti %[v], 40(%%rdx)\n\t"
        "movnti %[v], 48(%%rdx)\n\t"
        "movnti %[v], 56(%%rdx)\n\t"
        "addq $64, %%rdx\n\t"
        "dec %%rcx\n\t"
        "jnz 1b\n\t"
        "sfence\n\t"
        :
        : [mem] "r" (mem),
          [len] "r" (len),
          [v] "r" (v)
        : "rcx", "rdx", "memory"
    );
}


static void mem_write_test_x86asm_x8(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    size_t len = size / sizeof(uint64_t) / 8;
    uint64_t *mem = ptr;
    __asm__ __volatile__(
        "movq %[mem], %%rdx\n\t"
        "movq %[len], %%rcx\n\t"
    "1:\n\t"
        "movq %[v], (%%rdx)\n\t"
        "movq %[v], 8(%%rdx)\n\t"
        "movq %[v], 16(%%rdx)\n\t"
        "movq %[v], 24(%%rdx)\n\t"
        "movq %[v], 32(%%rdx)\n\t"
        "movq %[v], 40(%%rdx)\n\t"
        "movq %[v], 48(%%rdx)\n\t"
        "movq %[v], 56(%%rdx)\n\t"
        "addq $64, %%rdx\n\t"
        "dec %%rcx\n\t"
        "jnz 1b\n\t"
        :
        : [mem] "r" (mem),
          [len] "r" (len),
          [v] "r" (v)
        : "rcx", "rdx", "memory"
    );
}


static void mem_write_test_x86asm_nt_x32(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    size_t len = size / sizeof(uint64_t) / 32;
    uint64_t *mem = ptr;
    __asm__ __volatile__(
        "movq %[mem], %%rdx\n\t"
        "movq %[len], %%rcx\n\t"
    "1:\n\t"
        "movnti %[v], (%%rdx)\n\t"
        "movnti %[v], 8(%%rdx)\n\t"
        "movnti %[v], 16(%%rdx)\n\t"
        "movnti %[v], 24(%%rdx)\n\t"
        "movnti %[v], 32(%%rdx)\n\t"
        "movnti %[v], 40(%%rdx)\n\t"
        "movnti %[v], 48(%%rdx)\n\t"
        "movnti %[v], 56(%%rdx)\n\t"
        "movnti %[v], 64(%%rdx)\n\t"
        "movnti %[v], 72(%%rdx)\n\t"
        "movnti %[v], 80(%%rdx)\n\t"
        "movnti %[v], 88(%%rdx)\n\t"
        "movnti %[v], 96(%%rdx)\n\t"
        "movnti %[v], 104(%%rdx)\n\t"
        "movnti %[v], 112(%%rdx)\n\t"
        "movnti %[v], 120(%%rdx)\n\t"
        "movnti %[v], 128(%%rdx)\n\t"
        "movnti %[v], 136(%%rdx)\n\t"
        "movnti %[v], 144(%%rdx)\n\t"
        "movnti %[v], 152(%%rdx)\n\t"
        "movnti %[v], 160(%%rdx)\n\t"
        "movnti %[v], 168(%%rdx)\n\t"
        "movnti %[v], 176(%%rdx)\n\t"
        "movnti %[v], 184(%%rdx)\n\t"
        "movnti %[v], 192(%%rdx)\n\t"
        "movnti %[v], 200(%%rdx)\n\t"
        "movnti %[v], 208(%%rdx)\n\t"
        "movnti %[v], 216(%%rdx)\n\t"
        "movnti %[v], 224(%%rdx)\n\t"
        "movnti %[v], 232(%%rdx)\n\t"
        "movnti %[v], 240(%%rdx)\n\t"
        "movnti %[v], 248(%%rdx)\n\t"
        "addq $256, %%rdx\n\t"
        "dec %%rcx\n\t"
        "jnz 1b\n\t"
        "sfence\n\t"
        :
        : [mem] "r" (mem),
          [len] "r" (len),
          [v] "r" (v)
        : "rcx", "rdx", "memory"
    );
}


static void mem_write_test_x86asm_x32(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    size_t len = size / sizeof(uint64_t) / 32;
    uint64_t *mem = ptr;
    __asm__ __volatile__(
        "movq %[mem], %%rdx\n\t"
        "movq %[len], %%rcx\n\t"
    "1:\n\t"
        "movq %[v], (%%rdx)\n\t"
        "movq %[v], 8(%%rdx)\n\t"
        "movq %[v], 16(%%rdx)\n\t"
        "movq %[v], 24(%%rdx)\n\t"
        "movq %[v], 32(%%rdx)\n\t"
        "movq %[v], 40(%%rdx)\n\t"
        "movq %[v], 48(%%rdx)\n\t"
        "movq %[v], 56(%%rdx)\n\t"
        "movq %[v], 64(%%rdx)\n\t"
        "movq %[v], 72(%%rdx)\n\t"
        "movq %[v], 80(%%rdx)\n\t"
        "movq %[v], 88(%%rdx)\n\t"
        "movq %[v], 96(%%rdx)\n\t"
        "movq %[v], 104(%%rdx)\n\t"
        "movq %[v], 112(%%rdx)\n\t"
        "movq %[v], 120(%%rdx)\n\t"
        "movq %[v], 128(%%rdx)\n\t"
        "movq %[v], 136(%%rdx)\n\t"
        "movq %[v], 144(%%rdx)\n\t"
        "movq %[v], 152(%%rdx)\n\t"
        "movq %[v], 160(%%rdx)\n\t"
        "movq %[v], 168(%%rdx)\n\t"
        "movq %[v], 176(%%rdx)\n\t"
        "movq %[v], 184(%%rdx)\n\t"
        "movq %[v], 192(%%rdx)\n\t"
        "movq %[v], 200(%%rdx)\n\t"
        "movq %[v], 208(%%rdx)\n\t"
        "movq %[v], 216(%%rdx)\n\t"
        "movq %[v], 224(%%rdx)\n\t"
        "movq %[v], 232(%%rdx)\n\t"
        "movq %[v], 240(%%rdx)\n\t"
        "movq %[v], 248(%%rdx)\n\t"
        "addq $256, %%rdx\n\t"
        "dec %%rcx\n\t"
        "jnz 1b\n\t"
        :
        : [mem] "r" (mem),
          [len] "r" (len),
          [v] "r" (v)
        : "rcx", "rdx", "memory"
    );
}
#endif  // x86_64


#ifdef __AVX2__
static void mem_write_test_avx2_nt(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    const __m256i vec = _mm256_set1_epi64x(v);
    __m256i *mem = ptr;
    for (size_t i = 0; i < size / sizeof(__m256i); i++) {
         _mm256_stream_si256(mem + i, vec);
    }
    _mm_sfence();
}


static void mem_write_test_avx2(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    const __m256i vec = _mm256_set1_epi64x(v);
    __m256i *mem = ptr;
    for (size_t i = 0; i < size / sizeof(__m256i); i++) {
         _mm256_store_si256(mem + i, vec);
    }
}


# ifdef __AVX512F__
static void mem_write_test_avx512_nt(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    const __m512i vec = _mm512_set1_epi64(v);
    __m512i *mem = ptr;
    for (size_t i = 0; i < size / sizeof(__m512i); i++) {
         _mm512_stream_si512(mem + i, vec);
    }
    _mm_sfence();
}


static void mem_write_test_avx512(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    const __m512i vec = _mm512_set1_epi64(v);
    __m512i *mem = ptr;
    for (size_t i = 0; i < size / sizeof(__m512i); i++) {
         _mm512_store_si512(mem + i, vec);
    }
}
# endif  // avx512
#endif  // avx2


#ifdef __aarch64__
static void mem_write_test_armasm(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    size_t len = size / sizeof(uint64_t);
    uint64_t *mem = ptr;
    __asm__ __volatile__(
        "mov x0, %[mem]\n\t"
        "mov x1, %[len]\n\t"
    "1: \n\t"
        "stp %[v], %[v], [x0]\n\t"
        "add x0, x0, #16\n\t"
        "subs x1, x1, #2\n\t"
        "b.gt 1b\n\t"
        :
        : [mem] "r" (mem),
          [len] "r" (len),
          [v] "r" (v)
        : "x0", "x1", "memory"
    );
}


static void mem_write_test_armasm_x8(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    size_t len = size / sizeof(uint64_t);
    uint64_t *mem = ptr;
    __asm__ __volatile__(
        "mov x0, %[mem]\n\t"
        "mov x1, %[len]\n\t"
    "1: \n\t"
        "stp %[v], %[v], [x0]\n\t"
        "stp %[v], %[v], [x0, #16]\n\t"
        "stp %[v], %[v], [x0, #32]\n\t"
        "stp %[v], %[v], [x0, #48]\n\t"
        "stp %[v], %[v], [x0, #64]\n\t"
        "stp %[v], %[v], [x0, #80]\n\t"
        "stp %[v], %[v], [x0, #96]\n\t"
        "stp %[v], %[v], [x0, #112]\n\t"
        "add x0, x0, #128\n\t"
        "subs x1, x1, #16\n\t"
        "b.gt 1b\n\t"
        :
        : [mem] "r" (mem),
          [len] "r" (len),
          [v] "r" (v)
        : "x0", "x1", "memory"
    );
}


static void mem_write_test_armasm_nt(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    size_t len = size / sizeof(uint64_t);
    uint64_t *mem = ptr;
    __asm__ __volatile__(
        "mov x0, %[mem]\n\t"
        "mov x1, %[len]\n\t"
    "1: \n\t"
        "stnp %[v], %[v], [x0]\n\t"
        "add x0, x0, #16\n\t"
        "subs x1, x1, #2\n\t"
        "b.gt 1b\n\t"
        "dsb ishst\n\t"
        :
        : [mem] "r" (mem),
          [len] "r" (len),
          [v] "r" (v)
        : "x0", "x1", "memory"
    );
}


static void mem_write_test_armasm_nt_x8(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    size_t len = size / sizeof(uint64_t);
    uint64_t *mem = ptr;
    __asm__ __volatile__(
        "mov x0, %[mem]\n\t"
        "mov x1, %[len]\n\t"
    "1: \n\t"
        "stnp %[v], %[v], [x0]\n\t"
        "stnp %[v], %[v], [x0, #16]\n\t"
        "stnp %[v], %[v], [x0, #32]\n\t"
        "stnp %[v], %[v], [x0, #48]\n\t"
        "stnp %[v], %[v], [x0, #64]\n\t"
        "stnp %[v], %[v], [x0, #80]\n\t"
        "stnp %[v], %[v], [x0, #96]\n\t"
        "stnp %[v], %[v], [x0, #112]\n\t"
        "add x0, x0, #128\n\t"
        "subs x1, x1, #16\n\t"
        "b.gt 1b\n\t"
        "dsb ishst\n\t"
        :
        : [mem] "r" (mem),
          [len] "r" (len),
          [v] "r" (v)
        : "x0", "x1", "memory"
    );
}
#endif  // aarch64


#ifdef __ARM_NEON
void mem_write_test_armneon(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    uint64x2_t vec = vdupq_n_u64(v);
    uint64x2_t *mem = ptr;
    for (size_t i = 0; i < size / sizeof(uint64x2_t); i++) {
        vst1q_u64((uint64_t*) (mem + i), vec);
    }
}
#endif  // arm_neon


static void mem_write_test_c(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    uint64_t *mem = ptr;
    for (size_t i = 0; i < size / sizeof(uint64_t); i++) {
        mem[i] = v;
    }
}


static void mem_write_test_c_x8(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    uint64_t *mem = ptr;
    for (size_t i = 0; i < size / sizeof(uint64_t); i += 8) {
        mem[i] = v;
        mem[i + 1] = v;
        mem[i + 2] = v;
        mem[i + 3] = v;
        mem[i + 4] = v;
        mem[i + 5] = v;
        mem[i + 6] = v;
        mem[i + 7] = v;
    }
}


static void mem_write_test_c_x32(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    uint64_t *m = ptr;
    for (size_t i = 0; i < size / sizeof(uint64_t); i += 32) {
        m[i]    = v; m[i+1]  = v; m[i+2]  = v; m[i+3]  = v; m[i+4]  = v; m[i+5]  = v; m[i+6]  = v; m[i+7]  = v;
        m[i+8]  = v; m[i+9]  = v; m[i+10] = v; m[i+11] = v; m[i+12] = v; m[i+13] = v; m[i+14] = v; m[i+15] = v;
        m[i+16] = v; m[i+17] = v; m[i+18] = v; m[i+19] = v; m[i+20] = v; m[i+21] = v; m[i+22] = v; m[i+23] = v;
        m[i+24] = v; m[i+25] = v; m[i+26] = v; m[i+27] = v; m[i+28] = v; m[i+29] = v; m[i+30] = v; m[i+31] = v;
    }
}


static void mem_write_test_c_x128(void *ptr, size_t size, size_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    uint64_t *m = ptr;
    for (size_t i = 0; i < size / sizeof(uint64_t); i += 128) {
        m[i]     = v; m[i+1]   = v; m[i+2]   = v; m[i+3]   = v; m[i+4]   = v; m[i+5]   = v; m[i+6]   = v; m[i+7]  =  v;
        m[i+8]   = v; m[i+9]   = v; m[i+10]  = v; m[i+11]  = v; m[i+12]  = v; m[i+13]  = v; m[i+14]  = v; m[i+15] =  v;
        m[i+16]  = v; m[i+17]  = v; m[i+18]  = v; m[i+19]  = v; m[i+20]  = v; m[i+21]  = v; m[i+22]  = v; m[i+23] =  v;
        m[i+24]  = v; m[i+25]  = v; m[i+26]  = v; m[i+27]  = v; m[i+28]  = v; m[i+29]  = v; m[i+30]  = v; m[i+31] =  v;
        m[i+32]  = v; m[i+33]  = v; m[i+34]  = v; m[i+35]  = v; m[i+36]  = v; m[i+37]  = v; m[i+38]  = v; m[i+39] =  v;
        m[i+40]  = v; m[i+41]  = v; m[i+42]  = v; m[i+43]  = v; m[i+44]  = v; m[i+45]  = v; m[i+46]  = v; m[i+47] =  v;
        m[i+48]  = v; m[i+49]  = v; m[i+50]  = v; m[i+51]  = v; m[i+52]  = v; m[i+53]  = v; m[i+54]  = v; m[i+55] =  v;
        m[i+56]  = v; m[i+57]  = v; m[i+58]  = v; m[i+59]  = v; m[i+60]  = v; m[i+61]  = v; m[i+62]  = v; m[i+63] =  v;
        m[i+64]  = v; m[i+65]  = v; m[i+66]  = v; m[i+67]  = v; m[i+68]  = v; m[i+69]  = v; m[i+70]  = v; m[i+71] =  v;
        m[i+72]  = v; m[i+73]  = v; m[i+74]  = v; m[i+75]  = v; m[i+76]  = v; m[i+77]  = v; m[i+78]  = v; m[i+79] =  v;
        m[i+80]  = v; m[i+81]  = v; m[i+82]  = v; m[i+83]  = v; m[i+84]  = v; m[i+85]  = v; m[i+86]  = v; m[i+87] =  v;
        m[i+88]  = v; m[i+89]  = v; m[i+90]  = v; m[i+91]  = v; m[i+92]  = v; m[i+93]  = v; m[i+94]  = v; m[i+95] =  v;
        m[i+96]  = v; m[i+97]  = v; m[i+98]  = v; m[i+99]  = v; m[i+100] = v; m[i+101] = v; m[i+102] = v; m[i+103] = v;
        m[i+104] = v; m[i+105] = v; m[i+106] = v; m[i+107] = v; m[i+108] = v; m[i+109] = v; m[i+110] = v; m[i+111] = v;
        m[i+112] = v; m[i+113] = v; m[i+114] = v; m[i+115] = v; m[i+116] = v; m[i+117] = v; m[i+118] = v; m[i+119] = v;
        m[i+120] = v; m[i+121] = v; m[i+122] = v; m[i+123] = v; m[i+124] = v; m[i+125] = v; m[i+126] = v; m[i+127] = v;
    }
}


static void mem_write_test_memset(void *ptr, size_t size, size_t iter) {
    const char b = iter % 0xff;
    memset(ptr, b, size);
}


static void mem_write_test_memcpy(void *ptr, size_t size, size_t iter) {
    char * restrict scratch = aligned_alloc(g_page_size, g_page_size);
    if (scratch == NULL) {
        fprintf(stderr, "Mem alloc failed %s\n", strerror(errno));
        exit(1);
    }
    const char b = iter % 0xff;
    memset(scratch, b, g_page_size);
    char *mem = ptr;
    for (size_t i = 0; i < size; i += g_page_size) {
        memcpy(mem + i, scratch, g_page_size);
    }
    free(scratch);
}


#ifdef __linux__
static cpus_topology_t * get_cpus_topology() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    sched_getaffinity(0, sizeof(cpuset), &cpuset); // has ambiguous return value
    int count = CPU_COUNT(&cpuset);
    if (count < 1 || count > 0xffff) {
        errno = ENOENT;
        return NULL;
    }
    cpus_topology_t *topo = malloc(sizeof(cpus_topology_t));
    if (topo == NULL) {
        return NULL;
    }
    topo->cpus = calloc(count, sizeof(topo->cpus[0]));
    if (topo->cpus == NULL) {
        free(topo);
        return NULL;
    }
    topo->count = count;
    for (int cpu = 0, i = 0; cpu < CPU_SETSIZE; cpu++) {
        if (CPU_ISSET(cpu, &cpuset)) {
            topo->cpus[i++] = cpu;
        }
    }
    return topo;
}
#endif


static void maybe_draw_progress(draw_state_t *state, bool force) {
    bool sizable = false;
    for (; state->ticks * GB < g_transferred; state->ticks++) {
        sizable = true;
    }
    if (sizable || force) {
        double t = get_time();
        double elapsed = t - state->last_time;
        if (elapsed > 0.200 || force) {
            printf("\r%80s\r", "");
            printf("\rCurrent: %10s/s  |  Avg: %10s/s  |  Transferred: %s",
                human_size((g_transferred - state->last_sz) / (t - state->last_time)),
                human_size(g_transferred / (t - g_start_time)),
                human_size(g_transferred));
            fflush(stdout);
            state->last_sz = g_transferred;
            state->last_time = t;
        }
    }
}


static void* threaded_test_runner(void *_options) {
    thread_options_t *options = _options;
#ifdef __linux__
    char name[128];
    snprintf(name, sizeof(name), "memspeed-%03d", options->id);
    ZERO_OR_EXIT(prctl(PR_SET_NAME, name));
#endif
    ZERO_OR_EXIT(pthread_mutex_lock(options->ready_mut));
    (*options->ready)++;
    ZERO_OR_EXIT(pthread_cond_signal(options->ready_cond));
    ZERO_OR_EXIT(pthread_mutex_unlock(options->ready_mut));

    ZERO_OR_EXIT(pthread_mutex_lock(options->start_mut));
    ZERO_OR_EXIT(pthread_cond_wait(options->start_cond, options->start_mut));
    ZERO_OR_EXIT(pthread_mutex_unlock(options->start_mut));
    void *mem = (fd == -1) ? options->mem + options->offset : NULL;
    for (size_t iter = 1; iter <= options->iterations; iter++) {
        if (fd != -1) {
            // Remapping required with btrfs to workaround 50% perf issue
            mem = mmap(NULL, options->size, PROT_WRITE, MAP_SHARED, fd, options->offset);
            if (options->mem == MAP_FAILED) {
                fprintf(stderr, "Mem map failed %s\n", strerror(errno));
                exit(1);
            }
        }
        options->test(mem, options->size, iter);
        if (fd != -1) {
            if (msync(mem, options->size, MS_SYNC) == -1) {
                fprintf(stderr, "Mem sync to fd failed %s\n", strerror(errno));
                exit(1);
            }
            if (munmap(mem, options->size) == -1) {
                fprintf(stderr, "Mem unmap failed %s\n", strerror(errno));
                exit(1);
            }
        }
        ZERO_OR_EXIT(pthread_mutex_lock(options->prog_mut));
        g_transferred += options->size;
        if (iter == options->iterations) {
            (*options->done)++;
        }
        ZERO_OR_EXIT(pthread_cond_signal(options->prog_cond));
        ZERO_OR_EXIT(pthread_mutex_unlock(options->prog_mut));
    }
    return NULL;
}


static void bench_threaded(void *mem, size_t buffer_size, size_t total_size, mem_write_test test) {
    pthread_t *threads = calloc(g_thread_count, sizeof(pthread_t));
    if (threads == NULL) {
        fprintf(stderr, "Mem alloc failed %s\n", strerror(errno));
        exit(1);
    }

    pthread_cond_t ready_cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t ready_mut = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t start_cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t start_mut = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t prog_cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t prog_mut = PTHREAD_MUTEX_INITIALIZER;
    size_t ready = 0;
    size_t done = 0;

#ifdef __linux__
    cpus_topology_t *cpus_topo = get_cpus_topology();
    if (cpus_topo == NULL) {
        fprintf(stderr, "Failed to get CPU topology: %s", strerror(errno));
        exit(1);
    }
    if (g_verbose) {
        int last_cpu = cpus_topo->cpus[0];
        printf("Available CPU cores: %d", last_cpu);
        bool spanning = false;
        for (int i = 1; i < cpus_topo->count; i++) {
            int cpu = cpus_topo->cpus[i];
            if (last_cpu == cpu - 1) {
                last_cpu = cpu;
                spanning = true;
                continue;
            }
            if (spanning) {
                printf(" -> %d", last_cpu);
            }
            printf(", %d", cpu);
            spanning = false;
            last_cpu = cpu;
        }
        if (spanning) {
            printf(" -> %d", cpus_topo->cpus[cpus_topo->count - 1]);
        }
        printf("\n");
    }
#endif

    for (size_t i = 0; i < g_thread_count; i++) {
        thread_options_t *options = malloc(sizeof(thread_options_t));
        if (options == NULL) {
            fprintf(stderr, "Mem alloc failed %s\n", strerror(errno));
            exit(1);
        }
        size_t shard_size = buffer_size / g_thread_count;
        options->id = i;
        options->test = test;
        options->mem = mem;
        options->offset = shard_size * i;

        options->size = shard_size;
        options->ready = &ready;
        options->done = &done;
        options->ready_cond = &ready_cond;
        options->ready_mut = &ready_mut;
        options->start_cond = &start_cond;
        options->start_mut = &start_mut;
        options->prog_cond = &prog_cond;
        options->prog_mut = &prog_mut;
        options->iterations = total_size / buffer_size;
        ZERO_OR_EXIT(pthread_create(&threads[i], NULL, threaded_test_runner, options));
#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        int cpu = cpus_topo->cpus[options->id % cpus_topo->count];
        if (g_verbose) {
            printf("Thread %d mapped to CPU core: %d\n", options->id, cpu);
        }
        CPU_SET(cpu, &cpuset);
        ZERO_OR_EXIT(pthread_setaffinity_np(threads[i], sizeof(cpuset), &cpuset));
#endif
    }

    ZERO_OR_EXIT(pthread_mutex_lock(&ready_mut));
    while (ready < g_thread_count) {
        ZERO_OR_EXIT(pthread_cond_wait(&ready_cond, &ready_mut));
    }
    ZERO_OR_EXIT(pthread_mutex_unlock(&ready_mut));

    ZERO_OR_EXIT(pthread_mutex_lock(&start_mut));
    ZERO_OR_EXIT(pthread_cond_broadcast(&start_cond));
    ZERO_OR_EXIT(pthread_mutex_unlock(&start_mut));
    g_start_time = get_time();
    draw_state_t draw_state = {.last_time = g_start_time};

    ZERO_OR_EXIT(pthread_mutex_lock(&prog_mut));
    while (done < g_thread_count) {
        ZERO_OR_EXIT(pthread_cond_wait(&prog_cond, &prog_mut));
        maybe_draw_progress(&draw_state, /*force*/ false);
    }
    maybe_draw_progress(&draw_state, /*force*/ true);
    ZERO_OR_EXIT(pthread_mutex_unlock(&prog_mut));
    printf("\n");

    for (size_t i = 0; i < g_thread_count; i++) {
        pthread_join(threads[i], NULL);
    }
}


static void bench(void *mem, size_t buffer_size, size_t total_size, mem_write_test test) {
    if (buffer_size < 1 || total_size < buffer_size) {
        fprintf(stderr, "Invalid bench args\n");
        exit(1);
    }
    g_start_time = get_time();
    draw_state_t draw_state = {.last_time = g_start_time};
    for (size_t iter = 1; iter <= total_size / buffer_size; iter++) {
        if (fd != -1) {
            // Remapping required with btrfs to workaround 50% perf issue
            mem = mmap(NULL, buffer_size, PROT_WRITE, MAP_SHARED, fd, 0);
            if (mem == MAP_FAILED) {
                fprintf(stderr, "Mem map failed %s\n", strerror(errno));
                exit(1);
            }
        }
        test(mem, buffer_size, iter);
        if (fd != -1) {
            if (msync(mem, buffer_size, MS_SYNC) == -1) {
                fprintf(stderr, "Mem sync to fd failed %s\n", strerror(errno));
                exit(1);
            }
            if (munmap(mem, buffer_size) == -1) {
                fprintf(stderr, "Mem unmap failed %s\n", strerror(errno));
                exit(1);
            }
        }
        g_transferred += buffer_size;
        maybe_draw_progress(&draw_state, /*force*/ false);
    }
    printf("\n");
}


static void print_results(double time) {
    printf("Transferred: %s\n", human_size(g_transferred));
    printf("Time: %.3f s\n", time);
    printf("Speed: %s/s\n", human_size(g_transferred / time));
}


static void on_interrupted(int _) {
    (void) _;
    double end_time = get_time();
    printf("\n\nINTERRUPTED\n\n");
    if (g_start_time != -1) {
        print_results(end_time - g_start_time);
    }
    exit(1);
}


int main(int argc, char *argv[]) {
    g_page_size = sysconf(_SC_PAGESIZE);
    size_t buffer_size_mb = 4 * 1024;
    size_t total_size_mb = 1024 * 1024;
    char *strategy = "c";
    int use_mmap = 0;
    int use_mmap_disk = 0;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--strat", 7) == 0) {
            if (argc < i + 2) {
                fprintf(stderr, "Expected STRATEGY argument\n");
                exit(1);
            }
            strategy = argv[++i];
        } else if (strncmp(argv[i], "--total", 7) == 0) {
            if (argc < i + 2) {
                fprintf(stderr, "Expected TOTAL_SIZE_GB argument\n");
                exit(1);
            }
            total_size_mb = str_to_pos_u64(argv[++i]);
        } else if (strncmp(argv[i], "--threads", 9) == 0) {
            if (argc < i + 2) {
                fprintf(stderr, "Expected THREAD_COUNT argument\n");
                exit(1);
            }
            g_thread_count = str_to_pos_u64(argv[++i]);
            if (g_thread_count < 1 || g_thread_count > 1000) {
                fprintf(stderr, "Invalid THREAD_COUNT: %ld\n", g_thread_count);
                exit(1);
            }
        } else if (strcmp(argv[i], "--mmap") == 0) {
            use_mmap = 1;
        } else if (strcmp(argv[i], "--mmap-disk") == 0) {
            use_mmap_disk = 1;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            g_verbose = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            char pad[128] = {0};
            memset(pad, 0, sizeof(pad));
            memset(pad, ' ', MIN(sizeof(pad) - 1, strlen(argv[0])));
            fprintf(stderr, "Usage: %s [--strat[egy] STRATEGY]\n", argv[0]);
            fprintf(stderr, "       %s [--mmap]\n", pad);
            fprintf(stderr, "       %s [--mmap-disk]\n", pad);
            fprintf(stderr, "       %s [--verbose]\n", pad);
            fprintf(stderr, "       %s [--total] TOTAL_SIZE_MB]\n", pad);
            fprintf(stderr, "       %s [--threads THREAD_COUNT]\n", pad);
            fprintf(stderr, "       %s BUFFER_SIZE_MB\n", pad);
            fprintf(stderr, "\n");
            fprintf(stderr, "    STRATEGY:\n");
            fprintf(stderr, "        c               : A C loop subject to compiler optimizations\n");
            fprintf(stderr, "        c_x8            : A C loop with 8 x 64bit writes\n");
            fprintf(stderr, "        c_x32           : A C loop with 32 x 64bit writes\n");
            fprintf(stderr, "        c_x128          : A C loop with 128 x 64bit writes\n");
            fprintf(stderr, "        memset          : Byte by byte memset() in a loop\n");
            fprintf(stderr, "        memcpy          : Aligned page memcpy in a loop\n");
#ifdef __x86_64__
            fprintf(stderr, "        x86asm          : 64bit x86 ASM\n");
            fprintf(stderr, "        x86asm_nt       : 64bit x86 ASM (non-temporal)\n");
            fprintf(stderr, "        x86asm_x8       : 8 x 64bit x86 ASM\n");
            fprintf(stderr, "        x86asm_nt_x8    : 8 x 64bit x86 ASM (non-temporal)\n");
            fprintf(stderr, "        x86asm_x32      : 32 x 64bit x86 ASM\n");
            fprintf(stderr, "        x86asm_nt_x32   : 32 x 64bit x86 ASM (non-temporal)\n");
#endif
#ifdef __AVX2__
            fprintf(stderr, "        avx2            : 256bit AVX2 intrinsics\n");
            fprintf(stderr, "        avx2_nt         : 256bit AVX2 intrinsics (non-temporal)\n");
# ifdef __AVX512F__
            fprintf(stderr, "        avx512          : 512bit AVX512 intrinsics\n");
            fprintf(stderr, "        avx512_nt       : 512bit AVX512 intrinsics (non-temporal)\n");
# endif
#endif
#ifdef __aarch64__
            fprintf(stderr, "        armasm          : 128bit ARM ASM (STP)\n");
            fprintf(stderr, "        armasm_nt       : 128bit ARM ASM (non-temporal, STNP)\n");
            fprintf(stderr, "        armasm_x8       : 8 x 128bit ARM ASM (STP)\n");
            fprintf(stderr, "        armasm_nt_x8    : 8 x 128bit ARM ASM (non-temporal, SSTP)\n");
# ifdef __ARM_NEON
            fprintf(stderr, "        armneon         : 128bit ARM NEON SIMD intrinsics\n");
# endif
#endif
            fprintf(stderr, "\n");
            fprintf(stderr, "    TOTAL_SIZE_MB: Total amount to transfer through memory in MB\n");
            exit(0);
        } else {
            buffer_size_mb = str_to_pos_u64(argv[i]);
        }
    }
    size_t buffer_size = buffer_size_mb * MB;
    if (!buffer_size || (buffer_size % (g_page_size * g_thread_count))) {
        size_t div = g_page_size * g_thread_count;
        buffer_size = buffer_size > div ?
            (buffer_size / div) * div :
            div;
        fprintf(stderr, "WARNING: Adjusting BUFFER_SIZE_MB: %s\n", human_size(buffer_size));
    }
    size_t shard_size = buffer_size / g_thread_count;
    size_t total_size = total_size_mb * MB;
    if (total_size % buffer_size) {
        total_size = total_size > buffer_size ?
            (total_size / buffer_size) * buffer_size :
            buffer_size;
        fprintf(stderr, "NOTE: Adjusting TOTAL_SIZE_MB: %s\n", human_size(total_size));
    }
    assert(buffer_size && buffer_size % g_page_size == 0);
    assert(shard_size && shard_size % g_page_size == 0);
    assert(total_size && total_size % g_page_size == 0);
    mem_write_test test;
    if (strcmp(strategy, "c") == 0) {
        test = mem_write_test_c;
    } else if (strcmp(strategy, "c_x8") == 0) {
        test = mem_write_test_c_x8;
    } else if (strcmp(strategy, "c_x32") == 0) {
        test = mem_write_test_c_x32;
    } else if (strcmp(strategy, "c_x128") == 0) {
        test = mem_write_test_c_x128;
    } else if (strcmp(strategy, "memset") == 0) {
        test = mem_write_test_memset;
    } else if (strcmp(strategy, "memcpy") == 0) {
        test = mem_write_test_memcpy;
#ifdef __x86_64__
    } else if (strcmp(strategy, "x86asm") == 0) {
        test = mem_write_test_x86asm;
    } else if (strcmp(strategy, "x86asm_nt") == 0) {
        test = mem_write_test_x86asm_nt;
    } else if (strcmp(strategy, "x86asm_x8") == 0) {
        test = mem_write_test_x86asm_x8;
    } else if (strcmp(strategy, "x86asm_nt_x8") == 0) {
        test = mem_write_test_x86asm_nt_x8;
    } else if (strcmp(strategy, "x86asm_x32") == 0) {
        test = mem_write_test_x86asm_x32;
    } else if (strcmp(strategy, "x86asm_nt_x32") == 0) {
        test = mem_write_test_x86asm_nt_x32;
#endif
#ifdef __AVX2__
    } else if (strcmp(strategy, "avx2") == 0) {
        test = mem_write_test_avx2;
    } else if (strcmp(strategy, "avx2_nt") == 0) {
        test = mem_write_test_avx2_nt;
# ifdef __AVX512F__
    } else if (strcmp(strategy, "avx512") == 0) {
        test = mem_write_test_avx512;
    } else if (strcmp(strategy, "avx512_nt") == 0) {
        test = mem_write_test_avx512_nt;
# endif
#endif
#ifdef __aarch64__
    } else if (strcmp(strategy, "armasm") == 0) {
        test = mem_write_test_armasm;
    } else if (strcmp(strategy, "armasm_x8") == 0) {
        test = mem_write_test_armasm_x8;
    } else if (strcmp(strategy, "armasm_nt") == 0) {
        test = mem_write_test_armasm_nt;
    } else if (strcmp(strategy, "armasm_nt_x8") == 0) {
        test = mem_write_test_armasm_nt_x8;
# ifdef __ARM_NEON
    } else if (strcmp(strategy, "armneon") == 0) {
        test = mem_write_test_armneon;
# endif
#endif
    } else {
        fprintf(stderr, "Invalid test strategy\n");
        exit(1);
    }
    printf("Strategy: %s\n", strategy);
    printf("Page size: %s\n", human_size(g_page_size));
    printf("Transfer size: %s\n", human_size(total_size));
    if (g_thread_count > 1) {
        printf("Threads: %ld\n", g_thread_count);
        printf("Thread shard: %s\n", human_size(buffer_size / g_thread_count));
    }
    printf("Allocating buffer [%s]: %s\n", use_mmap_disk ? "mmap_disk" : use_mmap ? "mmap" : "malloc",
           human_size(buffer_size));
    void *mem;
    if (use_mmap_disk != 0) {
        fd = open("./memspeed.test", O_CREAT|O_DIRECT|O_RDWR, 0640);
        if (fd == -1) {
            fprintf(stderr, "Failed to open/create memspeed.test file: %s\n", strerror(errno));
        }
        if (fallocate(fd, /*flags*/ 0, /*offt*/ 0, buffer_size) == -1) {
            fprintf(stderr, "Failed to allocate memspeed.test file: %s\n", strerror(errno));
        }
        posix_fadvise64(fd, 0, buffer_size, POSIX_FADV_SEQUENTIAL);
        mem = mmap(NULL, buffer_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        if (mem == MAP_FAILED) {
            fprintf(stderr, "Mem alloc failed %s\n", strerror(errno));
            exit(1);
        }
    } else if (use_mmap != 0) {
        mem = mmap(NULL, buffer_size, PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        if (mem == MAP_FAILED) {
            fprintf(stderr, "Mem alloc failed %s\n", strerror(errno));
            exit(1);
        }
    } else {
        mem = aligned_alloc(g_page_size, buffer_size);
        if (mem == NULL) {
            fprintf(stderr, "Mem alloc failed %s\n", strerror(errno));
            exit(1);
        }
    }
    // NOTE: Memset can get optimized out, must write by hand...
    srand((unsigned int) time(NULL));
    printf("Prefaulting buffer:          ");
    fflush(stdout);
    for (size_t i = 0; i < buffer_size / sizeof(int); i++) {
        ((int*) mem)[i] = rand();
        if (i % (100 * MB) == 0) {
            printf("\rPrefaulting buffer: %8x ", ((int*) mem)[i]);
            fflush(stdout);
        }
    }
    for (size_t i = 0; i < buffer_size / sizeof(int); i++) {
        (void) ((int*) mem)[i];
    }
    if (use_mmap_disk != 0) {
        if (msync(mem, buffer_size, MS_SYNC) == -1) {
            fprintf(stderr, "Mem sync to fd failed %s\n", strerror(errno));
            exit(1);
        }
        if (munmap(mem, buffer_size) == -1) {
            fprintf(stderr, "Mem unmap failed %s\n", strerror(errno));
            exit(1);
        }
        mem = NULL;
    }
    printf("\n");

    usleep(100000);

    printf("Running test...\n");
    signal(SIGINT, on_interrupted);
    if (g_thread_count > 1) {
        bench_threaded(mem, buffer_size, total_size, test);
    } else {
        bench(mem, buffer_size, total_size, test);
    }
    double end_time = get_time();

    if (use_mmap != 0) {
        if (munmap((void*) mem, buffer_size) == -1) {
            fprintf(stderr, "Mem unmap failed %s\n", strerror(errno));
            exit(1);
        }
    } else if (use_mmap_disk != 0) {
        if (close(fd) == -1) {
            fprintf(stderr, "File close %s\n", strerror(errno));
            exit(1);
        }
    } else {
        free((void*) mem);
    }
    printf("\nCOMPLETED\n\n");
    print_results(end_time - g_start_time);
    return 0;
}
