#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <signal.h>
#include <sys/param.h>
#include <sys/mman.h>
#ifndef __APPLE__
# include <immintrin.h>
#else
# include <arm_neon.h>
#endif

#define GB (1L * 1024 * 1024 * 1024)
#define MB (1L * 1024 * 1024)

static size_t page_size = 0;
struct timespec _ts;
typedef void (*mem_write_test)(void *ptr, size_t size, uint64_t iter);

static double start_time = 0;
static size_t transferred = 0;


static double get_time() {
    if (clock_gettime(CLOCK_MONOTONIC, &_ts) != 0) {
        fprintf(stderr, "Get time error");
        return -1;
    }
    return _ts.tv_sec + _ts.tv_nsec / 1e9;
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


static void *alloc(size_t size, int use_mmap) {
    void *ptr;
    if (use_mmap != 0) {
        printf("Using MMAP (SHARED)\n");
        ptr = mmap(NULL, size, PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            fprintf(stderr, "Mem alloc failed %s\n", strerror(errno));
            exit(1);
        }
    } else {
        printf("Using MALLOC\n");
        ptr = aligned_alloc(page_size, size);
        if (ptr == NULL) {
            fprintf(stderr, "Mem alloc failed %s\n", strerror(errno));
            exit(1);
        }
    }
    return ptr;
}


static void dealloc(void *ptr, size_t size, int use_mmap) {
    if (use_mmap != 0) {
        munmap((void*) ptr, size);
    } else {
        free((void*) ptr);
    }
}


#ifndef __APPLE__
static void mem_write_test_x86asm(void *ptr, size_t size, uint64_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (int i = 0; i < sizeof(uint64_t); i++) {
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


static void mem_write_test_x86asm_unrolled_x8(void *ptr, size_t size, uint64_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (int i = 0; i < sizeof(uint64_t); i++) {
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


static void mem_write_test_x86asm_unrolled_x32(void *ptr, size_t size, uint64_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (int i = 0; i < sizeof(uint64_t); i++) {
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


static void mem_write_test_avx2(void *ptr, size_t size, uint64_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (int i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    const __m256i vec = _mm256_set1_epi64x(v);
    __m256i *mem = ptr;
    for (size_t i = 0; i < size / sizeof(__m256i); i += 4) {
         _mm256_stream_si256(mem + i, vec);
         _mm256_stream_si256(mem + i + 1, vec);
         _mm256_stream_si256(mem + i + 2, vec);
         _mm256_stream_si256(mem + i + 3, vec);
    }
    _mm_sfence();
}


static void mem_write_test_avx512(void *ptr, size_t size, uint64_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (int i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    const __m512i vec = _mm512_set1_epi64(v);
    __m512i *mem = ptr;
    for (size_t i = 0; i < size / sizeof(__m512i); i += 4) {
         _mm512_stream_si512(mem + i, vec);
         _mm512_stream_si512(mem + i + 1, vec);
         _mm512_stream_si512(mem + i + 2, vec);
         _mm512_stream_si512(mem + i + 3, vec);
    }
    _mm_sfence();
}

#else

void mem_write_test_armneon(void *ptr, size_t size, uint64_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (int i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    uint64x2_t vec = vdupq_n_u64(v);
    uint64x2_t *mem = ptr;
    for (size_t i = 0; i < size / sizeof(uint64x2_t); i += 4) {
        vst1q_u64((uint64_t*)(mem + i), vec);
        vst1q_u64((uint64_t*)(mem + i + 1), vec);
        vst1q_u64((uint64_t*)(mem + i + 2), vec);
        vst1q_u64((uint64_t*)(mem + i + 3), vec);
    }
}
#endif


static void mem_write_test_c_loop(void *ptr, size_t size, uint64_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (int i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    uint64_t *mem = ptr;
    for (size_t i = 0; i < size / sizeof(uint64_t); i++) {
        mem[i] = v;
    }
}


static void mem_write_test_c_loop_unrolled_x8(void *ptr, size_t size, uint64_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (int i = 0; i < sizeof(uint64_t); i++) {
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


static void mem_write_test_c_loop_unrolled_x32(void *ptr, size_t size, uint64_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (int i = 0; i < sizeof(uint64_t); i++) {
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


static void mem_write_test_c_loop_unrolled_x128(void *ptr, size_t size, uint64_t iter) {
    const uint64_t b = iter % 0xff;
    uint64_t v = 0;
    for (int i = 0; i < sizeof(uint64_t); i++) {
        v = (v << 8) | b;
    }
    uint64_t *m = ptr;
    for (size_t i = 0; i < size / sizeof(uint64_t);) {
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
        m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v; m[i++] = v;
    }
}


static void mem_write_test_memset(void *ptr, size_t size, uint64_t iter) {
    const char b = iter % 0xff;
    memset(ptr, b, size);
}


static void mem_write_test_memcpy(void *ptr, size_t size, uint64_t iter) {
    const char b = iter % 0xff;
    char buf[8];
    for (int i = 0; i < sizeof(buf); i++) {
        buf[i] = b;
    }
    for (size_t i = 0; i < size; i += sizeof(buf)) {
        memcpy((char*) ptr + i, buf, sizeof(buf));
    }
}


static void bench(void *mem, size_t buffer_size, size_t transfer_size, mem_write_test test) {
    const size_t iterations = transfer_size / buffer_size;
    uint64_t iter;
    int gb_count = 0;
    size_t last_draw_sz = 0;
    start_time = get_time();
    double last_draw_time = start_time;
    for (iter = 1, transferred = 0; iter <= iterations; iter++) {
        test(mem, buffer_size, iter);
        transferred += buffer_size;
        if (0) {
            for (; gb_count * GB < transferred; gb_count++) {
                printf(".");
                fflush(stdout);
            }
        } else {
            int draw = 0;
            for (; gb_count * GB < transferred; gb_count++) {
                draw = 1;
            }
            if (draw) {
                double t = get_time();
                double elapsed = t - last_draw_time;
                if (elapsed > 0.100) {
                    printf("\r%80s\r", "");
                    printf("\rCurrent Speed: %.3f GB/s",
                        (transferred - last_draw_sz) / (t - last_draw_time) / GB);
                    fflush(stdout);
                    last_draw_sz = transferred;
                    last_draw_time = t;
                }
            }
        }
    }
    printf("\n");
}


static void print_results(double time) {
    double transferred_gb = (double) transferred / GB;
    printf("Transferred: %.1f GB\n", transferred_gb);
    printf("Time: %.3f s\n", time);
    printf("Speed: %.3f GB/s\n", transferred_gb / time);
}


static void on_interrupted(int _) {
    double end_time = get_time();
    printf("\n\nINTERRUPTED\n\n");
    print_results(end_time - start_time);
    exit(1);
}


int main(int argc, char *argv[]) {
    page_size = sysconf(_SC_PAGESIZE);
    size_t buffer_size_mb = 4 * 1024;
    size_t transfer_size_gb = 100;
    char *strategy = "c_loop";
    int use_mmap = 0;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--strat", 7) == 0) {
            if (argc < i + 2) {
                fprintf(stderr, "Expected STRATEGY argument\n");
                exit(1);
            }
            strategy = argv[++i];
        } else if (strncmp(argv[i], "--trans", 7) == 0) {
            if (argc < i + 2) {
                fprintf(stderr, "Expected TRANSFER_SIZE_GB argument\n");
                exit(1);
            }
            transfer_size_gb = str_to_pos_u64(argv[++i]);
        } else if (strcmp(argv[i], "--mmap") == 0) {
            use_mmap = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr, "Usage: %s [--strat[egy] STRATEGY] [--mmap] [--trans[fer] TRANSFER_SIZE_GB] BUFFER_SIZE_MB\n", argv[0]);
            fprintf(stderr, "    STRATEGY:\n");
            fprintf(stderr, "        c_loop               : A standard C loop subject to compiler optimizations\n");
            fprintf(stderr, "        c_loop_unrolled_x8   : A standard C loop with 8 x 64bit writes\n");
            fprintf(stderr, "        c_loop_unrolled_x32  : A standard C loop with 32 x 64bit writes\n");
            fprintf(stderr, "        c_loop_unrolled_x129 : A standard C loop with 128 x 64bit writes\n");
            fprintf(stderr, "        memset               : Byte by byte memset(2) in a loop\n");
            fprintf(stderr, "        memcpy               : 8 byte stride memcpy from stack buffer\n");
#ifndef __APPLE__
            fprintf(stderr, "        x86asm               : Non-temporal 64bit x86 ASM, MOVNTI\n");
            fprintf(stderr, "        x86asm_unrolled_x8   : Non-temporal 64bit x86 ASM, 8 x MOVNTI\n");
            fprintf(stderr, "        x86asm_unrolled_x32  : Non-temporal 64bit x86 ASM, 32 x MOVNTI\n");
            fprintf(stderr, "        avx2                 : 256bit AVX2 intrinsics (C based)\n");
            fprintf(stderr, "        avx512               : 512bit AVX512 intrinsics (C based)\n");
#else
            fprintf(stderr, "        armneon              : 128bit ARM NEON SIMD (C based)\n");
#endif
            fprintf(stderr, "    TRANSFER_SIZE_GB: Total amount to transfer through memory in GB\n");
            exit(0);
        } else {
            buffer_size_mb = str_to_pos_u64(argv[i]);
        }
    }
    const size_t buffer_size = buffer_size_mb * MB;
    const size_t transfer_size = transfer_size_gb * GB;
    if (buffer_size < page_size) {
        fprintf(stderr, "Invalid BUFFER_SIZE (too small)\n");
        exit(1);
    }
    if (transfer_size < buffer_size) {
        fprintf(stderr, "Invalid TRANSFER_SIZE_GB (too small)\n");
        exit(1);
    }
    mem_write_test test;
    if (strcmp(strategy, "c_loop") == 0) {
        test = mem_write_test_c_loop;
    } else if (strcmp(strategy, "c_loop_unrolled_x8") == 0) {
        test = mem_write_test_c_loop_unrolled_x8;
    } else if (strcmp(strategy, "c_loop_unrolled_x32") == 0) {
        test = mem_write_test_c_loop_unrolled_x32;
    } else if (strcmp(strategy, "c_loop_unrolled_x128") == 0) {
        test = mem_write_test_c_loop_unrolled_x128;
    } else if (strcmp(strategy, "memset") == 0) {
        test = mem_write_test_memset;
    } else if (strcmp(strategy, "memcpy") == 0) {
        test = mem_write_test_memcpy;
#ifndef __APPLE__
    } else if (strcmp(strategy, "x86asm") == 0) {
        test = mem_write_test_x86asm;
    } else if (strcmp(strategy, "x86asm_unrolled_x8") == 0) {
        test = mem_write_test_x86asm_unrolled_x8;
    } else if (strcmp(strategy, "x86asm_unrolled_x32") == 0) {
        test = mem_write_test_x86asm_unrolled_x32;
    } else if (strcmp(strategy, "avx2") == 0) {
        test = mem_write_test_avx2;
    } else if (strcmp(strategy, "avx512") == 0) {
        test = mem_write_test_avx512;
#else
    } else if (strcmp(strategy, "armneon") == 0) {
        test = mem_write_test_armneon;
#endif
    } else {
        fprintf(stderr, "Invalid test strategy\n");
        exit(1);
    }
    printf("Strategy: %s\n", strategy);
    printf("Page size: %ld\n", sysconf(_SC_PAGESIZE));
    printf("Target transfer size: %ld GB\n", transfer_size_gb);
    if (buffer_size_mb >= 1024) {
        printf("Allocating memory: %.2f GB\n", buffer_size_mb / 1024.0);
    } else {
        printf("Allocating memory: %ld MB\n", buffer_size_mb);
    }
    void *mem = alloc(buffer_size, use_mmap);
    printf("Pre-faulting memory...\n");
    // NOTE: Memset gets optimized out, and timing is maleffected if we don't prefill.
    for (size_t i = 0; i < buffer_size; i++) {
        ((char*)mem)[i] = 0;
    }
    signal(SIGINT, on_interrupted);
    printf("Running test...\n");
    bench(mem, buffer_size, transfer_size, test);
    double end_time = get_time();
    printf("\nCOMPLETED\n\n");
    dealloc(mem, buffer_size, use_mmap);
    print_results(end_time - start_time);
    return 0;
}
