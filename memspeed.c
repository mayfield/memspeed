#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <sys/param.h>
#include <sys/mman.h>

#define BUS_TYPE uint64_t
//#define USE_MMAP 1

static uint64_t page_size = 0;
struct timespec _ts;

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
        exit(-1);
    } else if (end == raw) {
        fprintf(stderr, "Not a number: %s\n", raw);
        exit(-1);
    }
    return num;
}


static BUS_TYPE * restrict alloc(size_t size) {
    BUS_TYPE *ptr;
    #ifdef USE_MMAP
        printf("Using MMAP\n");
        ptr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            fprintf(stderr, "Mem alloc failed %s\n", strerror(errno));
            exit(-1);
        }
    #else
        printf("Using MALLOC\n");
        ptr = aligned_alloc(page_size, size);
        if (ptr == NULL) {
            fprintf(stderr, "Mem alloc failed %s\n", strerror(errno));
            exit(-1);
        }
    #endif
    return ptr;
}


static void dealloc(BUS_TYPE *ptr, size_t size) {
    #ifdef USE_MMAP
        munmap((void*) ptr, size);
    #else
        free((void*) ptr);
    #endif
}

static void mem_write_test1(BUS_TYPE * restrict mem, uint64_t size, uint64_t iter) {
    const BUS_TYPE b = iter % 0xff;
    BUS_TYPE v = 0;
    for (int i = 0; i < sizeof(BUS_TYPE); i++) {
        v = (v << 8) | b;
    }
    uint64_t len = size / sizeof(BUS_TYPE);
    for (uint64_t i = 0; i < len; i++) {
        *(mem + i) = v;
    }
}


static void mem_write_test2(BUS_TYPE * restrict mem, uint64_t size, uint64_t iter) {
    const char b = iter % 0xff;
    memset((void*) mem, b, size);
}


static uint64_t bench(BUS_TYPE * restrict mem, uint64_t size) {
    if (size == 0) {
        return 0;
    }
    const uint64_t target_size = 100l * 1024 * 1024 * 1024;
    const uint64_t max_iter = target_size / size;
    uint64_t iter;
    for (iter = 1; iter <= max_iter; iter++) {
        mem_write_test1(mem, size, iter);
        if (mem[1] == 0xdeadbeef) { // Force compiler to perform write..
            printf("Memory Error!\n");
            return 0;
        }
        printf(".");
        fflush(stdout);
    }
    return (iter - 1) * size;
}


int main(int argc, char *argv[]) {
    page_size = sysconf(_SC_PAGESIZE);
    uint64_t size_mb = 8 * 1024;
    if (argc > 1) {
        size_mb = str_to_pos_u64(argv[1]);
    }
    const uint64_t size = size_mb * 1024 * 1024;
    printf("Page size: %ld\n", sysconf(_SC_PAGESIZE));
    printf("Allocating memory: %.1fGB\n", size_mb / 1024.0);
    BUS_TYPE * restrict mem = alloc(size);
    printf("Clearing memory...\n");
    // NOTE: Memset gets optimized out, and timing is malaffected if we don't prefill.
    for (uint64_t i = 0; i < size; i++) {
        ((char*)mem)[i] = 0;
    }
    printf("Running test...\n");
    double start = get_time();
    uint64_t transferred = bench(mem, size);
    double end = get_time();
    printf("\nDONE\n");
    dealloc(mem, size);
    double transferred_gb = (double) transferred / 1024 / 1024 / 1024;
    printf("Transferred: %.2fGB\n", transferred_gb);
    printf("Time: %.3fs\n", end - start);
    printf("Speed: %.2fGB/s\n", transferred_gb / (end - start));
    return 0;
}
