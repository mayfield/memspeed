memspeed
========
Simple C/ASM based memory/cache bandwidth benchmark.

This test essentially boils down to various implementations of this loop...
```c
u64 *mem = malloc(size);
for (u64 i = 0; i < size / sizeof(u64); i++) {
    mem[i] = <non_zero_value>;
}
```

 * Care is taken to prevent compilers (gcc, clang) from optimizing out the loop.
 * Data written to memory is altered between iterations but is not verified, i.e. write-only.
 * Some of the test strategies use non-temporal instructions, sometimes called streaming
   or out of order writes.  This can be faster on lower end CPUs and/or with higher buffer sizes
   that exceed CPU caches.
 * If you're interested in exploring your cache sizes, play with the buffer size and observe the
   jumps in speed.
 * Higher overall bandwidth may be seen with `--threads N`.
 * Memory subsystems are far more complicated than a single "mega-transfers" number
    * https://www.akkadia.org/drepper/cpumemory.pdf

https://github.com/user-attachments/assets/c29a2675-22b6-40b3-b6d2-5b10ac9847c9

Compatibility
--------
* Linux, macOS
* ASM for Aarch64 and x86_64


Building
--------
```shell
:; make
cc -O3 -mtune=native -march=native -std=c11 -Wall -Wextra -lpthread memspeed.c -o memspeed
```


Usage
--------
NOTE: Options will vary depending on your platform and instruction set.

Example Linux usage...
```
:; ./memspeed --help
Usage: ./memspeed [--strat[egy] STRATEGY]
                  [--mmap]
                  [--verbose]
                  [--trans[fer] TRANSFER_SIZE_GB]
                  [--threads THREAD_COUNT]
                  BUFFER_SIZE_MB

    STRATEGY:
        c               : A C loop subject to compiler optimizations
        c_x8            : A C loop with 8 x 64bit writes
        c_x32           : A C loop with 32 x 64bit writes
        c_x128          : A C loop with 128 x 64bit writes
        memset          : Byte by byte memset() in a loop
        memcpy          : Aligned page memcpy in a loop
        x86asm          : 64bit x86 ASM
        x86asm_nt       : 64bit x86 ASM (non-temporal)
        x86asm_x8       : 8 x 64bit x86 ASM
        x86asm_nt_x8    : 8 x 64bit x86 ASM (non-temporal)
        x86asm_x32      : 32 x 64bit x86 ASM
        x86asm_nt_x32   : 32 x 64bit x86 ASM (non-temporal)
        avx2            : 256bit AVX2 intrinsics
        avx2_nt         : 256bit AVX2 intrinsics (non-temporal)
        avx512          : 512bit AVX512 intrinsics
        avx512_nt       : 512bit AVX512 intrinsics (non-temporal)

    TRANSFER_SIZE_GB: Total amount to transfer through memory in GB
```


Running
--------
**Basic**
```
:; ./memspeed
Strategy: c
Page size: 4 KB
Transfer size: 100 GB
Allocating memory [malloc]: 4 GB
Pre-faulting memory...
Running test...
Current:   28.51 GB/s  |  Avg:   28.53 GB/s  |  Transferred: 96 GB              

COMPLETED

Transferred: 100 GB
Time: 3.505 s
Speed: 28.53 GB/s
```

**Advanced**
Combine taskset to isolate threads to specific cores or CCDs...
```
:; taskset -c 0,8 ./memspeed --strat avx512_nt --threads 2 --transfer 2000 --verbose
Strategy: avx512_nt
Page size: 4 KB
Transfer size: 2000 GB
Threads: 2
Thread shard: 2 GB
Allocating memory [malloc]: 4 GB
Pre-faulting memory...
Running test...
Available CPU cores: 0, 8
Thread 0 mapped to CPU core: 0
Thread 1 mapped to CPU core: 8
Current:   59.12 GB/s  |  Avg:   59.12 GB/s  |  Transferred: 1992 GB            

COMPLETED

Transferred: 2000 GB
Time: 33.828 s
Speed: 59.12 GB/s
```
