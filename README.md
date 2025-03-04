memspeed
========
Simple C/ASM based memory bandwidth benchmark.

This test essentially boils down to various implementations of this loop...
```c
u64 *mem = malloc(size);
for (u64 i = 0; i < size / sizeof(u64); i++) {
    mem[i] = <non_zero_value>;
}
```

 * Care is taken to prevent compilers (gcc, clang) from optimizing out the loop.
 * Data written to memory is altered between iterations but is not verified, i.e. write-only.
 * Some of the test strategies use "njon-temporal" instructions, sometimes called streaming
   or out of order writes.  This can be faster on lower end CPUs and/or with higher buffer sizes
   that exceed CPU caches.
 * If you're interested in exploring your cache sizes, play with the buffer size and observe the
   jumps in speed.
 * Some platforms may achieve higher overall bandwidth with `--threads N`.
 * Memory subsystems are far more complicated than a single "mega-transfers" number

https://github.com/user-attachments/assets/4be00edb-746f-4131-8f6a-555299536ee6

Compatibility
--------
* x86 (64bit) Linux
* ARM macOS


Building
--------
```shell
:; make
cc -O3 -mtune=native -march=native -std=c11 -Wall -Wextra -lpthread memspeed.c -o memspeed
```


Usage
--------
NOTE: Options will vary between Linux and macOS.

Example Linux usage...
```
:; ./memspeed --help
Usage: ./memspeed [--strat[egy] STRATEGY]
                  [--mmap]
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
Page size: 4096
Transfer size: 100 GB
Allocating memory: 4 GB
Using MALLOC
Pre-faulting memory...
Running test...
Current Speed: 25.23 GB/s                                                       

COMPLETED

Transferred: 100 GB
Time: 3.959 s
Speed: 25.259 GB/s
```

**Advanced**
Explore maximum cache performance...
```
:; ./memspeed --strat avx512 --threads 32 --transfer 2000 32
Strategy: avx512
Page size: 4096
Transfer size: 2000 GB
Allocating memory: 32 MB
Threads: 32
Thread Shard: 1024 KB
Using MALLOC
Pre-faulting memory...
Running test...
Current Speed: 1509.85 GB/s                                                     

COMPLETED

Transferred: 2000 GB
Time: 1.350 s
Speed: 1481.769 GB/s
```

