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
 * Some of the test strategies use "non-temporal" instructions, sometimes called streaming
   or out of order writes.  This can be faster on lower end CPUs and/or with higher buffer sizes
   that exceed CPU caches.
 * If you're interested in exploring your cache sizes, play with the buffer size and observe the
   jumps in speed.
 * Some platforms may achieve higher overall bandwidth by running multiple copies.
 * Memory subsystems are far more complicated than a single "mega-transfers" number


Compatibility
--------
* x86 (64bit) Linux
* ARM macOS


Building
--------
```shell
:; make
cc -O3 -mtune=native -march=native -Wall memspeed.c -o memspeed
```


Usage
--------
NOTE: Options will vary between Linux and macOS.

Example Linux usage...
```
:; ./memspeed --help
Usage: ./memspeed [--strat[egy] STRATEGY] [--mmap] [--trans[fer] TRANSFER_SIZE_GB] BUFFER_SIZE_MB
    STRATEGY:
        c_loop          : A standard C loop subject to compiler optimizations
        c_loop_unrolled : A standard C loop with 8 x 64bit writes
        memset          : Byte by byte memset(2) in a loop
        memcpy          : 8 byte stride memcpy from stack buffer
        x86asm          : Non-temporal 64bit x86 ASM loop
        x86asm_unrolled : Non-temporal 64bit x86 ASM, 8 chunks unrolled
        avx2            : AVX2 intrinsics (C based)
    TRANSFER_SIZE_GB:     Total amount to transfer through memory in GB
```


Running
--------
**Basic**
```
:; ./memspeed
Strategy: c_loop
Page size: 4096
Target transfer size: 100 GB
Allocating memory: 4.00 GB
Using MALLOC
Pre-faulting memory...
Running test...
Current Speed: 25.388 GB/s                                                      

COMPLETED

Transferred: 100.0 GB
Time: 3.960 s
Speed: 25.255 GB/s
```

**Advanced**
Run 8 copies, pinned to first 8 cpu cores.
```
for x in $(seq 0 7) ; do taskset -c $x ./memspeed --strat avx2 --transfer 100 128 & done ; wait
```

