memspeed
--------

Simple C based memory bandwidth benchmark.

Basically doing this C call to page aligned heap (or shared mmap if you define USE_MMAP)...
```c
const u64 b = iter % 0xff;
u64 v = 0;
for (int i = 0; i < sizeof(u64); i++) {
    v = (v << 8) | b;
}
u64 len = size / sizeof(u64);
for (u64 i = 0; i < len; i++) {
    *(mem + i) = v;
}
if (mem[1] == 0xdeadbeef) { // Force compiler to perform write..
   exit(1);
}
```


Building
========
```shell
make
```

Produces `memspeed` exec


Running
========
```shell
$ ./memspeed 
Page size: 4096
Allocating memory: 8.0GB
Using MALLOC
Clearing memory...
Running test...
............
DONE
Transferred: 96.00GB
Time: 3.790s
Speed: 25.33GB/s
```
