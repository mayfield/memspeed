CC := cc
CFLAGS := -O3 -mtune=native -march=native -Wall -Wno-unused-function -fverbose-asm

default: memspeed

memspeed: memspeed.c Makefile
	$(CC) $(CFLAGS) memspeed.c -o $@
