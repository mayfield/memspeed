CC := cc
CFLAGS := -O3 -mtune=native -march=native -Wall

default: memspeed

memspeed: memspeed.c Makefile
	$(CC) $(CFLAGS) memspeed.c -o $@

asm: memspeed.c Makefile
	$(CC) $(CFLAGS) -S memspeed.c

clean:
	rm -f memspeed
