CC := cc
CFLAGS := -O3 -mtune=native -march=native -Wall

default: memspeed

memspeed: memspeed.c
	$(CC) $(CFLAGS) memspeed.c -o $@
