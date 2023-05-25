CC=gcc
CFLAGS=-I.
LIBS=-lm
SRC=demo.c cml.c

cml:
	$(CC) -o demo $(SRC) $(LIBS)