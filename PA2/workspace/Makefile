CC       = mpicc
OPT     ?= -O3
CFLAGS   = -Wall -std=c99 $(OPT) -fopenmp -mfma
LDFLAGS  = -Wall -fopenmp -lm
LDLIBS   = $(LDFLAGS)

prerequisites  = check.o benchmark.o
targets        = benchmark-naive benchmark-omp benchmark-mpi
targets       += benchmark-omp-blocked benchmark-omp-oblivious benchmark-omp-timeskew benchmark-omp-circqueue
targets       += benchmark-omp-simd benchmark-omp-intrinsic
targets       += benchmark-mpi-naive benchmark-mpi-inaive benchmark-mpi-blocked


.PHONY : default
default : all

.PHONY : all
all : $(targets)

benchmark-naive : $(prerequisites) stencil-naive.o
	$(CC) -o $@ $^ $(LDFLAGS)

benchmark-omp : $(prerequisites) stencil-omp.o
	$(CC) -o $@ $^ $(LDFLAGS)

benchmark-omp-blocked : $(prerequisites) stencil-omp-blocked.o
	$(CC) -o $@ $^ $(LDFLAGS)

benchmark-omp-oblivious : $(prerequisites) stencil-omp-oblivious.o
	$(CC) -o $@ $^ $(LDFLAGS)

benchmark-omp-timeskew : $(prerequisites) stencil-omp-timeskew.o
	$(CC) -o $@ $^ $(LDFLAGS)

benchmark-omp-circqueue : $(prerequisites) stencil-omp-circqueue.o
	$(CC) -o $@ $^ $(LDFLAGS)

benchmark-omp-simd : $(prerequisites) stencil-omp-simd.o
	$(CC) -o $@ $^ $(LDFLAGS)

benchmark-omp-intrinsic : $(prerequisites) stencil-omp-intrinsic.o
	$(CC) -o $@ $^ $(LDFLAGS)

benchmark-mpi : $(prerequisites) stencil-mpi.o
	$(CC) -o $@ $^ $(LDFLAGS)

benchmark-mpi-naive : $(prerequisites) stencil-mpi-naive.o
	$(CC) -o $@ $^ $(LDFLAGS)

benchmark-mpi-inaive : $(prerequisites) stencil-mpi-inaive.o
	$(CC) -o $@ $^ $(LDFLAGS)

benchmark-mpi-blocked : $(prerequisites) stencil-mpi-blocked.o
	$(CC) -o $@ $^ $(LDFLAGS)

%.o : %.c common.h
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf benchmark-* *.o
