CC = gcc
CPP = g++
NVCC = nvcc
GCC_OPTIONS = -Wall --pedantic-error
NVCC_OPTIONS = -arch=sm_52 -D_TREND -D_DEBUG -g

debug := CFALGS= -g

# NVCC_OPTIONS = -arch=sm_86 -D_TREND
# NVCC_OPTIONS = -arch=sm_86 -D_DEBUG -D_TREND
# NVCC_OPTIONS = --device-c -arch=sm_86 -D_DEBUG

GNUOPENMP = -fopenmp
INTELOPEMP = -openmp

APPS = onemax

TARGET = ${APPS}

all: ${APPS}
build: ${APPS}

Parameters.o: Parameters.cu
	$(NVCC) ${GPP_OPTIONS} $(NVCC_OPTIONS) -o $@ -c $<

CUDAKernels.o: CUDAKernels.cu
	$(NVCC) ${GPP_OPTIONS} $(NVCC_OPTIONS) -o $@ -c $<

Misc.o: Misc.cu
	$(NVCC) ${GPP_OPTIONS} $(NVCC_OPTIONS) -o $@ -c $<

onemax : main.cu CUDAKernels.o Misc.o Parameters.o
	$(NVCC) ${GPP_OPTIONS} $(NVCC_OPTIONS) $^ -o $@

test : test.cpp Parameters.o
	$(NVCC) ${GPP_OPTIONS} $(NVCC_OPTIONS) $^ -o $@

run: build

clean:
	rm -f ${APPS}
	rm -f *.o

clobber: clean
