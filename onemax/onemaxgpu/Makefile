CC = gcc
CPP = g++
NVCC = nvcc
GCC_OPTIONS = -Wall --pedantic-error
GPP_OPTIONS = -std=c++17 -Wall --pedantic-error -D_DEBUG
NVCC_OPTIONS = -arch=sm_52 -D_DEBUG
#NVCC_OPTIONS = -arch=sm_86 -D_DEBUG

GNUOPENMP = -fopenmp
INTELOPEMP = -openmp

APPS = onemax test

TARGET = ${APPS}

all: ${APPS}
build: ${APPS}

Parameters.o: Parameters.cpp Parameters.hpp
	$(NVCC) $(NVCC_OPTIONS) -o $@ -c $<

CUDAKernels.o: CUDAKernels.cu CUDAKernels.h
	$(NVCC) $(NVCC_OPTIONS) -o $@ -c $<

Misc.o: Misc.cu Misc.h
	$(NVCC) $(NVCC_OPTIONS) -o $@ -c $<

onemax : main.cu CUDAKernels.o Misc.o Parameters.o
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

test : test.cpp Parameters.o
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

run: build

clean:
	rm -f ${APPS} rm -f *.o

clobber: clean
