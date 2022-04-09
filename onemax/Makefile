CC = gcc
CPP = g++
NVCC = nvcc
GCC_OPTIONS = -Wall --pedantic-error
GPP_OPTIONS = -std=c++17 -Wall --pedantic-error
NVCC_OPTIONS = -arch=sm_86

GNUOPENMP = -fopenmp
INTELOPEMP = -openmp

APPS = population test_population

TARGET = ${APPS}

all: ${APPS}
build: ${APPS}

population : population.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

test_population : test_population.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

individual.o: individual.cu individual.hpp
	$(NVCC) $(NVCC_OPTIONS) -o $@ -c $<

parameters.o: parameters.cu parameters.hpp
	$(NVCC) $(NVCC_OPTIONS) -o $@ -c $<

onemax.o: onemax.cu
	$(NVCC) $(NVCC_OPTIONS) -o $@ -c $<

onemax: onemax.o individual.o parameters.o
	$(NVCC) $(NVCC_OPTIONS) -o $@ $+

run: build

clean:
	rm -f ${APPS} rm -f *.o

clobber: clean
