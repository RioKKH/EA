CC = gcc
CPP = g++
NVCC = nvcc
GCC_OPTIONS = -Wall --pedantic-error
GPP_OPTIONS = -std=c++17 -Wall --pedantic-error -D_DEBUG
#NVCC_OPTIONS = -arch=sm_50
NVCC_OPTIONS = -arch=sm_86 -D_DEBUG

GNUOPENMP = -fopenmp
INTELOPEMP = -openmp

APPS = population curand curand_kernel curand_kernel2\
       curand_kernel4 curand_kernel5 curand_kernel6\
       curand_kernel7 curand_kernel8\
       onemax

TARGET = ${APPS}

all: ${APPS}
build: ${APPS}

population : population.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

test_population : test_population.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

individual.o: individual.cu individual.hpp
	$(NVCC) $(NVCC_OPTIONS) -o $@ -c $<

curand : curand.cu
	$(NVCC) $(NVCC_OPTIONS) -lcurand $^ -o $@

curand_kernel : curand_kernel.cu
	$(NVCC) $(NVCC_OPTIONS) -lcurand $^ -o $@

curand_kernel2 : curand_kernel2.cu
	$(NVCC) $(NVCC_OPTIONS) -lcurand $^ -o $@

curand_kernel4 : curand_kernel4.cu
	$(NVCC) $(NVCC_OPTIONS) -lcurand $^ -o $@

curand_kernel5 : curand_kernel5.cu
	$(NVCC) $(NVCC_OPTIONS) -lcurand $^ -o $@

curand_kernel6 : curand_kernel6.cu
	$(NVCC) $(NVCC_OPTIONS) -lcurand $^ -o $@

curand_kernel7 : curand_kernel7.cu
	$(NVCC) $(NVCC_OPTIONS) -lcurand $^ -o $@

curand_kernel8 : curand_kernel8.cu
	$(NVCC) $(NVCC_OPTIONS) -D_DEBUG -lcurand $^ -o $@

parameters.o: parameters.cu parameters.hpp
	$(NVCC) $(NVCC_OPTIONS) -o $@ -c $<

onemax : onemax.cu
	$(NVCC) $(NVCC_OPTIONS) $^ -o $@

#onemax.o: onemax.cu
#	$(NVCC) $(NVCC_OPTIONS) -o $@ -c $<
#
#onemax: onemax.o individual.o parameters.o
#	$(NVCC) $(NVCC_OPTIONS) -o $@ $+

run: build

clean:
	rm -f ${APPS} rm -f *.o

clobber: clean
