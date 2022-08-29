CC   = gcc
CPP  = g++
NVCC = nvcc

NVCC_OPTIONS = --generate-code arch=compute_86,code=sm_86

CPPFLAGS=-O3 -Xptxas=-v
CPPINCLUDE=-I./Common 
debug := CFLAGS = -g

APPS = gpuonemax

TARGET = ${APPS}
LDFLAGS =

all: ${APPS}
build: ${APPS}

CUDAKernels.o: CUDAKernels.cu
	${NVCC} ${NVCC_OPTIONS} ${CPPINCLUDE} -o $@ -c $<

Evolution.o: Evolution.cu
	${NVCC} ${NVCC_OPTIONS} ${CPPINCLUDE} -o $@ -c $<

Parameters.o: Parameters.cu
	${NVCC} ${NVCC_OPTIONS} ${CPPINCLUDE} -o $@ -c $<

Population.o: Population.cu
	${NVCC} ${NVCC_OPTIONS} ${CPPINCLUDE} -o $@ -c $<

main.o: main.cpp
	${CPP} ${CFLAGS} ${CPPINCLUDE} -c main.cpp

#%.o : $.cu
#	${NVCC} ${CPPINCLUDE} ${CPPINCLUDE} ${NVCC_OPTIONS} -c $<

gpuonemax: main.o CUDAKernels.o Evolution.o Parameters.o Population.o
	${NVCC} ${LDFLAGS} ${NVCC_OPTIONS} ${CPPINCLUDE} $^ -o $@

clean:
	rm -f ${APPS}
	rm -f *.o


