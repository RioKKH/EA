CC   = gcc
CPP  = g++
NVCC = nvcc

NVCC_OPTIONS = --generate-code arch=compute_50,code=sm_50,\
							 --generate-code arch=compute_52,code=sm_52
#NVCC_OPTIONS = --generate-code arch=compute_86,code=sm_86

CPPFLAGS=-O3 -Xptxas=-v
CPPINCLUDE=-I./Common 
CFLAGS=
debug= -g -D_DEBUG

APPS = gpuonemax

TARGET = ${APPS}
LDFLAGS =

all: ${APPS}
build: ${APPS}

CUDAKernels.o: CUDAKernels.cu
	${NVCC} ${NVCC_OPTIONS} ${debug} ${CPPINCLUDE} -o $@ -c $<

Evolution.o: Evolution.cu
	${NVCC} ${NVCC_OPTIONS} ${debug} ${CPPINCLUDE} -o $@ -c $<

Parameters.o: Parameters.cu
	${NVCC} ${NVCC_OPTIONS} ${debug} ${CPPINCLUDE} -o $@ -c $<

Population.o: Population.cu
	${NVCC} ${NVCC_OPTIONS} ${debug} ${CPPINCLUDE} -o $@ -c $<
 
#
#%.o : $.cu
	${NVCC} ${CPPINCLUDE} ${debug} ${CPPINCLUDE} ${NVCC_OPTIONS} -c $<
main.o: main.cpp
	${CPP} ${CFLAGS} ${debug} ${CPPINCLUDE} -c main.cpp


gpuonemax: main.o CUDAKernels.o Evolution.o Parameters.o Population.o
	${NVCC} ${LDFLAGS} ${NVCC_OPTIONS} ${debug} ${CPPINCLUDE} $^ -o $@

clean:
	rm -f ${APPS}
	rm -f *.o


