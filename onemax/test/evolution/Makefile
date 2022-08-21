CC   = gcc
CPP  = g++
NVCC = nvcc

NVCC_OPTIONS = --generate-code arch=compute_52,code=sm_52 \
							 --generate-code arch=compute_53,code=sm_53 \
							 --generate-code arch=compute_60,code=sm_60 \
							 --generate-code arch=compute_61,code=sm_61 \
							 --generate-code arch=compute_62,code=sm_62 \
							 --generate-code arch=compute_70,code=sm_70 \
							 --generate-code arch=compute_72,code=sm_72 \
							 --generate-code arch=compute_75,code=sm_75 \
							 --generate-code arch=compute_80,code=sm_80 \

debug := CFLAGS = -g

APPS = gpuonemax

TARGET = ${APPS}
LDFLAGS =

all: ${APPS}
build: ${APPS}

CUDAKernels.o: CUDAKernels.cu
	${NVCC} ${NVCC_OPTIONS} -o $@ -c $<

Evolution.o: Evolution.cu
	${NVCC} ${NVCC_OPTIONS} -o $@ -c $<

Parameters.o: Parameters.cu
	${NVCC} ${NVCC_OPTIONS} -o $@ -c $<

main.o: main.cpp
	${NVCC} ${NVCC_OPTIONS} -o $@ -c $<

gpuonemax: main.o CUDAKernels.o Evolution.o Parameters.o
	${NVCC} ${LDFLAGS} ${NVCC_OPTIONS} $^ -o $@

clean:
	rm -f ${APPS}
	rm -f *.o


