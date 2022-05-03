#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {	\
	printf("Error at %s:%d\n", __FILE__, __LINE__);	\
	return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandState *state)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state, int n, unsigned int *result)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int count = 0;
	unsigned int x;
	
	curandState localState = state[id];

	for (int i = 0; i < n; i++)
	{
		x = curand(&localState);
		if (x & 1)
		{
			count++;
		}
	}

	state[id] = localState;
	result[id] += count;
}

__global__ void generate_uniform_kernel(curandState *state, float *result)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int count = 0;
	float x;

	curandState localState = state[id];

	x = curand_uniform(&localState);

	state[id] = localState;
	result[id] = x;
	// result[id] += count;
}


int main(int argc, char **argv)
{
	const unsigned int threadPerBlock = 64;
	const unsigned int blockCount = 64;
	const unsigned int totalThreads = threadPerBlock * blockCount;

	unsigned int i;
	unsigned int total;

	curandState *devStates;

	unsigned int *devResults, *hostResults;
	float *hostfResults, *devfResults;

	int sampleCount = 10000;

	int device;
	struct cudaDeviceProp properties;

	CUDA_CALL(cudaGetDevice(&device));
	CUDA_CALL(cudaGetDeviceProperties(&properties, device));

	hostResults = (unsigned int *)calloc(totalThreads, sizeof(int));
	hostfResults = (float *)calloc(totalThreads, sizeof(float));

	// CUDA_CALL(cudaMalloc((void **)&devResults, totalThreads * sizeof(unsigned int)));
	// CUDA_CALL(cudaMemset(devResults, 0, totalThreads * sizeof(unsigned int)));
	CUDA_CALL(cudaMalloc((void **)&devfResults, totalThreads * sizeof(float)));
	CUDA_CALL(cudaMemset(devfResults, 0.0f, totalThreads * sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&devStates, totalThreads * sizeof(curandState)));

	setup_kernel<<<64, 64>>>(devStates);

	// CUDA_CALL(cudaMemset(devfResults, 0, totalThreads * sizeof(float)));
	CUDA_CALL(cudaMemset(devfResults, 0.0f, totalThreads * sizeof(float)));

	// for (i = 0; i < 50; ++i)
	// {
		generate_uniform_kernel<<<64, 64>>>(devStates, devfResults);
	// }

	// CUDA_CALL(cudaMemcpy(hostResults, devResults, totalThreads * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(hostfResults, devfResults, totalThreads * sizeof(float), cudaMemcpyDeviceToHost));
	total = 0;
	for (i = 0; i < totalThreads; i++)
	{
		printf("%f\n", hostfResults[i]);
		total += hostfResults[i];
	}
	printf("\n");
	printf("Fraction of uniforms > 0.5 was %10.13f\n", (float)total / (totalThreads * sampleCount * 50.0f));

    CUDA_CALL(cudaFree(devStates));
	CUDA_CALL(cudaFree(devResults));
	CUDA_CALL(cudaFree(devfResults));
	free(hostResults);
	free(hostfResults);

    return EXIT_SUCCESS;
}


