#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

__global__ void setup_kernel(curandState *state, uint64_t seed)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, tid, 0, &state[tid]);
}

__global__ void generate_randoms(curandState *globalState, float *randoms)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	curandState localState = globalState[tid];
	randoms[tid * 2 + 0] = curand_uniform(&localState);
	randoms[tid * 2 + 1] = curand_uniform(&localState);
}

int main()
{
	int threads = 256;
	int blocks = 5120;
	int threadCount = blocks * threads;
	int N = blocks * threads * 2;

	curandState *dev_curand_states;
	float *dev_randomValues;
	float *host_randomValues;
	int *host_int;

	host_randomValues = (float *)malloc(N * sizeof(float));
	host_int = (int *)malloc(N * sizeof(float));

	cudaMalloc(&dev_curand_states, threadCount * sizeof(curandState));
	cudaMalloc(&dev_randomValues, N * sizeof(float));

	generate_randoms<<<blocks, threads>>>(dev_curand_states, dev_randomValues);

	cudaMemcpy(host_randomValues, dev_randomValues, N * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i=0; i<N; ++i)
	{
		if (i < 8)
		{
			printf("%.8f, ", host_randomValues[i]);
		}
	}

	cudaFree(dev_curand_states);
	cudaFree(dev_randomValues);

	free(host_randomValues);
	free(host_int);

	return 0;
}

