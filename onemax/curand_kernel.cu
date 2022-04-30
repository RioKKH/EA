#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>

#define N (1024)

__global__ void random(float *value)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int seed;
	curandState stat;

	// 乱数のシード
    seed = 1;
	// 第1引数は乱数のシード。全スレッドで同じ値を用いる
	// 第2引数はシーケンス番号。スレッド番号など、全スレッドで異なる値
	curand_init(seed, i, 0, &stat);

	value[i] = curand_uniform(&stat);
}

int main()
{
	int i;
	float *value, *value_d;
	cudaMalloc((void**)&value_d, N * sizeof(float));

	random<<<1, 1>>>(value_d);

    value = (float * )malloc(N * sizeof(float));
	cudaMemcpy(value, value_d, N * sizeof(float), cudaMemcpyDeviceToHost);

	for (i=0; i<N; i++)
	{
		printf("%f \n", value[i]);
	}

	cudaFree(value_d);
	free(value);

	return 0;
}
