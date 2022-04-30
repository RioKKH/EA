#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>

#define N 1024

/**
 * CPUから関数を呼ぶ場合
 */
int main()
{
	int i;
	float *value, *value_d;

	// generatorを宣言
	curandGenerator_t gen;
	cudaMalloc((void**)&value_d, N * sizeof(float));
	// Generatorの生成
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	// 一様乱数を生成
	curandGenerateUniform(gen, value_d, N);
	value = (float *)malloc(N * sizeof(float));
	cudaMemcpy(value, value_d, N * sizeof(float), cudaMemcpyDeviceToHost);

	for (i=0; i<N; i++)
	{
		printf("%f\n", value[i]);
	}

	curandDestroyGenerator(gen);
	cudaFree(value_d);
	free(value);
	return 0;
}
