#include <stdio.h>
#include <curand_kernel.h>

__global__ void test_curand()
{
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	curandState rgnState;
	curand_init(1234, tid, 0, &rgnState);

    float x = curand_uniform(&rgnState);
    float y = curand_uniform(&rgnState);
	printf("%f,%f\n", x, y);
}

int main(void)
{
	printf("start\n");
	test_curand<<<1, 1>>>();
	printf("end\n");
	return 0;
}
