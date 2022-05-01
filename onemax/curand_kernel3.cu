#include <iostream>
#include <iomanip>
#include <curand_kernel.h>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

template <typename T>
void printMatrix(const T *matrix, const int ldm, const int n)
{
	for (int j=0; j<ldm; j++)
	{
		for (int i=0; i<n; i++)
		{
			std::cout << std::fixed << std::setw(12)
				<< std::setprecision(4) << matrix[IDX2C(i, j, ldm)];
			std::cout << std::endl;
		}
	}
}

__global__ void setup_kernel(curandState_t *state)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// Each thread gets same seed,
	// a different sequence number, no offset
	curand_init(2019UL, idx, 0, &state[idx]);
}

__global__ void generate_kernel(unsigned int *generated_out, curandState_t *state)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	generated_out[idx] = curand(&state[idx]) & 0xFF;
}

__global__ void generate_uniform_kernel(float *generated_out, curandState_t *state)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	generated_out[idx] = curand_uniform(&state[idx]);
}

