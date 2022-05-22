#include <stdio.h>
#include <stdlib.h>

#include "CUDAKernels.h"
#include "Parameters.hpp"
#include "Misc.h"

extern __constant__ EvolutionParameters *gpuEvoPrms;
extern __constant__ int POPSIZE;

/*
#define CUDA_CALL(x) do                                     \
{                                                           \
    if((x) != cudaSuccess)                                  \
    {                                                       \
        printf("Error at %s:%d\n", __FILE__, __LINE__);     \
        return EXIT_FAILURE;                                \
    }                                                       \
} while (0)                                                 \

#define CURAND_CALL(x) do                                   \
{                                                           \
    if((x) != CURAND_STATUS_SUCCESS)                        \
    {                                                       \
        printf("Error at %s:%d\n", __FILE__, __LINE__);     \
        return EXIT_FAILURE;                                \
    }                                                       \
} while (0)                                                 \
*/

__global__ void setup_kernel(curandState *state)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state, float *result)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	float x;

	curandState localState = state[id];
	
	x = curand_uniform(&localState);

	state[id] = localState;
	result[id] = x;
}

__global__ void evaluation(int *population, int *fitness)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	int stride;

	extern __shared__ volatile int s_idata[];
	s_idata[tx] = population[i];
	__syncthreads();

	for (stride = 1; stride <= blockDim.x/2; stride <<= 1)
	{
		if (tx % (2 * stride) == 0)
		{
			s_idata[tx] = s_idata[tx] + s_idata[tx + stride];
		}
		__syncthreads();
	}

	if (tx == 0)
	{
		fitness[blockIdx.x] = s_idata[0];
	}
}

__host__ __device__ int getBestIndividual(const int *fitness)
{
	int best = 0;
	int best_index = 0;
	for (int i = 0; i < gpuEvoPrms->TOURNAMENT_SIZE; ++i)
	// for (int i = 0; i < TOURNAMENT_SIZE; ++i)
	{
		if (fitness[i] > best)
		{
			best = fitness[i];
			best_index = i;
		}
	}

	return best_index;
}

__device__ int tournamentSelection(const int *fitness,
                                   curandState *dev_States,
                                   const int &ix,
                                   PARENTS mf,
                                   int gen,
                                   int *tournament_individuals,
                                   int *tournament_fitness)
{
	int best_id;
	// int tournament_individuals[gpuEvoPrms->TOURNAMENT_SIZE];
	// int tournament_fitness[gpuEvoPrms->TOURNAMENT_SIZE];
	// int tournament_individuals[TOURNAMENT_SIZE];
	// int tournament_fitness[TOURNAMENT_SIZE];
	unsigned int random_id;
	unsigned int offset = (gpuEvoPrms->POPSIZE * gpuEvoPrms->TOURNAMENT_SIZE) * mf;
	// unsigned int offset = (POPSIZE * TOURNAMENT_SIZE) * mf;

	for (int i = 0; i < gpuEvoPrms->TOURNAMENT_SIZE; ++i) {
	// for (int i = 0; i < TOURNAMENT_SIZE; ++i) {
		// curand_uniform returns random number uniformly distributed between (0, 1].
		curandState localState = dev_States[ix * gpuEvoPrms->TOURNAMENT_SIZE
                                            + i + offset + gpuEvoPrms->POPSIZE * gen];
		// curandState localState = dev_States[ix * TOURNAMENT_SIZE + i + offset + POPSIZE * gen];
		// curandState localState = dev_States[ix * TOURNAMENT_SIZE + i + offset]; // w/o generation
		random_id = (unsigned int)(curand_uniform(&localState) * (gpuEvoPrms->POPSIZE));
		// random_id = (unsigned int)(curand_uniform(&localState) * (POPSIZE));
		tournament_individuals[i] = random_id;
		tournament_fitness[i] = fitness[random_id]; }
	best_id = getBestIndividual(tournament_fitness);

	return tournament_individuals[best_id];
}

__global__ void selection(int* fitness,
                          int* sortedid,
                          curandState *dev_States,
                          int* parent1,
                          int* parent2,
                          int gen,
                          int *tournament_individuals,
                          int *tournament_fitness)
{
    int tx = threadIdx.x;

    if (gpuEvoPrms->POPSIZE - gpuEvoPrms->NUM_OF_ELITE <= tx)
    // if (POPSIZE - NUM_OF_ELITE <= tx)
    {
        parent1[tx] = sortedid[tx];
        parent2[tx] = sortedid[tx];
    }
    else
    {
        parent1[tx] = tournamentSelection(fitness, dev_States, tx, MALE, gen,
                                          tournament_individuals, tournament_fitness);
        parent2[tx] = tournamentSelection(fitness, dev_States, tx, FEMALE, gen,
                                          tournament_individuals, tournament_fitness);
    }
}

__device__ void singlepointCrossover(const int *src, int *dst, int tx, curandState localState, int parent1, int parent2) 
{ 
	int i = 0;
	unsigned int point1;
	// txは個体のインデックス
	// 従ってこのoffsetはpopulation のオフセットになる。
	int offset = tx * gpuEvoPrms->CHROMOSOME;
	// int offset = tx * CHROMOSOME;

	point1 = (unsigned int)(curand_uniform(&localState) * (gpuEvoPrms->CHROMOSOME));
	// point1 = (unsigned int)(curand_uniform(&localState) * (CHROMOSOME));
	for (i = 0; i < point1; ++i)
	{
		dst[i + offset] = src[parent1 * gpuEvoPrms->CHROMOSOME + i];
		// dst[i + offset] = src[parent1 * CHROMOSOME + i];
	}
	for (; i < gpuEvoPrms->CHROMOSOME; ++i)
	// for (; i < CHROMOSOME; ++i)
	{
		dst[i + offset] = src[parent2 * gpuEvoPrms->CHROMOSOME + i];
		// dst[i + offset] = src[parent2 * CHROMOSOME + i];
	}
}

/**
 * @param[in] src		Population where current-generation data is stored.
 * @param[out] dst		Population where next-generation data is stored.
 * @param[in] parent1	Fitness of parent 1
 * @param[in] parent2	Fitness of parent 2
 * @return void
 */
__global__ void crossover(
		const int *src,
		int *dst,
		curandState *dev_States,
		const int *parent1,
		const int *parent2,
		const int gen)
{
	int tx = threadIdx.x;

	extern __shared__ volatile int s_parent[];
	s_parent[tx] = parent1[tx];
	s_parent[tx + gpuEvoPrms->POPSIZE] = parent2[tx];
	// s_parent[tx + POPSIZE] = parent2[tx];
	__syncthreads();

	curandState localState = dev_States[tx + gpuEvoPrms->POPSIZE * gen];
	// curandState localState = dev_States[tx + POPSIZE * gen];
	singlepointCrossover(src, dst, tx, localState, s_parent[tx], s_parent[tx + gpuEvoPrms->POPSIZE]);
	// singlepointCrossover(src, dst, tx, localState, s_parent[tx], s_parent[tx + POPSIZE]);
	__syncthreads();
}

__global__ void mutation(int *population, curandState *dev_States, const int gen)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = dev_States[id + gpuEvoPrms->POPSIZE * gen];
	// curandState localState = dev_States[id + POPSIZE * gen];

	if (curand_uniform(&localState) < gpuEvoPrms->MUTATION_RATE)
	// if (curand_uniform(&localState) < MUTATION_RATE)
	{
		population[id] ^= 1;
	}
}

__global__ void dev_show(int *population, int *fitness, int *sortedfitness, int *parent1, int *parent2)
{
	int tx = threadIdx.x;
	if (gpuEvoPrms->POPSIZE - gpuEvoPrms->NUM_OF_ELITE <= tx)
	// if (POPSIZE - NUM_OF_ELITE <= tx)
	{
		printf("%d,%d,%d,%d\n", tx, sortedfitness[tx], parent1[tx], parent2[tx]);
	}
	else {
		printf("%d,%d,%d,%d\n", tx, fitness[tx], parent1[tx], parent2[tx]);
	}
}

__global__ void dev_prms_show(void)
{
    printf("hello\n");
    printf("dev_prms_show %d\n", POPSIZE);
    // printf("%d\n", gpuEvoPrms->POPSIZE);
    // printf("%d\n", gpuEvoPrms->CHROMOSOME);
    // printf("%d\n", gpuEvoPrms->NUM_OF_ELITE);
}


