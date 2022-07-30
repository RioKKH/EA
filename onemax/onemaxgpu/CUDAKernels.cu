#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "Random123/philox.h"
#include "CUDAKernels.h"
#include "Parameters.hpp"
#include "Misc.h"


using namespace r123;
typedef r123::Philox2x32 RNG_2x32;
typedef r123::Philox4x32 RNG_4x32;
// typedef r123::Philox2x64 RNG_2x64;
// typedef r123::Philox4x32 RNG_4x64;

__constant__ float RANDMAX = 4294967295.0f;
__constant__ EvolutionParameters gpuEvoPrms;

#define CUDA_CALL(call)                                                    \
{                                                                          \
    const cudaError_t error = call;                                        \
    if(error != cudaSuccess)                                               \
    {                                                                      \
        printf("Error at %s:%d\n", __FILE__, __LINE__);                    \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        return EXIT_FAILURE;                                               \
    }                                                                      \
}                                                                          \

#define CURAND_CALL(x) do                                                  \
{                                                                          \
    if((x) != CURAND_STATUS_SUCCESS)                                       \
    {                                                                      \
        printf("Error at %s:%d\n", __FILE__, __LINE__);                    \
        return EXIT_FAILURE;                                               \
    }                                                                      \
} while (0)                                                                \

void copyToDevice(EvolutionParameters cpuEvoPrms)
{
#ifdef _DEBUG
    printf("copyToDevice %d\n", cpuEvoPrms.POPSIZE);
#endif // _DEBUG
    cudaMemcpyToSymbol(gpuEvoPrms,
                       &cpuEvoPrms,
                       sizeof(EvolutionParameters));
}

// __device__ RNG_2x32::ctr_type generateTwoRndValues(unsigned int key, unsigned int counter);
inline __device__ RNG_2x32::ctr_type generateTwoRndValues(unsigned int key,
                                                          unsigned int counter)
{
    RNG_2x32 rng;
    return rng({0, counter}, {key});
}

/*
__global__ void setup_kernel(curandState *state)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1234, id, 0, &state[id]);
}
*/

/*
__global__ void generate_kernel(curandState *state, float *result)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	float x;

	curandState localState = state[id];
	
	x = curand_uniform(&localState);

	state[id] = localState;
	result[id] = x;
}
*/

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
	for (int i = 0; i < gpuEvoPrms.TOURNAMENT_SIZE; ++i)
	// for (int i = 0; i < gpuEvoPrms->TOURNAMENT_SIZE; ++i)
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
                                   // curandState *dev_States,
                                   unsigned int *rand,
                                   const int &ix,
                                   PARENTS mf,
                                   int gen,
                                   int *tournament_individuals,
                                   int *tournament_fitness)
{
	int best_id;
	unsigned int random_id;
	unsigned int offset = (gpuEvoPrms.POPSIZE * gpuEvoPrms.TOURNAMENT_SIZE) * mf;

    for (int i = 0; i < gpuEvoPrms.TOURNAMENT_SIZE; ++i)
    {
        printf("%d\n", rand[i]);
    }

	for (int i = 0; i < gpuEvoPrms.TOURNAMENT_SIZE; ++i) {
		// curandState localState = dev_States[ix * gpuEvoPrms.TOURNAMENT_SIZE
        //                                     + i + offset + gpuEvoPrms.POPSIZE * gen];
		// random_id = (unsigned int)(curand_uniform(&localState) * (gpuEvoPrms.POPSIZE));
		// tournament_individuals[i] = random_id;
		tournament_fitness[i] = fitness[rand[i]]; }
	best_id = getBestIndividual(tournament_fitness);

	return tournament_individuals[best_id];
}

__global__ void selection(int* fitness,
                          int* sortedid,
                          // curandState *dev_States,
                          unsigned int *rand1,
                          unsigned int *rand2,
                          int* parent1,
                          int* parent2,
                          int gen,
                          int *tournament_individuals,
                          int *tournament_fitness)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tx = threadIdx.x;

    printf("selection %d\n", gen);

    // const RNG_2x32::ctr_type randomValues = generateTwoRndValues(idx, gen + i);
    for (int i = 0; i < gpuEvoPrms.TOURNAMENT_SIZE; ++i)
    {
        RNG_2x32::ctr_type randomValues = generateTwoRndValues(idx, gen + i);
        rand1[i] = randomValues.v[0] % gpuEvoPrms.POPSIZE;
        rand2[i] = randomValues.v[1] % gpuEvoPrms.POPSIZE;
    }

    //- エリート戦略で所定の数だけ上位の個体を残す
    if (gpuEvoPrms.POPSIZE - gpuEvoPrms.NUM_OF_ELITE <= tx)
    {
        parent1[tx] = sortedid[tx];
        parent2[tx] = sortedid[tx];
    }
    //- エリート戦略で残す個体以外をトーナメント選択で選ぶ
    else
    {
        parent1[tx] = tournamentSelection(fitness, rand1, tx, MALE, gen,
                                          tournament_individuals, tournament_fitness);
        parent2[tx] = tournamentSelection(fitness, rand2, tx, FEMALE, gen,
                                          tournament_individuals, tournament_fitness);
    }
}

/*
__device__ void singlepointCrossover(const int *src, int *dst, int tx, curandState localState, int parent1, int parent2) 
{ 
	int i = 0;
	unsigned int point1;
	// txは個体のインデックス
	// 従ってこのoffsetはpopulation のオフセットになる。
	int offset = tx * gpuEvoPrms.CHROMOSOME;

	point1 = (unsigned int)(curand_uniform(&localState) * (gpuEvoPrms.CHROMOSOME));
	for (i = 0; i < point1; ++i)
	{
		dst[i + offset] = src[parent1 * gpuEvoPrms.CHROMOSOME + i];
	}
	for (; i < gpuEvoPrms.CHROMOSOME; ++i)
	{
		dst[i + offset] = src[parent2 * gpuEvoPrms.CHROMOSOME + i];
	}
}
*/

__device__ void swap(unsigned int &point1, unsigned int &point2)
{
    const unsigned int tmp = point1;
    point1 = point2;
    point2 = tmp;
}

/*
__device__ void doublepointsCrossover(const int *src, int *dst, int tx, curandState localState, int
        parent1, int parent2)
{
	int i = 0;
	unsigned int point1;
	unsigned int point2;
	int offset = tx * gpuEvoPrms.CHROMOSOME;

	point1 = (unsigned int)(curand_uniform(&localState) * (gpuEvoPrms.CHROMOSOME));
	point2 = (unsigned int)(curand_uniform(&localState) * (gpuEvoPrms.CHROMOSOME));

    if (point1 > point2) swap(point1, point2);

    for (; i < point1; ++i)
    {
		dst[i + offset] = src[parent1 * gpuEvoPrms.CHROMOSOME + i];
    }
    for (; i < point2; ++i)
    {
		dst[i + offset] = src[parent2 * gpuEvoPrms.CHROMOSOME + i];
    }
    for (; i < gpuEvoPrms.CHROMOSOME; ++i)
    {
		dst[i + offset] = src[parent1 * gpuEvoPrms.CHROMOSOME + i];
    }
}
*/


/**
 * @param[in] src		Population where current-generation data is stored.
 * @param[out] dst		Populat ion where next-generation data is stored.
 * @param[in] parent1	Fitness of parent 1
 * @param[in] parent2	Fitness of parent 2
 * @return void
 */
/*
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
	s_parent[tx + gpuEvoPrms.POPSIZE] = parent2[tx];
	__syncthreads();

	curandState localState = dev_States[tx + gpuEvoPrms.POPSIZE * gen];
	doublepointsCrossover(src, dst, tx, localState, s_parent[tx], s_parent[tx + gpuEvoPrms.POPSIZE]);
	// singlepointCrossover(src, dst, tx, localState, s_parent[tx], s_parent[tx + gpuEvoPrms.POPSIZE]);
	__syncthreads();
}
*/

/*
__global__ void mutation(int *population, curandState *dev_States, const int gen)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = dev_States[id + gpuEvoPrms.POPSIZE * gen];

	if (curand_uniform(&localState) < gpuEvoPrms.MUTATION_RATE)
	{
		population[id] ^= 1;
	}
}
*/

__global__ void dev_show(int *population, int *fitness, int *sortedfitness, int *parent1, int *parent2)
{
	int tx = threadIdx.x;
	if (gpuEvoPrms.POPSIZE - gpuEvoPrms.NUM_OF_ELITE <= tx)
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
    printf("%d\n", gpuEvoPrms.POPSIZE);
    printf("%d\n", gpuEvoPrms.CHROMOSOME);
    printf("%d\n", gpuEvoPrms.NUM_OF_GENERATIONS);
    printf("%d\n", gpuEvoPrms.NUM_OF_ELITE);
    printf("%d\n", gpuEvoPrms.TOURNAMENT_SIZE);
    printf("%d\n", gpuEvoPrms.NUM_OF_CROSSOVER_POINTS);
    printf("%f\n", gpuEvoPrms.MUTATION_RATE);
}


