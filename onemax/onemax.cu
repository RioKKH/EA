#include <stdio.h>
#include <stdlib.h>
#include <bitset>

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

/**
  * @def POPSIZE
  * @brief Size of the population
  */
#define POPSIZE 8
// #define POPSIZE 512

/** @def CHROMOSOME
  * @brief Size of the chromosome
  */
#define CHROMOSOME 8
// #define CHROMOSOME 512

#define NUM_OF_GENERATIONS 5 
#define MUTATION_RATE 0.05
#define TOURNAMENT_SIZE 3
#define NUM_OF_CROSSOVER_POINTS 1
#define ELITISM true

#define N (POPSIZE * CHROMOSOME)
#define Nbytes (N*sizeof(int))
#define NT CHROMOSOME
#define NB POPSIZE
// #define NT (256)
// #define NB (N / NT) // 1より大きくなる

enum PARENTS {
	MALE   = 0,
	FEMALE = 1,
};

#define CUDA_CALL(x) do										\
{                                                           \
	if((x) != cudaSuccess)									\
	{														\
		printf("Error at %s:%d\n", __FILE__, __LINE__);		\
		return EXIT_FAILURE;                                \
	}														\
} while (0)													\

#define CURAND_CALL(x) do									\
{                                                           \
	if((x) != CURAND_STATUS_SUCCESS)                        \
	{                                                       \
		printf("Error at %s:%d\n", __FILE__, __LINE__);		\
		return EXIT_FAILURE;                                \
	}                                                       \
} while (0)                                                 \


__host__ __device__ int getBest()
{
	return 0;
}

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
	for (int i = 0; i < TOURNAMENT_SIZE; ++i)
	{
		// printf("getBestIndividual:%d\n", i);
		if (fitness[i] > best)
		{
			best = fitness[i];
			best_index = i;
		}
	}

	return best_index;
}

__device__ int tournamentSelection(const int *fitness, curandState *dev_States, const int &ix, PARENTS mf, int gen)
{
	int best_id;
	int tournament_individuals[TOURNAMENT_SIZE];
	int tournament_fitness[TOURNAMENT_SIZE];
	unsigned int random_id;
	unsigned int offset = (POPSIZE * TOURNAMENT_SIZE) * mf;

	for (int i = 0; i < TOURNAMENT_SIZE; ++i) {
		// curand_uniform returns random number uniformly distributed between (0, 1].
		curandState localState = dev_States[ix * TOURNAMENT_SIZE + i + offset + POPSIZE * gen];
		// curandState localState = dev_States[ix * TOURNAMENT_SIZE + i + offset]; // w/o generation
		random_id = (unsigned int)(curand_uniform(&localState) * (POPSIZE));
		tournament_individuals[i] = random_id;
		tournament_fitness[i] = fitness[random_id];

	}
	best_id = getBestIndividual(tournament_fitness);

	return tournament_individuals[best_id];
}


__global__ void selection(int* fitness, curandState *dev_States, int* parent1, int* parent2, int gen)
{
	int tx = threadIdx.x;

	// printf("tx:%d\n", tx);
	parent1[tx] = tournamentSelection(fitness, dev_States, tx, MALE, gen);
	parent2[tx] = tournamentSelection(fitness, dev_States, tx, FEMALE, gen);
}

__device__ void singlepointCrossover(const int *src, int *dst, int tx, curandState localState, int parent1, int parent2) 
{
	int i = 0;
	unsigned int point1;
	int offset = tx * CHROMOSOME;

	point1 = (unsigned int)(curand_uniform(&localState) * (POPSIZE));
	for (i = 0; i < point1; ++i)
	{
		dst[i + offset] = src[parent1 * CHROMOSOME + i];
	}
	for (; i < CHROMOSOME; ++i)
	{
		dst[i + offset] = src[parent2 * CHROMOSOME + i];
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
		const int *src, int *dst,
		curandState *dev_States,
		const int *parent1, const int *parent2,
		const int gen)
{
	int tx = threadIdx.x;

	// singlepointCrossover(src, dst, tx, dev_States[tx], parent1[tx], parent2[tx]);

	extern __shared__ volatile int s_parent[];
	s_parent[tx] = parent1[tx];
	s_parent[tx + POPSIZE] = parent2[tx];
	__syncthreads();

	curandState localState = dev_States[tx + POPSIZE * gen];
	singlepointCrossover(src, dst, tx, localState, s_parent[tx], s_parent[tx + CHROMOSOME]);
	__syncthreads();
}

__global__ void mutation()
{
}

int my_rand(void)
{
    static thrust::default_random_engine rng;
    static thrust::uniform_int_distribution<int> dist(0, 1);

    return dist(rng);
}

void initializePopulationOnCPU(int *population)
{
    thrust::generate(population, population + N, my_rand);

#ifdef _DEBUG
    for (int i=0; i<POPSIZE; ++i)
	{
		for (int j=0; j<CHROMOSOME; ++j)
		{
			std::cout << population[i * POPSIZE + j];
		}
		std::cout << std::endl;
	}
#endif // _DEBUG
}

void showPopulationOnCPU(int *population, int *fitness, int *parent1, int *parent2)
{
	for (int i = 0; i < POPSIZE; ++i)
	{
		printf("%d,%d,%d,", fitness[i], parent1[i], parent2[i]);
		for (int j = 0; j < CHROMOSOME; ++j)
		{
			printf("%d", population[i * POPSIZE + j]);
		}
		printf("\n");
	}
}

int main()
{
    //- GPU用変数 idata: 入力、odata: 出力(総和) --------------------------------------------------
    int *pdev_PopulationOdd;
    int *pdev_PopulationEven;
    int *pdev_Parent1;
    int *pdev_Parent2;
	thrust::device_vector<int> dev_Fitnesses(POPSIZE);
	thrust::device_vector<int> dev_Ranks(POPSIZE);

	int *pdev_Fitness = thrust::raw_pointer_cast(&dev_Fitnesses[0]);
	int *pdev_Ranks = thrust::raw_pointer_cast(&dev_Ranks[0]);
	thrust::sequence(dev_Ranks.begin(), dev_Ranks.end());

    cudaMalloc((void **)&pdev_PopulationOdd, Nbytes);
    cudaMalloc((void **)&pdev_PopulationEven, Nbytes);
    cudaMalloc((void **)&pdev_Parent1, NB * sizeof(int));
    cudaMalloc((void **)&pdev_Parent2, NB * sizeof(int));

    //- CPU用変数 ---------------------------------------------------------------------------------
    int *phost_Population;
	int *phost_Fitness;
	int *phost_Ranks;
	int *phost_Parent1;
	int *phost_Parent2;

    phost_Fitness = (int *)malloc(POPSIZE * sizeof(int));
	phost_Ranks   = (int *)malloc(POPSIZE * sizeof(int));
	phost_Parent1 = (int *)malloc(POPSIZE * sizeof(int));
	phost_Parent2 = (int *)malloc(POPSIZE * sizeof(int));

	//- 乱数用変数 --------------------------------------------------------------------------------
	curandState *dev_TournamentStates;
	cudaMalloc((void **)&dev_TournamentStates, POPSIZE * TOURNAMENT_SIZE * 2 * NUM_OF_GENERATIONS * sizeof(curandState));
	cudaDeviceSynchronize();

	curandState *dev_CrossoverStates;
	cudaMalloc((void **)&dev_CrossoverStates, POPSIZE * NUM_OF_CROSSOVER_POINTS * NUM_OF_GENERATIONS * sizeof(curandState));
	cudaDeviceSynchronize();

	//- Preparation -------------------------------------------------------------------------------

    // CPU側でデータを初期化してGPUへコピー
    phost_Population = (int *)malloc(Nbytes);
    initializePopulationOnCPU(phost_Population);
    cudaMemcpy(pdev_PopulationEven, phost_Population, Nbytes, cudaMemcpyHostToDevice);

	// --------------------------------
	// Main loop
	// --------------------------------

	// initialize random numbers array for tournament selection
	// 乱数はトーナメントセレクションで用いられるので、個体の数x2だけあれば良い
	setup_kernel<<<POPSIZE * NUM_OF_GENERATIONS, TOURNAMENT_SIZE * 2>>>(dev_TournamentStates);
	cudaDeviceSynchronize();

	setup_kernel<<<POPSIZE * NUM_OF_GENERATIONS, NUM_OF_CROSSOVER_POINTS>>>(dev_CrossoverStates);
	cudaDeviceSynchronize();

	evaluation<<<NB, NT, NT*sizeof(int)>>>(pdev_PopulationEven, pdev_Fitness);
	cudaDeviceSynchronize();

	for (int gen = 0; gen < NUM_OF_GENERATIONS; ++gen)
	{
		printf("#####Gen: %d #######\n", gen);

		selection<<<1, POPSIZE>>>(
				pdev_Fitness,
				dev_TournamentStates,
				pdev_Parent1,
				pdev_Parent2,
				gen);
		cudaDeviceSynchronize();

		if (gen % 2 == 0) // Even
		{
			crossover<<<1, POPSIZE, POPSIZE * sizeof(int) * 2>>>(
					pdev_PopulationEven,
					pdev_PopulationOdd,
					dev_CrossoverStates,
					pdev_Parent1,
					pdev_Parent2,
					gen);
			cudaDeviceSynchronize();

			evaluation<<<NB, NT, NT*sizeof(int)>>>(pdev_PopulationOdd, pdev_Fitness);
			cudaDeviceSynchronize();
		}
		else // Odd
		{
			crossover<<<1, POPSIZE, POPSIZE * sizeof(int) * 2>>>(
					pdev_PopulationOdd,
					pdev_PopulationEven,
					dev_CrossoverStates,
					pdev_Parent1,
					pdev_Parent2,
					gen);
			cudaDeviceSynchronize();

			evaluation<<<NB, NT, NT*sizeof(int)>>>(pdev_PopulationEven, pdev_Fitness);
			cudaDeviceSynchronize();
		}
#ifdef _DEBUG
		cudaMemcpy(phost_Fitness, pdev_Fitness, POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(phost_Parent1, pdev_Parent1, POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(phost_Parent2, pdev_Parent2, POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
		if (gen % 2 == 0)
		{
			cudaMemcpy(phost_Population, pdev_PopulationOdd, Nbytes, cudaMemcpyDeviceToHost);
		}
		else
		{
			cudaMemcpy(phost_Population, pdev_PopulationEven, Nbytes, cudaMemcpyDeviceToHost);
		}
		showPopulationOnCPU(phost_Population, phost_Fitness, phost_Parent1, phost_Parent2);
#endif // _DEBUG
	}

    cudaMemcpy(phost_Fitness, pdev_Fitness, POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(phost_Parent1, pdev_Parent1, POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(phost_Parent2, pdev_Parent2, POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(phost_Population, pdev_PopulationOdd, Nbytes, cudaMemcpyDeviceToHost);

	// cudaMemcpy(phost_Ranks, pdev_Ranks, POPSIZE * sizeof(int), cudaMemcpyHostToHost);

    cudaFree(pdev_PopulationOdd);
    cudaFree(pdev_PopulationEven);

    free(phost_Population);
	free(phost_Fitness);
	free(phost_Ranks);
	free(phost_Parent1);
	free(phost_Parent2);

    return 0;
}
