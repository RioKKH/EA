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
#define POPSIZE 16
// #define POPSIZE 512

/** @def CHROMOSOME
  * @brief Size of the chromosome
  */
#define CHROMOSOME 16
// #define CHROMOSOME 512

#define NUM_OF_GENERATIONS 20 
#define NUM_OF_ELITE 4
#define TOURNAMENT_SIZE 3
#define NUM_OF_CROSSOVER_POINTS 1
#define MUTATION_RATE 0.05

#define N (POPSIZE * CHROMOSOME)
#define Nbytes (N*sizeof(int))
// #define NT CHROMOSOME
// #define NB POPSIZE
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
	// TODO 恐らくtounamentSelection
	// がElitismに対応していない。具体的にはsortedidに対応していないことが、バグの原因になっている
	best_id = getBestIndividual(tournament_fitness);

	return tournament_individuals[best_id];
}

__global__ void selection(int* fitness, int* sortedid, curandState *dev_States, int* parent1, int* parent2, int gen)
{
	int tx = threadIdx.x;

	if (POPSIZE - NUM_OF_ELITE <= tx)
	{
		parent1[tx] = sortedid[tx];
		parent2[tx] = sortedid[tx];
	}
	else
	{
		parent1[tx] = tournamentSelection(fitness, dev_States, tx, MALE, gen);
		parent2[tx] = tournamentSelection(fitness, dev_States, tx, FEMALE, gen);
	}
	// printf("%d,%d,%d,%d\n", tx, fitness[tx], parent1[tx], parent2[tx]);
}

__device__ void singlepointCrossover(const int *src, int *dst, int tx, curandState localState, int parent1, int parent2) 
{ 
	int i = 0;
	unsigned int point1;
	int offset = tx * CHROMOSOME;

	point1 = (unsigned int)(curand_uniform(&localState) * (CHROMOSOME));
	// point1 = (unsigned int)(curand_uniform(&localState) * (POPSIZE));
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
		const int *src,
		int *dst,
		curandState *dev_States,
		const int *parent1,
		const int *parent2,
		const int gen)
{
	int tx = threadIdx.x;

	// singlepointCrossover(src, dst, tx, dev_States[tx], parent1[tx], parent2[tx]);

	extern __shared__ volatile int s_parent[];
	s_parent[tx] = parent1[tx];
	s_parent[tx + POPSIZE] = parent2[tx];
	__syncthreads();

	curandState localState = dev_States[tx + POPSIZE * gen];
	singlepointCrossover(src, dst, tx, localState, s_parent[tx], s_parent[tx + POPSIZE]);
	__syncthreads();
}

__global__ void mutation(int *population, curandState *dev_States, const int gen)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState localState = dev_States[id + POPSIZE * gen];

	if (curand_uniform(&localState) < MUTATION_RATE)
	{
		population[id] ^= 1;
	}
}

__global__ void dev_show(int *population, int *fitness, int *sortedfitness, int *parent1, int *parent2)
{
	int tx = threadIdx.x;
	if (POPSIZE - NUM_OF_ELITE <= tx)
	{
		printf("%d,%d,%d,%d\n", tx, sortedfitness[tx], parent1[tx], parent2[tx]);
	}
	else {
		printf("%d,%d,%d,%d\n", tx, fitness[tx], parent1[tx], parent2[tx]);
	}
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
		printf("%d,%d,%d,%d,", i, fitness[i], parent1[i], parent2[i]);
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
    // int *pdev_PopulationOdd;
    // int *pdev_PopulationEven;
    // int *pdev_Parent1;
    // int *pdev_Parent2;
	thrust::device_vector<int> dev_PopulationOdd(N);
	thrust::device_vector<int> dev_PopulationEven(N);
	thrust::device_vector<int> dev_Parent1(POPSIZE);
	thrust::device_vector<int> dev_Parent2(POPSIZE);
	thrust::device_vector<int> dev_Fitnesses(POPSIZE);
	thrust::device_vector<int> dev_SortedFitnesses(POPSIZE);
	thrust::device_vector<int> dev_SortedId(POPSIZE);

	int *pdev_PopulationOdd = thrust::raw_pointer_cast(&dev_PopulationOdd[0]);
	int *pdev_PopulationEven = thrust::raw_pointer_cast(&dev_PopulationEven[0]);
	int *pdev_Parent1 = thrust::raw_pointer_cast(&dev_Parent1[0]);
	int *pdev_Parent2 = thrust::raw_pointer_cast(&dev_Parent2[0]);
	int *pdev_Fitness = thrust::raw_pointer_cast(&dev_Fitnesses[0]);
	int *pdev_SortedFitness = thrust::raw_pointer_cast(&dev_SortedFitnesses[0]);
	int *pdev_SortedId = thrust::raw_pointer_cast(&dev_SortedId[0]);

    // cudaMalloc((void **)&pdev_PopulationOdd, Nbytes);
    // cudaMalloc((void **)&pdev_PopulationEven, Nbytes);
    // cudaMalloc((void **)&pdev_Parent1, POPSIZE * sizeof(int));
    // cudaMalloc((void **)&pdev_Parent2, POPSIZE * sizeof(int));

    //- CPU用変数 ---------------------------------------------------------------------------------
    int *phost_Population;
	int *phost_Fitness;
	int *phost_SortedId;
	int *phost_Parent1;
	int *phost_Parent2;

    phost_Fitness       = (int *)malloc(POPSIZE * sizeof(int));
	phost_SortedId   = (int *)malloc(POPSIZE * sizeof(int));
	phost_Parent1       = (int *)malloc(POPSIZE * sizeof(int));
	phost_Parent2       = (int *)malloc(POPSIZE * sizeof(int));

	//- 乱数用変数 --------------------------------------------------------------------------------
	curandState *dev_TournamentStates;
	cudaMalloc((void **)&dev_TournamentStates, POPSIZE * TOURNAMENT_SIZE * 2 * NUM_OF_GENERATIONS * sizeof(curandState));

	curandState *dev_CrossoverStates;
	cudaMalloc((void **)&dev_CrossoverStates, POPSIZE * NUM_OF_CROSSOVER_POINTS * NUM_OF_GENERATIONS * sizeof(curandState));

	curandState *dev_MutationStates;
	cudaMalloc((void **)&dev_MutationStates, POPSIZE * CHROMOSOME * NUM_OF_GENERATIONS * sizeof(curandState));

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

	setup_kernel<<<POPSIZE * NUM_OF_GENERATIONS, CHROMOSOME>>>(dev_MutationStates);
	cudaDeviceSynchronize();

	evaluation<<<POPSIZE, CHROMOSOME, CHROMOSOME*sizeof(int)>>>(pdev_PopulationEven, pdev_Fitness);
	cudaDeviceSynchronize();

	dev_show<<<1, POPSIZE>>>(pdev_PopulationEven, pdev_Fitness, pdev_SortedFitness, pdev_Parent1, pdev_Parent2);
	cudaDeviceSynchronize();

	// mutation<<<POPSIZE, CHROMOSOME>>>(pdev_PopulationEven, dev_MutationStates, 0);

	for (int gen = 0; gen < NUM_OF_GENERATIONS; ++gen)
	{
		printf("#####Gen: %d #######\n", gen);

		thrust::copy(thrust::device, dev_Fitnesses.begin(), dev_Fitnesses.end(), dev_SortedFitnesses.begin());
		thrust::sequence(dev_SortedId.begin(), dev_SortedId.end());
		thrust::sort_by_key(dev_SortedFitnesses.begin(), dev_SortedFitnesses.end(), dev_SortedId.begin()); 

		selection<<<1, POPSIZE>>>(
		// selection<<<N/POPSIZE, POPSIZE>>>(
				pdev_Fitness,
				pdev_SortedId,
				dev_TournamentStates,
				pdev_Parent1,
				pdev_Parent2,
				gen);
		cudaDeviceSynchronize();

		dev_show<<<1, POPSIZE>>>(pdev_PopulationEven, pdev_Fitness, pdev_SortedFitness, pdev_Parent1, pdev_Parent2);
		cudaDeviceSynchronize();

		if (gen % 2 == 0) // Even
		{
			crossover<<<1, POPSIZE, POPSIZE * sizeof(int) * 2>>>(
			// crossover<<<N/POPSIZE, POPSIZE, POPSIZE * sizeof(int) * 2>>>(
					pdev_PopulationEven,
					pdev_PopulationOdd,
					dev_CrossoverStates,
					pdev_Parent1,
					pdev_Parent2,
					gen);
			cudaDeviceSynchronize();

			mutation<<<POPSIZE, CHROMOSOME>>>(pdev_PopulationOdd, dev_MutationStates, gen);
			cudaDeviceSynchronize();

			evaluation<<<POPSIZE, CHROMOSOME, CHROMOSOME*sizeof(int)>>>(pdev_PopulationOdd, pdev_Fitness);
			cudaDeviceSynchronize();
		}
		else // Odd
		{
			crossover<<<1, POPSIZE, POPSIZE * sizeof(int) * 2>>>(
			// crossover<<<N/POPSIZE, POPSIZE, POPSIZE * sizeof(int) * 2>>>(
					pdev_PopulationOdd,
					pdev_PopulationEven,
					dev_CrossoverStates,
					pdev_Parent1,
					pdev_Parent2,
					gen);
			cudaDeviceSynchronize();

			mutation<<<POPSIZE, CHROMOSOME>>>(pdev_PopulationEven, dev_MutationStates, gen);
			cudaDeviceSynchronize();

			evaluation<<<POPSIZE, CHROMOSOME, CHROMOSOME*sizeof(int)>>>(pdev_PopulationEven, pdev_Fitness);
			cudaDeviceSynchronize();
		}
#ifdef _DEBUG
		cudaMemcpy(phost_Fitness,  pdev_Fitness,  POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(phost_SortedId, pdev_SortedId, POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(phost_Parent1,  pdev_Parent1,  POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(phost_Parent2,  pdev_Parent2,  POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
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

	// cudaMemcpy(phost_Ranks, pdev_SortedId, POPSIZE * sizeof(int), cudaMemcpyHostToHost);

    // cudaFree(pdev_PopulationOdd);
    // cudaFree(pdev_PopulationEven);

    free(phost_Population);
	free(phost_Fitness);
	free(phost_SortedId);
	free(phost_Parent1);
	free(phost_Parent2);

    return 0;
}
