#include <stdio.h>
#include <stdlib.h>

#include "CUDAKernels.h"
#include "Misc.h"


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
    for (int i = 0; i < POPSIZE; ++i)
	{
		for (int j = 0; j < CHROMOSOME; ++j)
		{
			printf("%d", population[i * CHROMOSOME + j]);
		}
		printf("\n");
	}
	std::cout << "end of initialization" << std::endl;
#endif // _DEBUG
}

void showPopulationOnCPU(int *population, int *fitness, int *parent1, int *parent2)
{
	for (int i = 0; i < POPSIZE; ++i)
	{
		printf("%d,%d,%d,%d,", i, fitness[i], parent1[i], parent2[i]);
		for (int j = 0; j < CHROMOSOME; ++j)
		{
			printf("%d", population[i * CHROMOSOME + j]);
		}
		printf("\n");
	}
}
