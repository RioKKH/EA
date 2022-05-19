#ifndef MISC_H
#define MISC_H

#include <stdio.h>
#include <stdlib.h>
// #include <bitset>

// #include <curand.h>
// #include <curand_kernel.h>
// #include <thrust/random.h>
// #include <thrust/generate.h>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #include <thrust/device_ptr.h>

#include "CUDAKernels.h"

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

#define NUM_OF_GENERATIONS 16 
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


int my_rand(void);

void initializePopulationOnCPU(int *population);

void showPopulationOnCPU(int *population,
                         int *fitness,
                         int *parent1,
                         int *parent2);

#endif // MISC_H
