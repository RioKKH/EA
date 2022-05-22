#ifndef MISC_H
#define MISC_H

#include <stdio.h>
#include <stdlib.h>
#include "Parameters.hpp"
// #include "CUDAKernels.h"

/**
  * @def POPSIZE
  * @brief Size of the population
  */
// #define POPSIZE 16
// #define POPSIZE 512

/** @def CHROMOSOME
  * @brief Size of the chromosome
  */
// #define CHROMOSOME 16
// #define CHROMOSOME 512

// #define NUM_OF_GENERATIONS 16 
// #define NUM_OF_ELITE 4
// #define TOURNAMENT_SIZE 3
// #define NUM_OF_CROSSOVER_POINTS 1
// #define MUTATION_RATE 0.05

// #define N (POPSIZE * CHROMOSOME)
// #define Nbytes (N*sizeof(int))
// #define NT CHROMOSOME
// #define NB POPSIZE
// #define NT (256)
// #define NB (N / NT) // 1より大きくなる

int my_rand(void);

void initializePopulationOnCPU(int *population, Parameters &prms);

void showPopulationOnCPU(int *population,
                         int *fitness,
                         int *parent1,
                         int *parent2,
                         Parameters &prms);

#endif // MISC_H
