#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "Population.h"
#include "Parameters.h"

enum class PARENTS_e : short
{
    MALE   = 0,
    FEMALE = 1,
};

void copyToDevice(EvolutionParameters cpuEvoPrms);

/**
 * Check and report CUDA errors.
 * @param [in] sourceFileName    - Source file where the error happened.
 * @param [in] sourceLineNumber  - Line where the error happened.
 */
void checkAndReportCudaError(const char* sourceFileName,
                             const int   sourceLineNumber);


__global__ void evaluation(PopulationData* populationData);
//__global__ void evaluation(int *population, int *fitness);

__host__ __device__ int getBestIndividual(const int *fitness);

__device__ int tournamentSelection(const int *fitness, unsigned int *rand,
                                   const int &ix, PARENTS_e mf,
                                   int gen,
                                   int *tournament_individual,
                                   int *tournament_fitness);

__device__ void selection(int *fitness, int *sortedid,
                          unsigned int *rand1, unsigned int *rand2,
                          int *parent1, int *parent2,
                          int gen,
                          int *tournament_individual,
                          int *tournament_fitness);

__device__ void singlepointCrossover(const int *src, int *dst, int tx,
                                     unsigned int randomSeed,
                                     int parent1,
                                     int parent2);

__device__ void swap(unsigned int &point1,
                     unsigned int &point2);

__device__ void doublepointsCrossover(const int *src, int *dst, int tx,
                                      unsigned int randomSeed,
                                      int parent1,
                                      int parent2);

__global__ void crossover(const int *src, int *dst,
                          unsigned int randomSeed,
                          const int *parent1, const int *parent2,
                          const int gen);

__global__ void mutation (int *population,
                          unsigned int randomSeed,
                          const int gen);

__global__ void dev_show(int *population, int *fitness, int *sortedfitness,
                         int *parent1, int *parent2);

__global__ void dev_prms_show(void);

__global__ void cudaCallRandomNumber(unsigned int randomSeed);

//__global__ void cudaGenerateFirstPopulationKernel(PopulationData* populationDataEven,
//                                                  PopulationData* populationDataOdd,
__global__ void cudaGenerateFirstPopulationKernel(PopulationData* populationData,
                                                  unsigned int    randomSeed);

/**
 * Genetic manipulation (Selection, Crossover, Mutation)
 * @param [in]  populationDataEven    - Even-numbered generations of population.
 * @param [in]  populationDataOdd     - Odd-numbered generations of population.
 * @param [in]  randomSeed            - Random seed.
 */
__global__ void cudaGeneticManipulationKernel(PopulationData* populationDataEven,
                                              PopulationData* populationDataOdd,
                                              unsigned int    randomSeed);

#endif // CUDA_KERNELS_H

