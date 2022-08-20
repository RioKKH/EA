#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "Parameters.hpp"

enum class PARENTS_e : short
{
    MALE   = 0,
    FEMALE = 1,
};

void copyToDevice(EvolutionParameters cpuEvoPrms);



__global__ void evaluation(int *population, int *fitness);

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

#endif // CUDA_KERNELS_H

