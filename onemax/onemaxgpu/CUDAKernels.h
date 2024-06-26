#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

// #include <curand.h>
// #include <curand_kernel.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "Parameters.hpp"

enum PARENTS {
    MALE    = 0,
    FEMALE  = 1,
};

void copyToDevice(EvolutionParameters cpuEvoPrms);

/*
__global__ void setup_kernel(curandState *state);

__global__ void generate_kernel(curandState *state, float *result);
*/

__global__ void evaluation(int *population, int *fitness);

__host__ __device__ int getBestIndividual(const int *fitness);

__device__ int tournamentSelection(const int *fitness,
                                   // curandState *dev_States,
                                   unsigned int *rand,
                                   const int &ix,
                                   PARENTS mf,
                                   int gen,
                                   int *tournament_individual,
                                   int *tournament_fitness);

__global__ void selection(int *fitness,
                          int *sortedid,
                          // curandState *dev_States,
                          float *rand1,
                          float *rand2,
                          int *parent1,
                          int *parent2,
                          int gen,
                          int *tournament_individual,
                          int *tournament_fitness);

/*
__device__ void singlepointCrossover(const int *src,
                                     int *dst,
                                     int tx, 
                                     // curandState localState,
                                     unsigned int randomSeed,
                                     int parent1,
                                     int parent2);

__device__ void swap(unsigned int &point1,
                     unsigned int &point2);

__device__ void doublepointsCrossover(const int *src,
                                     int *dst,
                                     int tx, 
                                     // curandState localState,
                                     unsigned int randomSeed,
                                     int parent1,
                                     int parent2);

__global__ void crossover(const int *src,
                          int *dst,
                          // curandState *dev_States,
                          unsigned int randomSeed,
                          const int *parent1,
                          const int *parent2,
                          const int gen);

__global__ void mutation(int *population,
                         // curandState *dev_States,
                         unsigned int randomSeed,
                         const int gen);
*/

__global__ void dev_show(int *sortedfitness, int *sortedid);

__global__ void dev_prms_show(void);

#endif // CUDA_KERNELS_H


