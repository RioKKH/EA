#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "Random123/philox.h"
#include "CUDAKernels.h"
#include "Parameters.h"
#include "Population.h"

typedef r123::Philox2x32 RNG_2x32;
typedef r123::Philox4x32 RNG_4x32;

__device__ RNG_2x32::ctr_type generateTwoRndValues(unsigned int key,
                                                   unsigned int counter);

__constant__ float RANDMAX = 4294967295.0f;
__constant__ EvolutionParameters gpuEvoPrms;


void copyToDevice(EvolutionParameters cpuEvoPrms)
{
#ifdef _DEBUG
    printf("copyToDevice %d\n", cpuEvoPrms.POPSIZE);
#endif // _DEBUG
    cudaMemcpyToSymbol(gpuEvoPrms,
                       &cpuEvoPrms, 
                       sizeof(EvolutionParameters));
}


void checkAndReportCudaError(const char* sourceFileName,
                             const int   sourceLineNumber)
{
    const cudaError_t cudaError = cudaGetLastError();

    if (cudaError != cudaSuccess)
    {
        fprintf(stderr,
                "Error in the CUDA routine: \"%s\"\nFile name: %s\nLine number: %d\n",
                cudaGetErrorString(cudaError),
                sourceFileName,
                sourceLineNumber);

        exit(EXIT_FAILURE);
    }
}


/*
inline __device__ int getIndex(unsigned int chromosomeIdx,
                               unsigned int geneIdx);
*/

inline __device__ RNG_2x32::ctr_type generateTwoRndValues(unsigned int key,
                                                          unsigned int counter)
{
    RNG_2x32 rng;

    return rng({0, counter}, {key});
} // end of TwoRandomINTs


/*
inline __device__ int getIndex(unsigned int chromosomeIdx,
                               unsigned int geneIdx)
{
    return (chromosomeIdx * gpuEvolutionParameters.chromosomeSize + geneIdx);
}
*/


__global__ void cudaCallRandomNumber(unsigned int randomSeed)
{
    // for (int i = 0; i < 10; ++i)
    // {
    //     printf("%d\n", i);
    // }
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("%d\n", idx);

    const RNG_2x32::ctr_type randomValues = generateTwoRndValues(idx, randomSeed);
    printf("%d, %d\n", randomValues.v[0], randomValues.v[1]);
}


__global__ void cudaGenerateFirstPopulationKernel(PopulationData* populationData,
                                                  unsigned int    randomSeed)
{
    size_t i      = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    const int nGenes = populationData->chromosomeSize * populationData->populationSize;

    while (i < nGenes)
    {
        const RNG_2x32::ctr_type randomValues = generateTwoRndValues(i, randomSeed);
        populationData->population[i] = randomValues.v[0] % 2;

        i += stride;
        if (i < nGenes)
        {
            populationData->population[i] = randomValues.v[1] % 2;
        }
        i += stride;
    }

    //- Zero fitness values
    i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < populationData->populationSize)
    {
        populationData->fitness[i] = 0.0f;
        i += stride;
    }
} // end of cudaGeneratePopulationKernel


__global__ void evaluation(PopulationData* populationData)
{
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int tx   = threadIdx.x;
    int stride;

    // 共有メモリの配列要素数をカーネル起動時に動的に決定
    extern __shared__ volatile int s_idata[];

    s_idata[tx] = populationData->population[idx];
    __syncthreads();

    for (stride = blockDim.x/2; stride >= 1; stride >>=1)
    // for (stride = 1; stride <= blockDim.x/2; stride <<= 1)
    {
        if (tx < stride)
        // if (tx % (2 * stride) == 0)
        {
            s_idata[tx] += s_idata[tx + stride];
            // s_idata[tx] = s_idata[tx] + s_idata[tx + stride];
        }
        __syncthreads();
    }

    if (tx == 0)
    {
        populationData->fitness[blockIdx.x] = s_idata[tx];
    }
}

__global__ void pseudo_elitism(PopulationData* populationData)
{
    int numOfEliteIdx     = blockIdx.x;  // size of NUM_OF_ELITE
    int localFitnessIdx   = threadIdx.x; // size of POPULATION / NUM_OF_ELITE
    int globalFitnessIdx  = threadIdx.x + blockIdx.x * blockDim.x; // size of POPULATION
    const int OFFSET      = blockDim.x * gridDim.x / 2;

    extern __shared__ volatile int s_fitness[];

    // 共有メモリの共有範囲は同一ブロック内のスレッド群
    if (globalFitnessIdx < OFFSET)
    {
        s_fitness[localFitnessIdx]          = populationData->fitness[globalFitnessIdx];
        s_fitness[localFitnessIdx + OFFSET] = globalFitnessIdx;
    }
    __syncthreads();

    for (int stride = OFFSET/2; stride >= 1; stride >>= 1)
    {
        if (localFitnessIdx < stride)
        {
            unsigned int index = (s_fitness[localFitnessIdx] >= s_fitness[localFitnessIdx + stride]) ? localFitnessIdx : localFitnessIdx + stride;
            s_fitness[localFitnessIdx] = s_fitness[index];
            s_fitness[localFitnessIdx + OFFSET] = s_fitness[index + OFFSET];
        }
        __syncthreads();
    }

    if (localFitnessIdx == 0)
    {
        populationData->elitesIdx[numOfEliteIdx] = s_fitness[localFitnessIdx + blockDim.x * gridDim.x / 2];
    }
}


__host__ __device__ int getBestIndividual()
{
}


__global__ void cudaGeneticManipulationKernel(PopulationData* populationDataEven,
                                              PopulationData* populationDataOdd,
                                              unsigned int    randomSeed)
{
    int geneIdx       = threadIdx.x;
    int chromosomeIdx = blockIdx.x;
    int total_geneIdx = threadIdx.x + blockIdx.x * blockDim.x;
    // int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tx  = threadIdx.x;

    // Init randome number generator
    RNG_4x32 rng_4x32;
    RNG_4x32::key_type key = {{static_cast<unsigned int>(geneIdx),
                               static_cast<unsigned int>(chromosomeIdx)}};
    RNG_4x32::ctr_type counter = {{0, 0, randomSeed, 0xbeeff00d}};
    RNG_4x32::ctr_type randomValues;

    // Produce new offspring
    // __shared__ int parent1Idx[gpuEvoPrms.POPSIZE];
    // __shared__ int parent2Idx[gpuEvoPrms.POPSIZE];
    // __shared__ int tournamentFitness[gpuEvoPrms.TOUNAMENT_SIZE];

    //- selection

    //- crossover

    //- mutation
}

__device__ void tournamentSelection(PopulationData* populationData)
{
}

__global__ void selection(PopulationData* populationDataEven,
                          PopulationData* populationDataOdd,
                          unsigned int randomSeed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tx  = threadIdx.x;
    // if (gpuEvoPrms.POPSIZE - gpu
}



