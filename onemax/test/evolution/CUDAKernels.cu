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
        populationData->population[i] = randomValues.v[0];

        i += stride;
        if (i < nGenes)
        {
            populationData->population[i] = randomValues.v[1];
        }
        i += stride;
    }

    // Zero fitness values
    i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < populationData->populationSize)
    {
     populationData->fitness[i] = 0.0f;
     i += stride;
    }
} // end of cudaGeneratePopulationKernel


__global__ void evaluation(PopulationData* populationData)
{
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int stride;
    extern __shared__ volatile int s_idata[];
    s_idata[tx] = populationData->population[i];
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
        populationData->fitness[blockIdx.x] = s_idata[0];
    }
}



