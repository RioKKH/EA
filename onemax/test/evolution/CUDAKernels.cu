#include "Random123/philox.h"

#include "CUDAKernels.h"

typedef r123::Philox2x32 RNG_2x32;
typedef r123::Philox4x32 RNG_4x32;

__device__ RNG_2x32::ctr_type generateTwoRndValues(unsigned int key,
                                                   unsigned int counter);

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
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
}

