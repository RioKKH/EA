#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "Random123/philox.h"
#include "Random123/uniform.hpp"
#include "CUDAKernels.h"
#include "Parameters.h"
#include "Population.h"

typedef r123::Philox2x32 RNG_2x32;
typedef r123::Philox4x32 RNG_4x32;

__device__ RNG_2x32::ctr_type generateTwoRndValues(unsigned int key,
                                                   unsigned int counter);

// __constant__ long RANDMAX = 4294967295;
__constant__ std::int64_t RANDMAX = 4294967295;
// __constant__ std::int64_t RANDMAX = 4294967295;
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


inline __device__ RNG_2x32::ctr_type generateTwoRndValues(unsigned int key,
                                                          unsigned int counter)
{
    RNG_2x32 rng;
    return rng({0, counter}, {key});
} // end of TwoRandomINTs


__global__ void cudaGenerateFirstPopulationKernel(PopulationData* populationData,
                                                  unsigned int    randomSeed)
{
    unsigned int idx    = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    const unsigned int nGenes = populationData->chromosomeSize * populationData->populationSize;

    while (idx < nGenes)
    {
        const RNG_2x32::ctr_type randomValues = generateTwoRndValues(idx, randomSeed);
        populationData->population[idx] = randomValues.v[0] % 2;
        
        idx += stride;
        if (idx < nGenes)
        {
            populationData->population[idx] = randomValues.v[1] % 2;
        }
        idx += stride;
    }

    if (threadIdx.x == 0)
    {
        populationData->fitness[blockIdx.x] = 0;
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
    {
        if (tx < stride)
        {
            s_idata[tx] += s_idata[tx + stride];
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
    int numOfEliteIdx     = blockIdx.x;  // size of NUM_OF_ELITE x 2
    int localFitnessIdx   = threadIdx.x; // size of POPULATION / NUM_OF_ELITE
    int globalFitnessIdx  = threadIdx.x + blockIdx.x * blockDim.x; // size of POPULATION x 2
    const int OFFSET      = blockDim.x;

    extern __shared__ volatile int s_fitness[];

    s_fitness[localFitnessIdx]          = populationData->fitness[globalFitnessIdx];
    s_fitness[localFitnessIdx + OFFSET] = globalFitnessIdx; 
    __syncthreads();

    for (int stride = OFFSET/2; stride >= 1; stride >>= 1)
    {
        if (localFitnessIdx < stride)
        {
            unsigned int index = (s_fitness[localFitnessIdx] >= s_fitness[localFitnessIdx + stride]) ? localFitnessIdx : localFitnessIdx + stride;
            s_fitness[localFitnessIdx]          = s_fitness[index];
            s_fitness[localFitnessIdx + OFFSET] = s_fitness[index + OFFSET];
        }
        __syncthreads();
    }

    if (localFitnessIdx == 0 && blockIdx.x < gridDim.x/2)
    {
        populationData->elitesIdx[numOfEliteIdx] = s_fitness[localFitnessIdx + OFFSET];
        printf("blockIdx:%d , eliteFitness:%d\n", blockIdx.x, s_fitness[localFitnessIdx + OFFSET]);
    }
}


__global__ void cudaGeneticManipulationKernel(PopulationData* mParentPopulation,
                                              PopulationData* mOffspringPopulation,
                                              unsigned int    randomSeed)
{
    // int chromosomeIdx = blockIdx.x;
    // int geneIdx       = threadIdx.x;
    // int total_geneIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const int CHR_PER_BLOCK = blockDim.x;

    // // Init randome number generator
    RNG_4x32 rng_4x32;
    RNG_4x32::key_type key = {{static_cast<unsigned int>(threadIdx.x), static_cast<unsigned int>(blockIdx.x)}};
    // RNG_4x32::ctr_type counter = {{0, 0, randomSeed, 0x00000d}};
    RNG_4x32::ctr_type counter = {{0, 0, 0, 0xbeeff00d}};
    // RNG_4x32::ctr_type counter = {{0, 0, randomSeed, 0xbeeff00d}};
    RNG_4x32::ctr_type randomValues1;
    RNG_4x32::ctr_type randomValues2;

    // Produce new offspring
    extern __shared__ int s[];
    int *parent1Idx  = s;
    int *parent2Idx  = (int *)(&parent1Idx[gpuEvoPrms.POPSIZE]);
    int *randNums    = (int *)(&parent2Idx[gpuEvoPrms.POPSIZE]);

    //- selection
    // 1ブロックごとに32スレッドにしている。
    // つまり1ブロック毎に最大で親32体を処理し、子供32体を生成する
    if (threadIdx.x < WARP_SIZE)
    {
        counter.incr();
        randomValues1 = rng_4x32(counter, key);
        counter.incr();
        randomValues2 = rng_4x32(counter, key);
        
        // 親1 : 0 ~ 31までのインデックス
        parent1Idx[threadIdx.x] = tournamentSelection(mParentPopulation, gpuEvoPrms.TOURNAMENT_SIZE,
                                                      randomValues1.v[0], randomValues1.v[1], 
                                                      randomValues1.v[2], randomValues1.v[3]);

        // 親2 : 0 ~ 31までのインデックス 
        parent2Idx[threadIdx.x] = tournamentSelection(mParentPopulation, gpuEvoPrms.TOURNAMENT_SIZE,
                                                      randomValues2.v[0], randomValues2.v[1],
                                                      randomValues2.v[2], randomValues2.v[3]);
    }
    __syncthreads();


    //- crossover
    if (threadIdx.x < WARP_SIZE)
    {
        counter.incr();
        randomValues1 = rng_4x32(counter, key);
        doublepointsCrossover(mParentPopulation,
                              mOffspringPopulation,
                              threadIdx.x, // offspring index
                              parent1Idx[threadIdx.x], parent2Idx[threadIdx.x],
                              randomValues1.v[0], randomValues1.v[1]);// ,
                              // randomValues1.v[2], randomValues2.v[3]);
    }
    __syncthreads();

    //- mutation
    if (threadIdx.x < WARP_SIZE)
    {
        printf("bitFlipMutation\n");
        counter.incr();
        randomValues1 = rng_4x32(counter, key);
        // randomValues1 = rng_4x32(counter, key);
        printf("BitFlipMutation: %f,%f,%f,%f\n", r123::u01fixedpt<float>(randomValues1.v[0]),
                                                 r123::u01fixedpt<float>(randomValues1.v[1]),
                                                 r123::u01fixedpt<float>(randomValues1.v[2]),
                                                 r123::u01fixedpt<float>(randomValues1.v[3]));
        bitFlipMutation(mOffspringPopulation,
                        randomValues1.v[0], randomValues1.v[1], randomValues1.v[2], randomValues1.v[3]);
    }
    __syncthreads();

    //- replacement
}

inline __device__ int getBestIndividual(const PopulationData* mParentPopulation,
                                                 const int& idx1, const int& idx2, const int& idx3, const int& idx4)
{
    int better1 = mParentPopulation->fitness[idx1] > mParentPopulation->fitness[idx2] ? idx1 : idx2;
    int better2 = mParentPopulation->fitness[idx3] > mParentPopulation->fitness[idx2] ? idx1 : idx4;
    int bestIdx = mParentPopulation->fitness[better1] > mParentPopulation->fitness[better2] ? better1 : better2;

    return bestIdx;
}


inline __device__ int tournamentSelection(const PopulationData* populationData, 
                                          int tounament_size,
                                          const std::uint32_t& random1, const std::uint32_t& random2,
                                          const std::uint32_t& random3, const std::uint32_t& random4)
{
    // トーナメントサイズは4で固定とする。
    // これはrand123が一度に返すことが出来る乱数の最大個数が4のため。
    unsigned int idx1 = random1 % populationData->populationSize;
    unsigned int idx2 = random2 % populationData->populationSize;
    unsigned int idx3 = random3 % populationData->populationSize;
    unsigned int idx4 = random4 % populationData->populationSize;
    int bestIdx = getBestIndividual(populationData, idx1, idx2, idx3, idx4);
    return idx1;
}

inline __device__ void swap(unsigned int& point1, unsigned int& point2)
{
    const unsigned int tmp = point1;
    if (point1 > point2)
    {
        point1 = point2;
        point2 = tmp;
    }
    else if (point1 == point2)
    {
        point2 += 1;
    }
}


inline __device__ void doublepointsCrossover(const PopulationData* parent,
                                             PopulationData* offspring,
                                             const unsigned int& offspringIdx,
                                             int& parent1Idx,
                                             int& parent2Idx,
                                             std::uint32_t& random1,
                                             std::uint32_t& random2)
                                             // unsigned int& random3,
                                             // unsigned int& random4)
{
    unsigned int idx1 = random1 % parent->chromosomeSize;
    unsigned int idx2 = random2 % parent->chromosomeSize;
    swap(idx1, idx2); // random1 <= random2

    int offset1 = 2 * offspringIdx * gpuEvoPrms.CHROMOSOME;
    int offset2 = 2 * offspringIdx * gpuEvoPrms.CHROMOSOME + gpuEvoPrms.CHROMOSOME;

    int i = 0;
    for (; i < idx1; ++i)
    {
        // Offspring1の0~idx1にParent1のGeneをコピーする
        offspring->population[offset1 + i] = parent->population[offset1 + i];
        // Offspring2の0~idx1にParent2のGeneをコピーする
        offspring->population[offset2 + i] = parent->population[offset2 + i];
    }
    for (; i < idx2; ++i)
    {
        // Offspring1のidx1+1~idx2にParent2のGeneをコピーする
        offspring->population[offset1 + i] = parent->population[offset2 + i];
        // Offspring2のidx1+1~idx2にParent1のGeneをコピーする
        offspring->population[offset2 + i] = parent->population[offset1 + i];
    }
    for (; i < gpuEvoPrms.CHROMOSOME; ++i)
    {
        // Offspring1のidx2+1~CHROMOSOMEにParent1のGeneをコピーする
        offspring->population[offset1 + i] = parent->population[offset1 + i];
        // Offspring2のidx2+1~CHROMOSOMEにParent2のGeneをコピーする
        offspring->population[offset2 + i] = parent->population[offset2 + i];
    }

#ifdef _DEBUG
    printf("%d,%d,%d,%d,%d\n", threadIdx.x, parent1Idx, parent2Idx, idx1, idx2);
#endif // _DEBUG
}

inline __device__ void bitFlipMutation(PopulationData* offspring,
                                       std::uint32_t& random1, std::uint32_t& random2,
                                       std::uint32_t& random3, std::uint32_t& random4)
{
    if (r123::u01fixedpt<float>(random1) < gpuEvoPrms.MUTATION_RATE)
    {
#ifdef _DEBUG
        printf("rand1 %d:%d:%d\n", random1 % gpuEvoPrms.CHROMOSOME, 
                                   offspring->population[random1 % gpuEvoPrms.CHROMOSOME],
                                   offspring->population[random1 % gpuEvoPrms.CHROMOSOME] ^ 1);
#endif // _DEBUG
        offspring->population[random1 % gpuEvoPrms.CHROMOSOME] ^= 1;
    }
    if (r123::u01fixedpt<float>(random2) < gpuEvoPrms.MUTATION_RATE)
    {
#ifdef _DEBUG
        printf("rand2 %d:%d:%d\n", random2 % gpuEvoPrms.CHROMOSOME, 
                                   offspring->population[random2 % gpuEvoPrms.CHROMOSOME],
                                   offspring->population[random2 % gpuEvoPrms.CHROMOSOME] ^ 1);
#endif // _DEBUG
        offspring->population[random2 % gpuEvoPrms.CHROMOSOME] ^= 1;
    }
    if (r123::u01fixedpt<float>(random3) < gpuEvoPrms.MUTATION_RATE)
    {
#ifdef _DEBUG
        printf("rand3 %d:%d:%d\n", random3 % gpuEvoPrms.CHROMOSOME, 
                                   offspring->population[random3 % gpuEvoPrms.CHROMOSOME],
                                   offspring->population[random3 % gpuEvoPrms.CHROMOSOME] ^ 1);
#endif // _DEBUG
        offspring->population[random3 % gpuEvoPrms.CHROMOSOME] ^= 1;
    }
    if (r123::u01fixedpt<float>(random4) < gpuEvoPrms.MUTATION_RATE)
    {
#ifdef _DEBUG
        printf("rand4 %d:%d:%d\n", random4 % gpuEvoPrms.CHROMOSOME, 
                                   offspring->population[random4 % gpuEvoPrms.CHROMOSOME],
                                   offspring->population[random4 % gpuEvoPrms.CHROMOSOME] ^ 1);
#endif // _DEBUG
        offspring->population[random4 % gpuEvoPrms.CHROMOSOME] ^= 1;
    }
}

