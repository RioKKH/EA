#include <stdio.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "Evolution.h"
#include "CUDAKernels.h"
#include "Parameters.h"

/**
 * Constructor of the class
 */
GPUEvolution::GPUEvolution()
    : mRandomSeed(0), mDeviceIdx(0)
{
}

GPUEvolution::GPUEvolution(Parameters* prms)
    : mRandomSeed(0),
      mDeviceIdx(0)
{
    //- Select device
    // cudaSetDevice(mDeviceIdx);
    // checkAndReportCudaError(__FILE__, __LINE__);

    //- Get parameters of the device
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, mDeviceIdx);

    // Create populations on CPU
    mHostPopulationOdd  = new CPUPopulation(prms->getPopsize(), prms->getChromosome(), prms->getNumOfElite());
    mHostPopulationEven = new CPUPopulation(prms->getPopsize(), prms->getChromosome(), prms->getNumOfElite());

    printf("check: %d, %d, %d\n",
            mHostPopulationOdd->getDeviceData()->chromosomeSize,
            mHostPopulationOdd->getDeviceData()->populationSize,
            mHostPopulationOdd->getDeviceData()->elitesSize);

    // Create populations on GPU
    mDevPopulationOdd  = new GPUPopulation(prms->getPopsize(), prms->getChromosome(), prms->getNumOfElite());
    mDevPopulationEven = new GPUPopulation(prms->getPopsize(), prms->getChromosome(), prms->getNumOfElite());

    // Copy population from CPU to GPU
    mDevPopulationEven->copyToDevice(mHostPopulationEven->getDeviceData());
    mDevPopulationOdd->copyToDevice(mHostPopulationOdd->getDeviceData());

    mMultiprocessorCount = prop.multiProcessorCount;
    // mParams.setNumberOfDeviceSMs(prop.multiProcessorCount);

    // Create statistics
    // mStatistics = new GPUStatistics();

    // Initialize Random seed
    initRandomSeed();
}; // end of GPUEvolution


/**
 * Destructor of the class
 */
GPUEvolution::~GPUEvolution()
{
    delete mHostPopulationEven;
    delete mHostPopulationOdd;
    delete mDevPopulationEven;
    delete mDevPopulationOdd;
} // end of Destructor


/**
 * Run Evolution
 */
void GPUEvolution::run(Parameters* prms)
{
    initialize(prms);

#ifdef _DEBUG
    showPopulation(prms);
#endif // _DEBUG

    // runEvolutionCycle(prms);
}


void GPUEvolution::initRandomSeed()
{
    struct timeval tp1;
    gettimeofday(&tp1, nullptr);
    mRandomSeed = (tp1.tv_sec / (mDeviceIdx + 1)) * tp1.tv_usec;
}

/**
 * Initialization of the GA
 */
void GPUEvolution::initialize(Parameters* prms)
{
    dim3 blocks;
    dim3 threads;

    blocks.x = prms->getPopsize();
    blocks.y = 1;
    blocks.z = 1;

    threads.x = prms->getChromosome();
    threads.y = 1;
    threads.z = 1;

    copyToDevice(prms->getEvoPrms());
    cudaGenerateFirstPopulationKernel<<<mMultiprocessorCount * 2, 256>>>
                                     (mDevPopulationEven->getDeviceData(),
                                      getRandomSeed());

    cudaGenerateFirstPopulationKernel<<<mMultiprocessorCount * 2, 256>>>
                                     (mDevPopulationOdd->getDeviceData(),
                                      getRandomSeed());
    cudaDeviceSynchronize();

    evaluation<<<prms->getPopsize(),
                 prms->getChromosome(),
                 prms->getChromosome() * sizeof(int)>>>(mDevPopulationEven->getDeviceData());

    evaluation<<<prms->getPopsize(),
                 prms->getChromosome(),
                 prms->getChromosome() * sizeof(int)>>>(mDevPopulationOdd->getDeviceData());
    cudaDeviceSynchronize();

} // end of initialize

/**
 * Run evolutionary cycle for defined number of generations
 */
void GPUEvolution::runEvolutionCycle(Parameters* prms)
{
    dim3 blocks;
    dim3 threads;

    blocks.x = prms->getPopsize();
    blocks.y = 1;
    blocks.z = 1;

    // threads.x = WARP_SIZE;
    //threads.y = prms->getChromosome() / WARP_SIZE;
    threads.x = prms->getChromosome();
    threads.y = 1;
    threads.z = 1;

    const int POPSIZE = prms->getPopsize();
    printf("POPSIZE %d\n", POPSIZE);

    // Every chromosome is treated by a single warp, theare are as many warps as individuals per block
    // threads.x = WARP_SIZE;
    // threads.y = CHR_PER_BLOCK;
    // threads.z = 1;

    printf("Before cuda kernel\n");
    cudaCallRandomNumber<<<32, 4>>>(getRandomSeed());
    // cudaCallRandomNumber<<<blocks, threads>>>(getRandomSeed());
    cudaDeviceSynchronize();
}

void GPUEvolution::showPopulation(Parameters* prms)
{
    int csize = prms->getChromosome();
    int psize = prms->getPopsize();

    mDevPopulationEven->copyFromDevice(mHostPopulationEven->getDeviceData());
    mDevPopulationOdd->copyFromDevice(mHostPopulationOdd->getDeviceData());

    for (int i = 0; i < psize; ++i)
    {
        printf("%d,", i);
        for (int j = 0; j < csize; ++j)
        {
            printf("%d", mHostPopulationEven->getDeviceData()->population[psize * i + j]);
        }
        printf(":%d\n", mHostPopulationEven->getDeviceData()->fitness[i]);
    }
}


