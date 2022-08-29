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

    mHostPopulationOdd  = new CPUPopulation(prms->getPopsize(), prms->getChromosome());
    mHostPopulationEven = new CPUPopulation(prms->getPopsize(), prms->getChromosome());

    printf("check: %d, %d\n",
            mHostPopulationOdd->getDeviceData()->chromosomeSize,
            mHostPopulationOdd->getDeviceData()->populationSize);

    // Create populations on CPU
    mDevPopulationOdd  = new GPUPopulation(prms->getPopsize(), prms->getChromosome());
    mDevPopulationEven = new GPUPopulation(prms->getPopsize(), prms->getChromosome());

    // Create populations on GPU
    mDevPopulationEven->copyToDevice(mHostPopulationEven->getDeviceData());
    mDevPopulationOdd->copyToDevice(mHostPopulationOdd->getDeviceData());

    mMultiprocessorCount = prop.multiProcessorCount;
    // mParams.setNumberOfDeviceSMs(prop.multiProcessorCount);

    // Load knapsack data from the file.
    // mGlobalData.LoadFromFile();

    // Create populations on GPU
    // mMasterPopulation = new GPUPopulation(mParams.getPopulationsSize(), mParams.getChromosomeSize());
    // mOffspringPopulation = new GPUPopulation(mParams.getOffspringPopulationsSize(), mParams.getChromosomeSize());

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
    mDevPopulationEven->copyFromDevice(mHostPopulationEven->getDeviceData());
    for (int i = 0; i < mHostPopulationEven->getDeviceData()->populationSize; i++)
    {
        printf("%d,%d\n", i, mHostPopulationEven->getDeviceData()->population[i]);
    }
#endif
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
    copyToDevice(prms->getEvoPrms());
    cudaGenerateFirstPopulationKernel<<<mMultiprocessorCount * 2, 256>>>
                                     (mDevPopulationEven->getDeviceData(),
                                      getRandomSeed());

    cudaGenerateFirstPopulationKernel<<<mMultiprocessorCount * 2, 256>>>
                                     (mDevPopulationOdd->getDeviceData(),
                                      getRandomSeed());
    cudaDeviceSynchronize();
    // TODO
    evaluation<<<,>>>(mDevPopulationEven->getDeviceData());
    evaluation<<<,>>>(mDevPopulationOdd->getDeviceData());

} // end of initialize

/**
 * Run evolutionary cycle for defined number of generations
 */
void GPUEvolution::runEvolutionCycle(Parameters* prms)
{
    dim3 blocks;
    dim3 threads;

    blocks.x = 1;
    blocks.y = 32;
    blocks.z = 1;

    threads.x = 32;
    threads.y = 16;
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

