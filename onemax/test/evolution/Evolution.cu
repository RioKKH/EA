#include <stdio.h>
#include <sys/time.h>

#include "Evolution.h"
#include "CUDAKernels.h"
// #include "Parameters.h"

/**
 * Constructor of the class
 */
GPUEvolution::GPUEvolution()
    : mRandomSeed(0),
      mDeviceIdx(0)
{
    // Select device
    // cudaSetDevice(mDeviceIdx);
    // checkAndReportCudaError(__FILE__, __LINE__);

    // Get parameters of the device
    cudaDeviceProp prop;

    // cudaGetDeviceProperties(&prop, mDeviceIdx);
    // checkAndReportCudaError(__FILE__, __LINE__);

    // multiprocessorCount = prop.multiProcessorCount;
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
} // end of Destructor


/**
 * Run Evolution
 */
void GPUEvolution::run()
{
    // initialize();
    runEvolutionCycle();
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
void GPUEvolution::initialize()
{
    /// TODO
} // end of initialize

/**
 * Run evolutionary cycle for defined number of generations
 */
void GPUEvolution::runEvolutionCycle()
{
    dim3 blocks;
    dim3 threads;

    blocks.x = 1;
    blocks.y = 32;
    blocks.z = 1;

    threads.x = 32;
    threads.y = 16;
    threads.z = 1;

    // Every chromosome is treated by a single warp, theare are as many warps as individuals per block
    // threads.x = WARP_SIZE;
    // threads.y = CHR_PER_BLOCK;
    // threads.z = 1;

    printf("Before cuda kernel\n");
    cudaCallRandomNumber<<<32, 4>>>(getRandomSeed());
    // cudaCallRandomNumber<<<blocks, threads>>>(getRandomSeed());
    cudaDeviceSynchronize();
}

