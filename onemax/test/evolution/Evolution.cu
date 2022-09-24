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
    printf("constructor\n");
    //- Select device
    // cudaSetDevice(mDeviceIdx);
    // checkAndReportCudaError(__FILE__, __LINE__);

    //- Get parameters of the device
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, mDeviceIdx);

    // Create populations on CPU
    mHostParentPopulation    = new CPUPopulation(prms->getPopsize(), prms->getChromosome(), prms->getNumOfElite());
    mHostOffspringPopulation = new CPUPopulation(prms->getPopsize(), prms->getChromosome(), prms->getNumOfElite());

#ifdef _DEBUG
    for (int i = 0; i < prms->getPopsize(); ++i)
    {
        printf("%d,", i);
        for (int j = 0; j < prms->getChromosome(); ++j)
        {
            printf("%d", mHostParentPopulation->getDeviceData()->population[i * prms->getChromosome() + j]);
        }
        printf(":%d\n", mHostParentPopulation->getDeviceData()->fitness[i]);
    }
#endif // _DEBUG

    // Create populations on GPU
    mDevParentPopulation    = new GPUPopulation(prms->getPopsize(), prms->getChromosome(), prms->getNumOfElite());
    mDevOffspringPopulation = new GPUPopulation(prms->getPopsize(), prms->getChromosome(), prms->getNumOfElite());

    // Copy population from CPU to GPU
    mDevParentPopulation->copyToDevice(mHostParentPopulation->getDeviceData());
    mDevOffspringPopulation->copyToDevice(mHostOffspringPopulation->getDeviceData());

    mMultiprocessorCount = prop.multiProcessorCount;
    // mParams.setNumberOfDeviceSMs(prop.multiProcessorCount);

    // Create statistics
    // mStatistics = new GPUStatistics();

    // Initialize Random seed
    initRandomSeed();
    printf("end of constructor\n");
}; // end of GPUEvolution


/**
 * Destructor of the class
 */
GPUEvolution::~GPUEvolution()
{
    delete mHostParentPopulation;
    delete mHostOffspringPopulation;
    delete mDevParentPopulation;
    delete mDevOffspringPopulation;
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

    runEvolutionCycle(prms);

#ifdef _DEBUG
    showPopulation(prms);
#endif // _DEBUG
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

    dim3 blocks;
    dim3 threads;


    //- 初期集団生成 ------------------------------------------------------------------------------
    blocks.x  = prms->getPopsize() / 2;
    blocks.y  = 1;
    blocks.z  = 1;

    threads.x = prms->getChromosome();
    threads.y = 1;
    threads.z = 1;

    cudaGenerateFirstPopulationKernel<<<blocks, threads>>>
                                     (mDevParentPopulation->getDeviceData(), getRandomSeed());
    checkAndReportCudaError(__FILE__, __LINE__);

    /*
    cudaGenerateFirstPopulationKernel<<<blocks, threads>>>
                                     (mDevOffspringPopulation->getDeviceData(),
                                      getRandomSeed());
    checkAndReportCudaError(__FILE__, __LINE__);
    */


    //- Fitness評価 -------------------------------------------------------------------------------
    blocks.x  = prms->getPopsize();
    blocks.y  = 1;
    blocks.z  = 1;

    threads.x = prms->getChromosome();
    threads.y = 1;
    threads.z = 1;

    evaluation<<<blocks, threads, prms->getChromosome() * sizeof(int)>>>(mDevParentPopulation->getDeviceData());
    checkAndReportCudaError(__FILE__, __LINE__);

    evaluation<<<blocks, threads, prms->getChromosome() * sizeof(int)>>>(mDevOffspringPopulation->getDeviceData());
    checkAndReportCudaError(__FILE__, __LINE__);


    //- 疑似エリート保存戦略 ----------------------------------------------------------------------
    blocks.x  = prms->getNumOfElite() * 2;
    blocks.y  = 1;
    blocks.z  = 1;

    threads.x = prms->getPopsize() / prms->getNumOfElite();
    threads.y = 1;
    threads.z = 1;
    printf("blocks.x:%d, threads.x:%d, offset:%d, shared_memory_size:%d\n",
            blocks.x, threads.x, blocks.x * threads.x / 2, prms->getPopsize() * 2);

    pseudo_elitism<<<blocks, threads, threads.x * 2 * sizeof(int)>>>(mDevParentPopulation->getDeviceData());
    checkAndReportCudaError(__FILE__, __LINE__);

    pseudo_elitism<<<blocks, threads, threads.x * 2 * sizeof(int)>>>(mDevOffspringPopulation->getDeviceData());
    checkAndReportCudaError(__FILE__, __LINE__);

    // ホストとデバイスの同期を取る
    // cudaDeviceSynchronize();

} // end of initialize


/**
 * Run evolutionary cycle for defined number of generations
 */
void GPUEvolution::runEvolutionCycle(Parameters* prms)
{
    dim3 blocks;
    dim3 threads;

    int CHR_PER_BLOCK = (prms->getPopsize() % WARP_SIZE == 0)
                         ? prms->getPopsize() / WARP_SIZE
                         // ? prms->getPopsize() / WARP_SIZE
                         : prms->getPopsize() / WARP_SIZE + 1;

    // printf("CHR_PER_BLOCK %d\n", CHR_PER_BLOCK);
    blocks.x = CHR_PER_BLOCK;
    blocks.y = 1;
    blocks.z = 1;

    // printf("WARP_SIZE %d\n", WARP_SIZE);
    // threads.x = (prms->getPopsize() > WARP_SIZE) ? WARP_SIZE : prms->getPopsize();
    threads.x = (prms->getPopsize() > WARP_SIZE) ? WARP_SIZE : prms->getPopsize() / 2;
    // threads.x = (prms->getPopsize() > WARP_SIZE) ? WARP_SIZE : prms->getPopsize() * 2;
    threads.y = 1;
    threads.z = 1;

    int shared_memory_size =   prms->getPopsize()        * sizeof(int)
                             + prms->getPopsize()        * sizeof(int)
                             + prms->getTournamentSize() * sizeof(int);

    printf("Start of cudaGeneticManipulationKernel\n");
    printf("GA: blocks: %d, threads: %d\n", blocks.x, threads.x);
    cudaGeneticManipulationKernel<<<blocks, threads, shared_memory_size>>> 
                                 (mDevParentPopulation->getDeviceData(),
                                  mDevOffspringPopulation->getDeviceData(),
                                  getRandomSeed());
    checkAndReportCudaError(__FILE__, __LINE__);
    printf("End of cudaGeneticManipulationKernel\n");

    //- Fitness評価 -------------------------------------------------------------------------------
    blocks.x  = prms->getPopsize();
    blocks.y  = 1;
    blocks.z  = 1;

    threads.x = prms->getChromosome();
    threads.y = 1;
    threads.z = 1;

    printf("Evaluation: blocks: %d, threads: %d\n", blocks.x, threads.x);
    evaluation<<<blocks, threads, prms->getChromosome() * sizeof(int)>>>(mDevParentPopulation->getDeviceData());
    checkAndReportCudaError(__FILE__, __LINE__);

    evaluation<<<blocks, threads, prms->getChromosome() * sizeof(int)>>>(mDevOffspringPopulation->getDeviceData());
    checkAndReportCudaError(__FILE__, __LINE__);

    //- 疑似エリート保存戦略 ----------------------------------------------------------------------
    blocks.x  = prms->getNumOfElite() * 2;
    blocks.y  = 1;
    blocks.z  = 1;

    threads.x = prms->getPopsize() / prms->getNumOfElite();
    threads.y = 1;
    threads.z = 1;
    printf("blocks.x:%d, threads.x:%d, offset:%d, shared_memory_size:%d\n",
            blocks.x, threads.x, blocks.x * threads.x / 2, prms->getPopsize() * 2);

    pseudo_elitism<<<blocks, threads, threads.x * 2 * sizeof(int)>>>(mDevParentPopulation->getDeviceData());
    checkAndReportCudaError(__FILE__, __LINE__);

    pseudo_elitism<<<blocks, threads, threads.x * 2 * sizeof(int)>>>(mDevOffspringPopulation->getDeviceData());
    checkAndReportCudaError(__FILE__, __LINE__);
}


void GPUEvolution::showPopulation(Parameters* prms)
{
    int csize = prms->getChromosome();
    int psize = prms->getPopsize();
    int esize = prms->getNumOfElite();

    mDevParentPopulation->copyFromDevice(mHostParentPopulation->getDeviceData());
    mDevOffspringPopulation->copyFromDevice(mHostOffspringPopulation->getDeviceData());

    printf("------------ Parent ------------ \n");
    // for (int k = 0; k < psize; ++k)
    for (int k = 0; k < esize; ++k)
    {
        printf("elite%d : %d\n", k, mHostParentPopulation->getDeviceData()->elitesIdx[k]);
    }
    printf("\n");

    for (int i = 0; i < psize; ++i)
    {
        printf("%d,", i);
        for (int j = 0; j < csize; ++j)
        {
            printf("%d", mHostParentPopulation->getDeviceData()->population[i * csize + j]);
        }
        printf(":%d\n", mHostParentPopulation->getDeviceData()->fitness[i]);
    }

    printf("------------ Offspring ------------ \n");
    for (int k = 0; k < esize; ++k)
    {
        printf("elite%d : %d\n", k, mHostOffspringPopulation->getDeviceData()->elitesIdx[k]);
    }
    printf("\n");
    
    for (int i = 0; i < psize; ++i)
    {
        printf("%d,", i);
        for (int j = 0; j < csize; ++j)
        {
            printf("%d", mHostOffspringPopulation->getDeviceData()->population[i * csize + j]);
        }
        printf(":%d\n", mHostOffspringPopulation->getDeviceData()->fitness[i]);
    }
}


