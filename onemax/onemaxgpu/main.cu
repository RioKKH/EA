#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "Parameters.hpp"
#include "CUDAKernels.h"
#include "Misc.h"

int main(int arc, char* argv[])
{
    Parameters *prms = new Parameters();
    prms->loadParams();
    copyToDevice(prms->getEvoPrms());

    const int POPSIZE                 = prms->getPopsize();
    const int CHROMOSOME              = prms->getChromosome();
    const int NUM_OF_GENERATIONS      = prms->getNumOfGenerations();
    const int NUM_OF_ELITE            = prms->getNumOfElite();
    const int TOURNAMENT_SIZE         = prms->getTournamentSize();
    const int NUM_OF_CROSSOTER_POINTS = prms->getNumOfCrossoverPoints();

    const float MUTATION_RATE         = prms->getMutationRate();

    const int N                       = POPSIZE * CHROMOSOME;
    const int Nbytes                  = N * sizeof(int);

    //- GPU用変数
    thrust::device_vector<int> dev_PopulationOdd(N);
    thrust::device_vector<int> dev_PopulationEven(N);
    thrust::device_vector<int> dev_Parent1(POPSIZE);
    thrust::device_vector<int> dev_Parent2(POPSIZE);
    thrust::device_vector<int> dev_Fitness(POPSIZE);
    thrust::device_vector<int> dev_SortedFitnesses(POPSIZE);
    thrust::device_vector<int> dev_SortedId(POPSIZE);
    thrust::device_vector<int> dev_TournamentIndividuals(TOURNAMENT_SIZE);
    thrust::device_vector<int> dev_TournamentFitness(TOURNAMENT_SIZE);

    int *pdev_PopulationOdd         = thrust::raw_pointer_cast(&dev_PopulationOdd[0]);
    int *pdev_PopulationEven        = thrust::raw_pointer_cast(&dev_PopulationEven[0]);
    int *pdev_Parent1               = thrust::raw_pointer_cast(&dev_Parent1[0]);
    int *pdev_Paretn2               = thrust::raw_pointer_cast(&dev_Parent2[0]);
    int *pdev_Fitness               = thrust::raw_pointer_cast(&dev_Fitness[0]);
    int *pdev_SortedFitness         = thrust::raw_pointer_cast(&dev_SortedFitnesses[0]);
    int *pdev_SortedId              = thrust::raw_pointer_cast(&dev_SortedId[0]);
    int *pdev_TournamentIndividuals = thrust::raw_pointer_cast(&dev_TournamentIndividuals[0]);
    int *pdev_TournamentFitness     = thrust::raw_pointer_cast(&dev_TournamentFitness[0]);

    //- GPU用変数 - 乱数用変数確保
    float *pdev_tournament_rand1;
    float *pdev_tournament_rand2;
    cudaMalloc((void **)&pdev_tournament_rand1, TOURNAMENT_SIZE * sizeof(float));
    cudaMalloc((void **)&pdev_tournament_rand2, TOURNAMENT_SIZE * sizeof(float));

    //- CPU用変数
    int *phost_Population;
    int *phost_Fitness;
    int *phost_SortedId;
    int *phost_Parent1;
    int *phost_Parent2;

    //- CPU用変数領域確保
    phost_Population = (int *)malloc(POPSIZE * CHROMOSOME * sizeof(int));
    phost_Fitness    = (int *)malloc(POPSIZE * sizeof(int));
    phost_SortedId   = (int *)malloc(POPSIZE * sizeof(int));
    phost_Parent1    = (int *)malloc(POPSIZE * sizeof(int));
    phost_Parent2    = (int *)malloc(POPSIZE * sizeof(int));


    //− 初期Population生成とデバイスメモリへコピー
    initializePopulationOnCPU(phost_Population, prms);
    cudaMemcpy(pdev_PopulationEven, phost_Population, Nbytes, cudaMemcpyHostToDevice);

    //- 評価
    evaluation<<<POPSIZE, CHROMOSOME, CHROMOSOME * sizeof(int)>>>(pdev_PopulationEven, pdev_Fitness);
    cudaDeviceSynchronize();

    //- エリート戦略のためのソート
    thrust::copy(thrust::device, dev_Fitness.begin(), dev_Fitness.end(), dev_SortedFitnesses.begin());
    thrust::sequence(dev_SortedId.begin(), dev_SortedId.end());
    thrust::sort_by_key(dev_SortedFitnesses.begin(), dev_SortedFitnesses.end(), dev_SortedId.begin());

    //- 評価値表示 エリート戦略のためのソート後
    dev_show<<<1, POPSIZE>>>(pdev_SortedFitness,  pdev_SortedId);
    cudaDeviceSynchronize();

    int gen = 1;
    //- セレクション
    selection<<<1, POPSIZE>>>(pdev_Fitness,               pdev_SortedId,
                              pdev_tournament_rand1,      pdev_tournament_rand2,
                              pdev_Parent1,               pdev_Paretn2,
                              gen,
                              pdev_TournamentIndividuals, pdev_TournamentFitness);
    cudaDeviceSynchronize();

    //- 後片付け
    free(phost_Population);
    free(phost_Fitness);
    free(phost_SortedId);
    free(phost_Parent1);
    free(phost_Parent2);

    cudaFree(pdev_tournament_rand1);
    cudaFree(pdev_tournament_rand2);

    return 0;
}



