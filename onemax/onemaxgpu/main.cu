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
    // const int Nbytes                  = N * sizeof(int);

    //- GPU用変数
    /*
    thrust::device_vector<int> dev_PopulationOdd(N);
    thrust::device_vector<int> dev_PopulationEven(N);
    thrust::device_vector<int> dev_Parent1(POPSIZE);
    thrust::device_vector<int> dev_Parent2(POPSIZE);
    thrust::device_vector<int> dev_Fitness(POPSIZE);
    thrust::device_vector<int> dev_SortedFitnesses(POPSIZE);
    thrust::device_vector<int> dev_SortedId(POPSIZE);
    thrust::device_vector<int> dev_TournamentIndividuals(TOURNAMENT_SIZE);
    thrust::device_vector<int> dev_TounamentFitness(TOURNAMENT_SIZE);
    */

    //- 乱数用変数確保
    unsigned int *dev_tournament_rand1;
    unsigned int *dev_tournament_rand2;
    cudaMalloc((void **)&dev_tournament_rand1, TOURNAMENT_SIZE * sizeof(unsigned int));
    cudaMalloc((void **)&dev_tournament_rand2, TOURNAMENT_SIZE * sizeof(unsigned int));



    //- 後片付け
    cudaFree(dev_tournament_rand1);
    cudaFree(dev_tournament_rand2);

    return 0;
}



