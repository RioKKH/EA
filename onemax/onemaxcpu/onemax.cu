#include <iostream>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include "population.hpp"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char **argv)
{
    int gen_max = 0;
    int pop_size = 0;
    int chromosome = 0;

    // 実行時間計測用
    float elapsed_time = 0.0f;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    Parameters *prms;
    prms = new Parameters();
    gen_max = prms->getGenMax();
    pop_size = prms->getPopSize();
    chromosome = prms->getNumberOfChromosome();

    srand((unsigned int)time(NULL));
    // std::cout << "!!!Start!!!" << std::endl;;

    population *pop;
    pop = new population(prms);

    // double iStart = cpuSecond();
    cudaEventRecord(start, 0);
    for (int i = 1; i <= gen_max; i++) {
        pop->alternate();
    }
    // double iElaps = cpuSecond() - iStart;
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    std::cout << pop_size << "," << chromosome << "," << elapsed_time << std::endl;
    // std::cout << pop_size << "," << chromosome << "," << iElaps << std::endl;
    // pop->print_result();

    // delete pointers
    delete pop;
    delete prms;
    // std::cout << "!!!End!!!" << std::endl;

    return 0;
}
