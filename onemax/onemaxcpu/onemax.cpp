#include <iostream>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include "population.hpp"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char **argv)
{
    int gen_max = 0;
    // int pop_size = 0;

    Parameters *prms;
    prms = new Parameters();
    gen_max = prms->getGenMax();

    srand((unsigned int)time(NULL));
    // std::cout << "!!!Start!!!" << std::endl;;

    population *pop;
    pop = new population(prms);

    double iStart = cpuSecond();
    for (int i = 1; i <= gen_max; i++) {
        pop->alternate();
    }
    double iElaps = cpuSecond() - iStart;
    std::cout << iElaps << std::endl;
    // pop->print_result();

    // delete pointers
    delete pop;
    delete prms;
    // std::cout << "!!!End!!!" << std::endl;

    return 0;
}
