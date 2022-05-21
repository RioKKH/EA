#include <iostream>
#include "Parameters.hpp"

int main(int argc, char **argv)
{
    Parameters *prms;
    prms = new Parameters();

    prms->loadParams();

    std::cout << prms->getPopsize() << std::endl;
    std::cout << prms->getChromosome() << std::endl;
    std::cout << prms->getNumOfGenerations() << std::endl;
    std::cout << prms->getNumOfElite() << std::endl;
    std::cout << prms->getTournamentSize() << std::endl;
    std::cout << prms->getNumOfCrossoverPoints() << std::endl;
    std::cout << prms->getMutationRate() << std::endl;
}

