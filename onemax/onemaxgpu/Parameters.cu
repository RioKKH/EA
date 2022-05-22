#include <iostream>
#include <cstdio>
#include <fstream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include "Parameters.hpp"
#include "GAregex.hpp"

__constant__ EvolutionParameters *gpuEvoPrms;

Parameters& Parameters::getInstance()
{
    static Parameters instance;
    return instance;
}

void Parameters::copyToDevice()
{
    printf("copyToDEvice %d\n", mEvolutionParameters.POPSIZE);
    cudaMemcpyToSymbol(gpuEvoPrms,
                       &mEvolutionParameters,
                       sizeof(mEvolutionParameters));
}

void Parameters::loadParams()
{
    std::ifstream infile(PARAMNAME);
    std::string line;
    std::smatch results;

    while (getline(infile, line))
    {
        if (std::regex_match(line, results, rePOPSIZE))
        {
            mEvolutionParameters.POPSIZE = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reCHROMOSOME))
        {
            mEvolutionParameters.CHROMOSOME = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reNUM_OF_GENERATIONS))
        {
            mEvolutionParameters.NUM_OF_GENERATIONS = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reNUM_OF_ELITE))
        {
            mEvolutionParameters.NUM_OF_ELITE = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reTOURNAMENT_SIZE))
        {
            mEvolutionParameters.TOURNAMENT_SIZE = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reNUM_OF_CROSSOVER_POINTS))
        {
            mEvolutionParameters.NUM_OF_CROSSOVER_POINTS = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reMUTATION_RATE))
        {
            mEvolutionParameters.MUTATION_RATE = std::stof(results[1].str());
        }
    }
    mEvolutionParameters.N = mEvolutionParameters.POPSIZE * mEvolutionParameters.CHROMOSOME;
    mEvolutionParameters.Nbytes = mEvolutionParameters.N * sizeof(int);

#ifdef _DEBUG
    std::cout << "POPSIZE: " << mEvolutionParameters.POPSIZE << std::endl;
    std::cout << "CHROMOSOME: " << mEvolutionParameters.CHROMOSOME << std::endl;
    std::cout << "NUM_OF_GENERATIONS: " << mEvolutionParameters.NUM_OF_GENERATIONS << std::endl;
    std::cout << "NUM_OF_ELITE: " << mEvolutionParameters.NUM_OF_ELITE << std::endl;
    std::cout << "TOURNAMENT_SIZE: " << mEvolutionParameters.TOURNAMENT_SIZE << std::endl;
    std::cout << "NUM_OF_CROSSOVER_POINTS: " << mEvolutionParameters.NUM_OF_CROSSOVER_POINTS << std::endl;
    std::cout << "MUTATION_RATE: " << mEvolutionParameters.MUTATION_RATE << std::endl;
#endif // _DEBUG

    infile.close();

    return;
}

int Parameters::getPopsize() const { return mEvolutionParameters.POPSIZE; }
int Parameters::getChromosome() const { return mEvolutionParameters.CHROMOSOME; }
int Parameters::getNumOfGenerations() const { return mEvolutionParameters.NUM_OF_GENERATIONS; }
int Parameters::getNumOfElite() const { return mEvolutionParameters.NUM_OF_ELITE; }
int Parameters::getTournamentSize() const { return mEvolutionParameters.TOURNAMENT_SIZE; }
int Parameters::getNumOfCrossoverPoints() const { return mEvolutionParameters.NUM_OF_CROSSOVER_POINTS; }
float Parameters::getMutationRate() const { return mEvolutionParameters.MUTATION_RATE; }
int Parameters::getN() const { return mEvolutionParameters.N; }
int Parameters::getNbytes() const { return mEvolutionParameters.Nbytes; }


