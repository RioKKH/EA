#include <iostream>
#include <cstdio>
#include <fstream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include "Parameters.hpp"
#include "GAregex.hpp"


void Parameters::loadParams()
{
    std::ifstream infile(PARAMNAME);
    std::string line;
    std::smatch results;

    while (getline(infile, line))
    {
        if (std::regex_match(line, results, rePOPSIZE))
        {
            cpuEvoPrms.POPSIZE = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reCHROMOSOME))
        {
            cpuEvoPrms.CHROMOSOME = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reNUM_OF_GENERATIONS))
        {
            cpuEvoPrms.NUM_OF_GENERATIONS = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reNUM_OF_ELITE))
        {
            cpuEvoPrms.NUM_OF_ELITE = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reTOURNAMENT_SIZE))
        {
            cpuEvoPrms.TOURNAMENT_SIZE = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reNUM_OF_CROSSOVER_POINTS))
        {
            cpuEvoPrms.NUM_OF_CROSSOVER_POINTS = std::stoi(results[1].str());
        }
        else if (std::regex_match(line, results, reMUTATION_RATE))
        {
            cpuEvoPrms.MUTATION_RATE = std::stof(results[1].str());
        }
    }
    cpuEvoPrms.N = cpuEvoPrms.POPSIZE * cpuEvoPrms.CHROMOSOME;
    cpuEvoPrms.Nbytes = cpuEvoPrms.N * sizeof(int);

#ifdef _DEBUG
    std::cout << "POPSIZE: " << cpuEvoPrms.POPSIZE << std::endl;
    std::cout << "CHROMOSOME: " << cpuEvoPrms.CHROMOSOME << std::endl;
    std::cout << "NUM_OF_GENERATIONS: " << cpuEvoPrms.NUM_OF_GENERATIONS << std::endl;
    std::cout << "NUM_OF_ELITE: " << cpuEvoPrms.NUM_OF_ELITE << std::endl;
    std::cout << "TOURNAMENT_SIZE: " << cpuEvoPrms.TOURNAMENT_SIZE << std::endl;
    std::cout << "NUM_OF_CROSSOVER_POINTS: " << cpuEvoPrms.NUM_OF_CROSSOVER_POINTS << std::endl;
    std::cout << "MUTATION_RATE: " << cpuEvoPrms.MUTATION_RATE << std::endl;
#endif // _DEBUG

    infile.close();

    return;
}

int Parameters::getPopsize() const { return cpuEvoPrms.POPSIZE; }
int Parameters::getChromosome() const { return cpuEvoPrms.CHROMOSOME; }
int Parameters::getNumOfGenerations() const { return cpuEvoPrms.NUM_OF_GENERATIONS; }
int Parameters::getNumOfElite() const { return cpuEvoPrms.NUM_OF_ELITE; }
int Parameters::getTournamentSize() const { return cpuEvoPrms.TOURNAMENT_SIZE; }
int Parameters::getNumOfCrossoverPoints() const { return cpuEvoPrms.NUM_OF_CROSSOVER_POINTS; }
float Parameters::getMutationRate() const { return cpuEvoPrms.MUTATION_RATE; }
int Parameters::getN() const { return cpuEvoPrms.N; }
int Parameters::getNbytes() const { return cpuEvoPrms.Nbytes; }
EvolutionParameters Parameters::getEvoPrms() const { return cpuEvoPrms; }


