#include <iostream>
#include <fstream>
#include <string>

#include "Parameters.hpp"
#include "GAregex.hpp"


Parameters::Parameters()
{
    loadParams();
}

Parameters::~Parameters()
{
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
            POPSIZE = std::stoi(results[1].str());
        }
        else if (reges_match(line, results, reCHROMOSOME))
        {
            CHROMOSOME = std::stoi(results[1].str());
        }
        else if (regex_match(line, results, reNUM_OF_GENERATIONS))
        {
            NUM_OF_GENERATIONS = std::stoi(results[1].str());
        }
        else if (regex_match(line, results, reNUM_OF_ELITE))
        {
            NUM_OF_ELITE = std::stoi(results[1].str());
        }
        else if (regex_match(line, results, reTOURNAMENT_SIZE))
        {
            TOURNAMENT_SIZE = std::stoi(results[1].str());
        }
        else if (regex_match(line, results, reNUM_OF_CROSSOVER_POINTS))
        {
            NUM_OF_CROSSOVER_POINTS = std::stoi(results[1].str());
        }
        else if (regex_match(line, results, reMUTATE_RATE))
        {
            MUTATION_RATE = std::stof(results[1].str());
        }
    }

#ifdef _DEBUG
    std::cout << "POPSIZE: " << POPSIZE << std::endl;
    std::cout << "CHROMOSOME: " << CHROMOSOME << std::endl;
    std::cout << "NUM_OF_GENERATIONS: " << NUM_OF_GENERATIONS << std::endl;
    std::cout << "NUM_OF_ELITE: " << NUM_OF_ELITE << std::endl;
    std::cout << "TOURNAMENT_SIZE: " << TOURNAMENT_SIZE << std::endl;
    std::cout << "NUM_OF_CROSSOVER_POINTS: " << NUM_OF_CROSSOVER_POINTS << std::endl;
    std::cout << "MUTATION_RATE: " << MUTATION_RATE << std::endl;
#endif // _DEBUG

    infile.close();

    return;
}

int Parameters::getPopsize() { return POPSIZE; }
int Parameters::getChromosome() { return CHROMOSOME; }
int Parameters::getNumOfGenerations() { return NUM_OF_GENERATIONS; }
int Parameters::getNumOfElite() { return NUM_OF_ELITE; }
int Parameters::getTournamentSize() { return TOURNAMENT_SIZE; }
int Parameters::getNumOfCrossoverPoints() { return NUM_OF_CROSSOVER_POINTS; }
float Parameters::getMutationRate() { return MUTATION_RATE; }

