#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <string>

class Parameters
{
private:
    const std::string PARAMNAME = "onemax.prms";
    int POPSIZE = 0;
    int CHROMOSOME = 0;
    int NUM_OF_GENERATIONS = 0;
    int NUM_OF_ELITE = 0;
    int TOURNAMENT_SIZE = 0;
    int NUM_OF_CROSSOVER_POINTS = 0;
    float MUTATION_RATE = 0.0f;
    int N = 0;
    int Nbytes = 0;

public:
    explicit Parameters();
    ~Parameters();

    void loadParams();
    int getPopsize();
    int getChromosome();
    int getNumOfGenerations();
    int getNumOfElite();
    int getTournamentSize();
    int getNumOfCrossoverPoints();
    float getMutationRate();
    int getN();
    int getNbytes();
};

#endif // PARAMETERS_HPP
