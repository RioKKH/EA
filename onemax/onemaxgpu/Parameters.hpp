#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <string>

/**
 * @struct EvolutionParameters
 * @brief  Parameters of the evolutionary process.
 */
typedef struct
{
    int POPSIZE;
    int CHROMOSOME;
    int NUM_OF_GENERATIONS;
    int NUM_OF_ELITE;
    int TOURNAMENT_SIZE;
    int NUM_OF_CROSSOVER_POINTS;
    float MUTATION_RATE;
    int N;
    int Nbytes;
} EvolutionParameters;

/**
 * @class Parameters
 * @blief Singleton class with Parameters maintaining them in CPU and GPU constant memory.
 */
class Parameters {
private:
    const std::string PARAMNAME = "onemax.prms";
    EvolutionParameters cpuEvoPrms;

public:
    explicit Parameters() {}
    ~Parameters() {}

    void loadParams(void);
    int getPopsize(void) const;
    int getChromosome(void) const;
    int getNumOfGenerations(void) const;
    int getNumOfElite(void) const;
    int getTournamentSize(void) const;
    int getNumOfCrossoverPoints(void) const;
    float getMutationRate(void) const;
    int getN(void) const;
    int getNbytes(void) const;
    EvolutionParameters getEvoPrms(void) const;
    void copyToDevice();
};

#endif // PARAMETERS_HPP
