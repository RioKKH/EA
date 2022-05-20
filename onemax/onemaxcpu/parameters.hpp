#pragma once
// #ifndef PARAMETERS_H
// #define PARAMETERS_H

#include <string>


class Parameters {
private:
    std::string PARAMNAME = "onemax_prms.dat";
    int gen_max = 0;
    int pop_size = 0;
    int elite = 0;
    int num_of_chromosome = 0;
    int tournament_size = 0;
    float mutate_prob = 0.0;

public:
    // Data members

    // Member functions
    Parameters();
    ~Parameters();

    void loadParams();
    int getGenMax();
    int getPopSize();
    int getElite();
    int getNumberOfChromosome();
    int getTournamentSize();
    float getMutateProbability();
};

// #endif /* PARAMETERS_H */
