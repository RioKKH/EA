#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

// cpp
#include <math.h>

// thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// EA
 #include "parameters.hpp"

class individual
{
private:
    int N = 0;
    float mutate_prob = 0.0;

public:
    // Data members
    thrust::host_vector<int> h_chromosome;
    int fitness;

    // Member functions
    individual();
    individual(Parameters *prms);
    ~individual();

    void evaluate();
    void load_params();

    // Single-point crossover
    void apply_corssover_sp(individual *p1, individual *p2);

    // Two-point crossover
    void apply_corssover_tp(individual *p1, individual *p2);

    // Uniform crossover
    void apply_corssover_uniform(individual *p1, individual *p2);

    // mutation
    void mutate();
};

#endif // INDIVIDUAL_H
