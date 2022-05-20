#pragma once
// 標準ヘッダーのインクルード
#include <time.h>
#include <stdlib.h>
#include <limits>
#include <float.h>
#include <math.h>

#include "parameters.hpp"

// 0以上1以下の実数乱数
#define RAND_01 ((double) rand() / RAND_MAX)

class individual
{
private:
    int N = 0;
    float mutate_prob = 0.0;

public:
    // Data members
    int *chromosome; // chromosome
    int fitness; // fitness

    // Member functions
    individual(Parameters *prms);
    ~individual();
    void evaluate();
    void load_params();
    // Single-point crossover
    void apply_crossover_sp(individual *p1, individual *p2);
    // Two-point crossover
    void apply_crossover_tp(individual *p1, individual *p2);
    // Uniform corssover
    void apply_crossover_uniform(individual *p1, individual *p2);
    void mutate(); // mutation
};
