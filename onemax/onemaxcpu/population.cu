// #ifndef POPULATION_H
// #define POPULATION_H

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>

class Population
{
private:
    int PopulationSize;
    int ChromosomeSize;

public:
    Population();
    Population(const int PopulationSize, const int ChromosomeSize);
    ~Population();

    void show() const;
};

Population::Population(const int PopulationSize, const int ChromosomeSize)
    : PopulationSize{PopulationSize}, ChromosomeSize(ChromosomeSize)
{
}

Population::~Population()
{
    std::cout << "deconstructor: Population" << std::endl;
}



void Population::show() const
{
    std::cout << "(" << PopulationSize << "," << ChromosomeSize << ")" << std::endl;
}

int my_rand(void)
{
    static thrust::default_random_engine rng;
    static thrust::uniform_int_distribution<int> dist(0, 1);
    return dist(rng);
}

struct init
{
    int a, b;

    __host__ __device__
    init(int _a=0, int _b=1) : a(_a), b(_b)
    {
    }

    __host__ __device__
    int operator() (const unsigned int n) const
    {
        thrust::default_random_engine rng;
        thrust::uniform_int_distribution<int> dist(a, b);
        // rng.discard(n);

        return dist(rng);
    }
};

int main()
{
    // Population population(10, 10);
    thrust::host_vector<int> h_vec(10);
    // thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    thrust::generate(h_vec.begin(), h_vec.end(), my_rand);

    // thrust::device_vector<int> d_vec = h_vec;
    // thrust::transform(h_vec.begin(), h_vec.end(), h_vec.begin(), init());

    for (const auto& a : h_vec)
    {
        std::cout << a;
    }
    std::cout << std::endl;

    return 0;
}

// #endif// 
