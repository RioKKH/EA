#include <thrust/random.h>
#include <thrust/generate.h>
#include <iostream>
#include <stdio.h>
#include <time.h>

#define N 512

int my_rand(void)
{
    static thrust::default_random_engine rng(1337);
    static thrust::uniform_int_distribution<int> dist(0, 10);

    return dist(rng);
}

int main(void)
{
    int *idata;
    idata = (int *)malloc(N * sizeof(int));

    thrust::generate(idata, idata + N, my_rand);

    for (int i=0; i<N; ++i)
    {
        std::cout << idata[i] << std::endl;
    }
}



