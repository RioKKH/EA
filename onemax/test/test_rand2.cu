#include <thrust/random.h>
#include <thrust/generate.h>
#include <iostream>
#include <stdio.h>
#include <time.h>

#define N 512

int main(void)
{
    int *idata;
    idata = (int *)malloc(N * sizeof(int));

    thrust::default_random_engine rng(1337);
    thrust::uniform_int_distribution<int> dist(0,1);
    thrust::generate(idata, idata + N, [&] {return dist(rng);});

    for (int i=0; i<N; ++i)
    {
        std::cout << idata[i] << std::endl;
    }
}



