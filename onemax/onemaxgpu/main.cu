#include <stdio.h>
#include <stdlib.h>

#include "CUDAKernels.h"
#include "Misc.h"


int main()
{
    //- GPU用変数 idata: 入力、odata: 出力(総和) --------------------------------------------------
    // int *pdev_PopulationOdd;
    // int *pdev_PopulationEven;
    // int *pdev_Parent1;
    // int *pdev_Parent2;
	thrust::device_vector<int> dev_PopulationOdd(N);
	thrust::device_vector<int> dev_PopulationEven(N);
	thrust::device_vector<int> dev_Parent1(POPSIZE);
	thrust::device_vector<int> dev_Parent2(POPSIZE);
	thrust::device_vector<int> dev_Fitnesses(POPSIZE);
	thrust::device_vector<int> dev_SortedFitnesses(POPSIZE);
	thrust::device_vector<int> dev_SortedId(POPSIZE);

	int *pdev_PopulationOdd = thrust::raw_pointer_cast(&dev_PopulationOdd[0]);
	int *pdev_PopulationEven = thrust::raw_pointer_cast(&dev_PopulationEven[0]);
	int *pdev_Parent1 = thrust::raw_pointer_cast(&dev_Parent1[0]);
	int *pdev_Parent2 = thrust::raw_pointer_cast(&dev_Parent2[0]);
	int *pdev_Fitness = thrust::raw_pointer_cast(&dev_Fitnesses[0]);
	int *pdev_SortedFitness = thrust::raw_pointer_cast(&dev_SortedFitnesses[0]);
	int *pdev_SortedId = thrust::raw_pointer_cast(&dev_SortedId[0]);

    // cudaMalloc((void **)&pdev_PopulationOdd, Nbytes);
    // cudaMalloc((void **)&pdev_PopulationEven, Nbytes);
    // cudaMalloc((void **)&pdev_Parent1, POPSIZE * sizeof(int));
    // cudaMalloc((void **)&pdev_Parent2, POPSIZE * sizeof(int));

    //- CPU用変数 ---------------------------------------------------------------------------------
	// thrust::host_vector<int> host_Population(N);
	// int *phost_Population = thrust::raw_pointer_cast(&host_Population[0]);
    int *phost_Population;
	int *phost_Fitness;
	int *phost_SortedId;
	int *phost_Parent1;
	int *phost_Parent2;

    phost_Fitness       = (int *)malloc(POPSIZE * sizeof(int));
	phost_SortedId   = (int *)malloc(POPSIZE * sizeof(int));
	phost_Parent1       = (int *)malloc(POPSIZE * sizeof(int));
	phost_Parent2       = (int *)malloc(POPSIZE * sizeof(int));

	//- 乱数用変数 --------------------------------------------------------------------------------
	curandState *dev_TournamentStates;
	cudaMalloc((void **)&dev_TournamentStates, POPSIZE * TOURNAMENT_SIZE * 2 * NUM_OF_GENERATIONS * sizeof(curandState));

	curandState *dev_CrossoverStates;
	cudaMalloc((void **)&dev_CrossoverStates, POPSIZE * NUM_OF_CROSSOVER_POINTS * NUM_OF_GENERATIONS * sizeof(curandState));

	curandState *dev_MutationStates;
	cudaMalloc((void **)&dev_MutationStates, POPSIZE * CHROMOSOME * NUM_OF_GENERATIONS * sizeof(curandState));

	//- Preparation -------------------------------------------------------------------------------

    // CPU側でデータを初期化してGPUへコピー
	// thrust::generate(host_Population.begin(), host_Population.end(), my_rand);
    phost_Population = (int *)malloc(POPSIZE * CHROMOSOME * sizeof(int));
    initializePopulationOnCPU(phost_Population);
	for (int i = 0; i < POPSIZE; ++i)
	{
		for (int j = 0; j < CHROMOSOME; ++j)
		{
			printf("%d", phost_Population[i * CHROMOSOME + j]);
		}
		printf("\n");
	}
    cudaMemcpy(pdev_PopulationEven, phost_Population, Nbytes, cudaMemcpyHostToDevice);

	// --------------------------------
	// Main loop
	// --------------------------------

	// initialize random numbers array for tournament selection
	// 乱数はトーナメントセレクションで用いられるので、個体の数x2だけあれば良い
	setup_kernel<<<POPSIZE * NUM_OF_GENERATIONS, TOURNAMENT_SIZE * 2>>>(dev_TournamentStates);
	cudaDeviceSynchronize();

	setup_kernel<<<POPSIZE * NUM_OF_GENERATIONS, NUM_OF_CROSSOVER_POINTS>>>(dev_CrossoverStates);
	cudaDeviceSynchronize();

	setup_kernel<<<POPSIZE * NUM_OF_GENERATIONS, CHROMOSOME>>>(dev_MutationStates);
	cudaDeviceSynchronize();

	evaluation<<<POPSIZE, CHROMOSOME, CHROMOSOME*sizeof(int)>>>(pdev_PopulationEven, pdev_Fitness);
	cudaDeviceSynchronize();

	// dev_show<<<1, POPSIZE>>>(pdev_PopulationEven, pdev_Fitness, pdev_SortedFitness, pdev_Parent1, pdev_Parent2);
	// cudaDeviceSynchronize();

	// mutation<<<POPSIZE, CHROMOSOME>>>(pdev_PopulationEven, dev_MutationStates, 0);

	for (int gen = 0; gen < NUM_OF_GENERATIONS; ++gen)
	{
		printf("#####Gen: %d #######\n", gen);

		thrust::copy(thrust::device, dev_Fitnesses.begin(), dev_Fitnesses.end(), dev_SortedFitnesses.begin());
		thrust::sequence(dev_SortedId.begin(), dev_SortedId.end());
		thrust::sort_by_key(dev_SortedFitnesses.begin(), dev_SortedFitnesses.end(), dev_SortedId.begin()); 

		selection<<<1, POPSIZE>>>(
		// selection<<<N/POPSIZE, POPSIZE>>>(
				pdev_Fitness,
				pdev_SortedId,
				dev_TournamentStates,
				pdev_Parent1,
				pdev_Parent2,
				gen);
		cudaDeviceSynchronize();

		// dev_show<<<1, POPSIZE>>>(pdev_PopulationEven, pdev_Fitness, pdev_SortedFitness, pdev_Parent1, pdev_Parent2);
		// cudaDeviceSynchronize();

		if (gen % 2 == 0) // Even
		{
			crossover<<<1, POPSIZE, POPSIZE * sizeof(int) * 2>>>(
			// crossover<<<N/POPSIZE, POPSIZE, POPSIZE * sizeof(int) * 2>>>(
					pdev_PopulationEven,
					pdev_PopulationOdd,
					dev_CrossoverStates,
					pdev_Parent1,
					pdev_Parent2,
					gen);
			cudaDeviceSynchronize();

			mutation<<<POPSIZE, CHROMOSOME>>>(pdev_PopulationOdd, dev_MutationStates, gen);
			cudaDeviceSynchronize();

			evaluation<<<POPSIZE, CHROMOSOME, CHROMOSOME*sizeof(int)>>>(pdev_PopulationOdd, pdev_Fitness);
			cudaDeviceSynchronize();
		}
		else // Odd
		{
			crossover<<<1, POPSIZE, POPSIZE * sizeof(int) * 2>>>(
			// crossover<<<N/POPSIZE, POPSIZE, POPSIZE * sizeof(int) * 2>>>(
					pdev_PopulationOdd,
					pdev_PopulationEven,
					dev_CrossoverStates,
					pdev_Parent1,
					pdev_Parent2,
					gen);
			cudaDeviceSynchronize();

			mutation<<<POPSIZE, CHROMOSOME>>>(pdev_PopulationEven, dev_MutationStates, gen);
			cudaDeviceSynchronize();

			evaluation<<<POPSIZE, CHROMOSOME, CHROMOSOME*sizeof(int)>>>(pdev_PopulationEven, pdev_Fitness);
			cudaDeviceSynchronize();
		}
#ifdef _DEBUG
		cudaMemcpy(phost_Fitness,  pdev_Fitness,  POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(phost_SortedId, pdev_SortedId, POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(phost_Parent1,  pdev_Parent1,  POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(phost_Parent2,  pdev_Parent2,  POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
		if (gen % 2 == 0)
		{
			cudaMemcpy(phost_Population, pdev_PopulationOdd, Nbytes, cudaMemcpyDeviceToHost);
		}
		else
		{
			cudaMemcpy(phost_Population, pdev_PopulationEven, Nbytes, cudaMemcpyDeviceToHost);
		}
		showPopulationOnCPU(phost_Population, phost_Fitness, phost_Parent1, phost_Parent2);
#endif // _DEBUG
	}

    cudaMemcpy(phost_Fitness, pdev_Fitness, POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(phost_Parent1, pdev_Parent1, POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(phost_Parent2, pdev_Parent2, POPSIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(phost_Population, pdev_PopulationOdd, Nbytes, cudaMemcpyDeviceToHost);

	// cudaMemcpy(phost_Ranks, pdev_SortedId, POPSIZE * sizeof(int), cudaMemcpyHostToHost);

    // cudaFree(pdev_PopulationOdd);
    // cudaFree(pdev_PopulationEven);

    free(phost_Population);
	free(phost_Fitness);
	free(phost_SortedId);
	free(phost_Parent1);
	free(phost_Parent2);

    return 0;
}
