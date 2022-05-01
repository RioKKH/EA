#include <stdio.h>
#include <stdlib.h>
#include <bitset>

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#define POPSIZE 512
#define CHROMOSOME 512
#define NUM_OF_GENERATIONS 100
#define MUTATION_RATE 0.05
#define TOURNAMENT_SIZE 5
#define ELITISM true

#define N (POPSIZE * CHROMOSOME)
#define Nbytes (N*sizeof(int))
#define NT CHROMOSOME
#define NB POPSIZE
// #define NT (256)
// #define NB (N / NT) // 1より大きくなる

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {  \
	printf("Error at %s:%d\n", __FILE__, __LINE__); \
	return EXIT_FAILURE;}} while (0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) {  \
	printf("Error at %s:%d\n", __FILE__, __LINE__); \
	return EXIT_FAILURE;}} while (0)


__global__ void reduction(int *idata, int *odata)
{
    // スレッドと配列の要素の対応
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // スレッド番号
    int tx = threadIdx.x;
    int stride; // "隣"の配列要素まで距離

    // コンパイラの最適化を抑制
    // 複数のスレッドからアクセスされる変数に対する最適化
    // コンパイラが不要と判断して処理を削除してしまうことが有り、
    // 複数スレッドが変数の値をプライベートな領域にコピーして
    // 書き戻さない等が発生してしまう-->なのでvolatileを指定する
    // externを共有メモリの宣言に追加
    extern __shared__ volatile int s_idata[]; // 共有メモリの宣言

    s_idata[tx] = idata[i]; // グローバルメモリから共有メモリへデータをコピー
    __syncthreads(); // 共有メモリのデータは全スレッドから参照されるので同期を取る
    
    // ストライドを2倍し、ストライドがN/2以下ならループを継続
    // <<= : シフト演算の代入演算子 a <<= 1 --> a = a << 1と同じ
    // 最終stepではstrideが配列要素数のN/2となるので、strideがN/2
    // より大きくなるとループを中断
    for (stride = 1; stride <= blockDim.x/2; stride <<= 1)
    {
        // 処理を行うスレッドを選択
        if (tx % (2 * stride) == 0)
        {
            s_idata[tx] = s_idata[tx] + s_idata[tx + stride];
        }
        __syncthreads(); // スレッド間の同期を取る
        // stride = stride * 2; // ストライドを2倍して次のstepに備える
    }
    if (tx == 0) // 各ブロックのスレッド0が総和を出力用変数odataに書き込んで終了
    {
        odata[blockIdx.x] = s_idata[0];
    }
}

__host__ __device__ int getBest()
{
	return 0;
}

__global__ void setup_kernel(curandState *state)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state, float *result)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	float x;

	curandState localState = state[id];
	
	x = curand_uniform(&localState);

	state[id] = localState;
	result[id] = x;
}

__global__ void evaluation(int *population, int *fitness)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	int stride;

	extern __shared__ volatile int s_idata[];
	s_idata[tx] = population[i];
	__syncthreads();

	for (stride = 1; stride <= blockDim.x/2; stride <<= 1)
	{
		if (tx % (2 * stride) == 0)
		{
			s_idata[tx] = s_idata[tx] + s_idata[tx + stride];
		}
		__syncthreads();
	}

	if (tx == 0)
	{
		fitness[blockIdx.x] = s_idata[0];
	}
}

__device__ int tournamentSelection(int &population, curandState *dev_States, const int &id)
{
	int tournament[TOURNAMENT_SIZE];
	int randNum;
	for (int i = 0; i < TOURNAMENT_SIZE; ++i)
	{
		curandState localState = dev_States[id];
		randNum = curand_uniform(&localState) * (N -1);
		tournament[i] = population
	}
}


__global__ void selection(int* population, curandState *dev_States, int* parents)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	parents[i * 2] = tournamentSelection(*population, dev_States, i);
	parents[i * 2 + 1] = tournamentSelection(*population, dev_States, i);
}

__global__ void crossover()
{
}

__global__ void mutation()
{
}

int my_rand(void)
{
    static thrust::default_random_engine rng;
    static thrust::uniform_int_distribution<int> dist(0, 1);

    return dist(rng);
}

void initializePopulationOnCPU(int *population)
{
    thrust::generate(population, population + N, my_rand);

#ifdef _DEBUG
    for (int i=0; i<POPSIZE; ++i)
	{
		for (int j=0; j<CHROMOSOME; ++j)
		{
			std::cout << idata[i * POPSIZE + j];
		}
		std::cout << std::endl;
	}
#endif // _DEBUG
}

int main()
{
    // GPU用変数 idata: 入力、odata: 出力(総和)
    int *pdev_Population;
	thrust::device_vector<int> dev_Fitnesses(POPSIZE);
	thrust::device_vector<int> dev_Ranks(POPSIZE);

	int *pdev_Fitness = thrust::raw_pointer_cast(&dev_Fitnesses[0]);
	int *pdev_Ranks = thrust::raw_pointer_cast(&dev_Ranks[0]);
	thrust::sequence(dev_Ranks.begin(), dev_Ranks.end());

    cudaMalloc((void **)&pdev_Population, Nbytes);

    // CPU用変数
    int *phost_Population;
	int *phost_Fitness;
	int *phost_Ranks;

    phost_Fitness = (int *)malloc(NB * sizeof(int));
	phost_Ranks = (int *)malloc(NB * sizeof(int));

	// 乱数用変数
	curandState *dev_States;
	cudaMalloc((void **)&dev_States, NB * sizeof(curandState));
	// cudaMalloc((void **)&dev_States, POPSIZE * sizeof(curandState));
	cudaDeviceSynchronize();

    // CPU側でデータを初期化してGPUへコピー
    phost_Population = (int *)malloc(Nbytes);
    initializePopulationOnCPU(phost_Population);
    cudaMemcpy(pdev_Population, phost_Population, Nbytes, cudaMemcpyHostToDevice);

	// --------------------------------
	// Main loop
	// --------------------------------

	// initialize random numbers array for tournament selection
	// 乱数はトーナメントセレクションで用いられるので、個体の数だけあれば良い
	setup_kernel<<<NB, 1>>>(dev_States);
	cudaDeviceSynchronize();

	evaluation<<<NB, NT, NT*sizeof(int)>>>(pdev_Population, pdev_Fitness);

	for (int i = 0; i < NUM_OF_GENERATIONS; ++i)
	{
		// selection<<<NB, NT, NT*sizeof(int)>>>();
		// crossover<<<NB, NT>>>();
		// mutation<<<NB, NT>>>();
		// evaluation<<<NB, NT, NT*sizeof(int)>>>(pdev_Population, pdev_Fitness);
	}

    // cudaMemcpy(phost_Fitness, pdev_Fitness, NB * sizeof(int), cudaMemcpyDeviceToHost);
	// cudaMemcpy(phost_Ranks, pdev_Ranks, POPSIZE * sizeof(int), cudaMemcpyHostToHost);

#ifdef _DEBUG
    for (int i=0; i < POPSIZE; ++i)
	{
		std::cout << ph_indivRanks[i] << ":" << ph_indivFitness[i] << ",";
		if (i % 8 == 0)
		{
			std::cout << std::endl;
		}
	}
    // printf("\n%d, %d, %d\n", N, NT, NB);
#endif // _DEBUG

    // printf("sum = %d\n", sum);
    // cudaFree(pdev_Population);
    // cudaFree(pdev_Fitness); thrust
	// cudaFree(pdev_Ranks); thrust

    // free(phost_Population);
	// free(phost_Fitness);
	// free(phost_Ranks);

    return 0;
}
