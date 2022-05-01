#include <stdio.h>
#include <stdlib.h>
#include <bitset>

#include <curand.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#define POPSIZE 512
#define CHROMOSOME 512
#define NUM_OF_GENERATIONS 100

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

__device__ int tournamentSelection()
{
}

__host__ __device__ int getBest()
{
	return 0;
}

__global__ void evaluation(int *idata, int *odata)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	int stride;

	extern __shared__ volatile int s_idata[];
	s_idata[tx] = idata[i];
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
		odata[blockIdx.x] = s_idata[0];
	}
}

__global__ void selection()
{
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

void initializePopulationOnCPU(int *idata)
{
    thrust::generate(idata, idata + N, my_rand);

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
    int *idata, *odata;
	thrust::device_vector<int> d_indivFitnesses(POPSIZE);
	thrust::device_vector<int> d_indivRanks(POPSIZE);
	int *pd_indivFitness = thrust::raw_pointer_cast(&d_indivFitnesses[0]);
	int *pd_indivRanks = thrust::raw_pointer_cast(&d_indivRanks[0]);
	thrust::sequence(d_indivRanks.begin(), d_indivRanks.end());

    // CPU用変数
    int *host_idata;
	int *ph_indivFitness;
	int *ph_indivRanks;

    ph_indivFitness = (int *)malloc(NB * sizeof(int));
	ph_indivRanks = (int *)malloc(NB * sizeof(int));


    cudaMalloc((void **)&idata, Nbytes);
    cudaMalloc((void **)&odata, NB*sizeof(int)); // ブロックの数だけ部分和が出るので

    // CPU側でデータを初期化してGPUへコピー
    host_idata = (int *)malloc(Nbytes);
    initializePopulationOnCPU(host_idata);
    cudaMemcpy(idata, host_idata, Nbytes, cudaMemcpyHostToDevice);

    // 共有メモリサイズを指定
	evaluation<<<NB, NT, NT*sizeof(int)>>>(idata, pd_indivFitness);

	for (int i = 0; i < NUM_OF_GENERATIONS; ++i)
	{
		selection<<<NB, NT, NT*sizeof(int)>>>();
		crossover<<<NB, NT>>>();
		mutation<<<NB, NT>>>();
		evaluation<<<NB, NT, NT*sizeof(int)>>>(idata, pd_indivFitness);
	}

    cudaMemcpy(ph_indivFitness, pd_indivFitness, NB * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(ph_indivRanks, pd_indivRanks, POPSIZE * sizeof(int), cudaMemcpyHostToHost);

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
    cudaFree(idata);
    cudaFree(odata);

    free(host_idata);
	free(ph_indivFitness);
	free(ph_indivRanks);

    return 0;
}
