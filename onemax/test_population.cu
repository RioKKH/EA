#include <stdio.h>
#include <stdlib.h>

#include <thrust/random.h>
#include <thrust/generate.h>

#define POPSIZE 512
#define CHROMOSOME 512

#define N (POPSIZE * CHROMOSOME)
#define Nbytes (N*sizeof(int))
#define NT CHROMOSOME
#define NB POPSIZE
// #define NT (256)
// #define NB (N / NT) // 1より大きくなる

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
    __syncthreads; // 共有メモリのデータは全スレッドから参照されるので同期を取る
    
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

int my_rand(void)
{
    static thrust::default_random_engine rng;
    static thrust::uniform_int_distribution<int> dist(0, 1);

    return dist(rng);
}

void init_thrust(int *idata)
{
    thrust::generate(idata, idata + N, my_rand);

#ifdef _DEBUG
    for (int i=0; i<N; ++i)
    {
        std::cout << idata[i] << ",";
    }
    std::cout << std::endl;
#endif // _DEBUG
}

void init(int *idata)
{
    int i;
    for (i=0; i<N; ++i)
    {
        idata[i] = 1;
    }
}

int main()
{
    // GPU用変数 idata: 入力、odata: 出力(総和)
    int *idata, *odata;

    // CPU用変数 host_idata: 初期化用、sum: 総和
    int *host_idata, *sum;

    cudaMalloc((void **)&idata, Nbytes);
    cudaMalloc((void **)&odata, NB*sizeof(int)); // ブロックの数だけ部分和が出るので

    // CPU側でデータを初期化してGPUへコピー
    host_idata = (int *)malloc(Nbytes);
    init_thrust(host_idata);
    // init(host_idata);
    cudaMemcpy(idata, host_idata, Nbytes, cudaMemcpyHostToDevice);
    free(host_idata);

    // 各ブロックで部分和を計算。iが入力、oが出力
    // 共有メモリサイズを指定
    reduction<<<NB, NT, NT*sizeof(int)>>>(idata, odata);
    // if (NB > 1)
    // {
    //     // 共有メモリサイズを指定
    //     reduction<<<1, NB, NB*sizeof(int)>>>(odata, odata); // 部分和から総和を計算。oが入出力
    // }

    sum = (int *)malloc(NB * sizeof(int));

    cudaMemcpy(sum, odata, NB * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < NB; ++i)
    {
        printf("%d\n", sum[i]);
    }

#ifdef _DEBUG
    printf("\n%d, %d, %d\n", N, NT, NB);
#endif // _DEBUG

    // printf("sum = %d\n", sum);
    cudaFree(idata);
    cudaFree(odata);

    return 0;
}
