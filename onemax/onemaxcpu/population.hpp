#include "individual.hpp"

class population
{
public:
    // population();
    population(Parameters *prms);
    ~population();
    void alternate();            // 世代交代をする
    void print_result();          // 結果を表示する

    individual **ind;            // 現世代の個体群のメンバ

private:
    // Data members
    int gen_max = 0;
    int pop_size = 0;
    int elite = 0;
    int N = 0;
    int tournament_size = 0;
    float mutate_prob = 0.0;

    // member functions
    void evaluate();             // 個体を評価する
    int select_by_tournament();  // Tournament selection
    int select_by_roulette();    // Roulette selection
    void sort(int lb, int ub);   // 個体を良い順に並び替える
    individual **next_ind;        // 次世代の個体群のメンバ
    double *tr_fit; // converted fitness value
    // double tr_fit[POP_SIZE]; // converted fitness value
    double denom;  // denominator used for roulette selection
};

// double trFit[POP_SIZE]; // 適応度を変換した値
// double denom;           // ルーレット選択の確率を求めるときの分母
