#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP #include <string>

#include <string>

/**
 * @struct EvolutionParameters
 * @brief  Parameters of the evolutionary process.
 */
struct EvolutionParameters
{
    int POPSIZE = 0;
    int CHROMOSOME = 0;
    int NUM_OF_GENERATIONS;
    int NUM_OF_ELITE;
    int TOURNAMENT_SIZE;
    int NUM_OF_CROSSOVER_POINTS;
    float MUTATION_RATE;
    int N;
    int Nbytes;
};


/**
 * @class Parameters
 * @blief Singleton class with Parameters maintaining them in CPU and GPU constant memory.
 */
class Parameters { private:
    const std::string PARAMNAME = "onemax.prms";
    EvolutionParameters mEvolutionParameters;

    // Singletonとする
    // Parameters() = default;
    Parameters()
    {
        loadParams();
    }
    ~Parameters() = default;

public:
    Parameters(const Parameters& obj) = delete; // コピーコンストラクタをdelete指定
    Parameters& operator=(const Parameters& obj) = delete; // コピー代入演算子もdelete指定
    Parameters(Parameters&&) = delete; // ムーブコンストラクターもdelete指定
    Parameters& operator=(Parameters&&) = delete; // ムーブ代入演算子をdelete指定

    static Parameters& getInstance();

    void copyToDevice(EvolutionParameters gpuprms);
    void loadParams();
    int getPopsize() const;
    int getChromosome() const;
    int getNumOfGenerations() const;
    int getNumOfElite() const;
    int getTournamentSize() const;
    int getNumOfCrossoverPoints() const;
    float getMutationRate() const;
    int getN() const;
    int getNbytes() const;
};

#endif // PARAMETERS_HPP
