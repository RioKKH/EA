#ifndef EVOLUTION_H
#define EVOLUTION_H

class GPUEvolution
{
public:
    /// Class constructor.
    GPUEvolution();

    /// Copy constructor is not allowed.
    GPUEvolution(const GPUEvolution&) = delete;

    /// Destructor
    /// クラスに仮想メンバー関数が存在する場合、そのクラスのデストラクタは
    /// virtualでなければならない
    virtual ~GPUEvolution();

    /// Run evolution
    void run();

protected:

    /// Initialize evolution.
    void initialize();

    /// Run evolution
    void runEvolutionCycle();

    /// Init random generator seed;
    void initRandomSeed();

    /// Get random generator seed and increment it.
    unsigned int getRandomSeed() { return mRandomSeed++; };


    /// Parameters of evolution
    Parameters& mParams;

    /// Actual generation.
    // int mActGeneration;

    /// Number of SM on GPU.
    // int mMultiprocessorCount;

    /// Device Id.
    int mDeviceIdx;

    /// Random Generator Seed.
    unsigned int mRandomSeed;

}; // end of GPU_Evolution

#endif // EVOLUTION_H
