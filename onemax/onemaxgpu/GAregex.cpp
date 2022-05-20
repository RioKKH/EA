#include "GAregex.hpp"

std::regex rePOPSIZE(R"(^POPSIZE\s+(\d+)$)");
std::regex reCHROMOSOME(R"(^CHROMOSOME\s+(\d+)$)");
std::regex reNUM_OF_GENERATIONS(R"(^NUM_OF_GENERATIONS\s+(\d+)$)");
std::regex reNUM_OV_ELITE(R"(^NUM_OF_ELITE\s+(\d+)$)");
std::regex reTOURNAMENT_SIZE(R"(^TOURNAMENT_SIZE\s+(\d+)$)");
std::regex reNUM_OF_CROSSOVER_POINTS(R"(^NUM_OF_CROSSOVER_POINTS\s+(\d+)$)");
std::regex reMUTATE_RATE(R"(^MUTATE_PROB\s+(\d\.\d+)$)");                                         
