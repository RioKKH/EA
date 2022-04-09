#pragma once

#include <regex>

std::regex GEN_MAX(R"(^GEN_MAX\s+(\d+)$)");
std::regex POP_SIZE(R"(^POP_SIZE\s+(\d+)$)");
std::regex ELITE(R"(^ELITE\s+(\d+)$)");
std::regex N(R"(^N\s+(\d+)$)");
std::regex TOURNAMENT_SIZE(R"(^TOURNAMENT_SIZE\s+(\d+)$)");
std::regex MUTATE_PROB(R"(^MUTATE_PROB\s+(\d\.\d+)$)");
