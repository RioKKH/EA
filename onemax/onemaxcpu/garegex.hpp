#pragma once

#include <regex>

const std::regex GEN_MAX(R"(^GEN_MAX\s+(\d+)$)");
const std::regex POP_SIZE(R"(^POP_SIZE\s+(\d+)$)");
const std::regex ELITE(R"(^ELITE\s+(\d+)$)");
const std::regex N(R"(^N\s+(\d+)$)");
const std::regex TOURNAMENT_SIZE(R"(^TOURNAMENT_SIZE\s+(\d+)$)");
const std::regex MUTATE_PROB(R"(^MUTATE_PROB\s+(\d\.\d+)$)");
