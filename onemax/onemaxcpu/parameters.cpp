#include <iostream>
#include <fstream>
#include <string>

#include "parameters.hpp"
#include "garegex.hpp"


Parameters::Parameters()
{
    loadParams();
}


Parameters::~Parameters() {}


void Parameters::loadParams()
{
    std::ifstream infile(PARAMNAME);
    std::string line;
    std::smatch results;

    while (getline(infile, line)) {
        if (regex_match(line, results, GEN_MAX)) {
            gen_max = std::stoi(results[1].str());
        } else if (regex_match(line, results, POP_SIZE)) {
            pop_size = std::stoi(results[1].str());
        } else if (regex_match(line, results, ELITE)) {
            elite = std::stoi(results[1].str());
        } else if (regex_match(line, results, N)) {
            num_of_chromosome = std::stoi(results[1].str());
        } else if (regex_match(line, results, TOURNAMENT_SIZE)) {
            tournament_size = std::stoi(results[1].str());
        } else if (regex_match(line, results, MUTATE_PROB)) {
            mutate_prob = std::stof(results[1].str());
        }
    }

    std::cout << "GEN_MAX: " << gen_max << " "
              << "POP_SIZE: " << pop_size << " "
              << "ELITE: " << elite << " "
              << "N: " << num_of_chromosome << " "
              << "TOURNAMENT_SIZE: " << tournament_size << " "
              << "MUTATE_PROB: " << mutate_prob
              << std::endl;
    infile.close();

    return;
}

int Parameters::getGenMax() { return gen_max; }
int Parameters::getPopSize() { return pop_size; }
int Parameters::getElite() { return elite; }
int Parameters::getNumberOfChromosome() { return num_of_chromosome; }
int Parameters::getTournamentSize() { return tournament_size; }
float Parameters::getMutateProbability() { return mutate_prob; }

