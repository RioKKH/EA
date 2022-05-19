#include <iostream>
#include <fstream>
#include <string>

#include "Parameters.hpp"
#include "garegex.hpp"


Parameters::Parameters()
{
    loadParams();
}

Parameters::~Parameters()
{
}

void Parameters::loadParams()
{
    std::ifstream infile(PARAMNAME);
    std::string line;

    while (getline(infile, line))
    {
        if (regex_match(line, results, 
