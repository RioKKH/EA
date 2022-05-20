#include <iostream>
#include <vector>
#include <random>
//#include <stdio.h>
#include <limits.h>

using namespace std;

#define POP_SIZE (1000)

int main(int argc, char const* argv[])
{
    vector<double> vec;
    std::random_device rd;
    std::mt19937 mt(rd());
    uniform_real_distribution<double> score(0.0, 10.0);
    srand((unsigned int)time(NULL));

    int r = rand();
    int rr = r << 16;
    int denom = POP_SIZE * (POP_SIZE + 1) / 2;
    int rrr = ((rand() << 16) + (rand() << 1) + (rand() % 2)) % denom + 1;
    /*
    cout << "rand: " << r << " "
         << "r << 16 " << rr << " "
         << "denom " << denom << " "
         << "rrr " << rrr << " " << endl;
    */

    cout << "max of unsigned int: " << UINT_MAX << " "
         << "rnad max: " << RAND_MAX  << " "
         << "ratio: " << (double)RAND_MAX / UINT_MAX << endl;

    for (int i = 0; i < 10; i++) {
        vec.push_back(score(mt));
    }

    cout << "sum = " << accumulate(vec.begin(), vec.end(), 0.0) << endl;

    return 0;
}
