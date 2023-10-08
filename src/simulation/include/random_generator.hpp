#ifndef RANDOM_GENERATOR_HPP
#define RANDOM_GENERATOR_HPP

#include <random>

class UniformRandomGenerator {
    std::mt19937_64 rng;
    uint64_t initialSeed;

public:
    UniformRandomGenerator();

    UniformRandomGenerator(uint64_t _initialSeed);

    template<typename T>
    T generateRandomNumber(const int &intervalStart, const int &intervalEnd);
};

#endif