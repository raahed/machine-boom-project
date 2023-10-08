#include <chrono>
#include "random_generator.hpp"

// From: https://stackoverflow.com/questions/9878965/rand-between-0-and-1

UniformRandomGenerator::UniformRandomGenerator() {
    initialSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    uint32_t full = -1;
    std::seed_seq seedSequence{uint32_t(initialSeed & full), std::uint32_t(initialSeed >> sizeof(uint32_t))};
    rng.seed(seedSequence);
}

UniformRandomGenerator::UniformRandomGenerator(uint64_t _initialSeed) : initialSeed(_initialSeed) {
    uint32_t full = -1;
    std::seed_seq seedSequence{uint32_t(initialSeed & full), std::uint32_t(initialSeed >> sizeof(uint32_t))};
    rng.seed(seedSequence);
}

template<typename T>
T UniformRandomGenerator::generateRandomNumber(const int &intervalStart, const int &intervalEnd) {
    std::uniform_real_distribution <T> unif(intervalStart, intervalEnd);
    return unif(rng);
}