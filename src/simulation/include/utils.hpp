#ifndef UTILS_HPP
#define UTILS_HPP

#include<vector>
#include<map>
#include<string>

#include<agx/Name.h>

std::vector<std::string> convertNameVector(const std::vector<agx::Name> nameVector) {
    std::vector<std::string> converted;
    for (auto name : nameVector) {
        converted.push_back(name.str());
    }
    return converted;
} 

#endif