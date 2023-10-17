#ifndef UTILS_HPP
#define UTILS_HPP

#include<vector>
#include<map>
#include<string>
#include<stdlib.h>
#include <stdexcept>

#include "global.hpp"

using namespace std;

vector <string> getListEnv(string key) {

    char *env = getenv(key.c_str());

    if (env == nullptr) return vector < string > {};

    vector <string> result;
    string t;
    stringstream es(env);

    while (getline(es, t, ','))
        result.push_back(t);

    return result;
}

JType guessJTypeFromString(string &name) {

    if (name.find("Prismatic") != string::npos) {
        return PRISMATIC;
    } else if (name.find("Hinge") != string::npos) {
        return ROTARY;
    } else if (name.find("Cylindrical") != string::npos) {
        return CYLINDRICAL;
    } else {
        throw invalid_argument("Unknown type");
    }
}

#endif //MACHINE_BOOM_PROJECT_UTILS_HPP
