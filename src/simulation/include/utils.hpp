#ifndef UTILS_HPP
#define UTILS_HPP

#include<vector>
#include<map>
#include<string>
#include<stdlib.h>

#include<agx/BallJoint.h>
#include<agx/DistanceJoint.h>
#include<agx/Hinge.h>
#include<agx/LockJoint.h>
#include<agx/UniversalJoint.h>
#include<agx/Prismatic.h>
#include<agx/PlaneJoint.h>
#include<agx/CylindricalJoint.h>
#include<agx/AngularLockJoint.h>
#include<agx/SplineJoint.h>
#include<agxVehicle/WheelJoint.h>
#include<agx/Constraint.h>

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

string generateAGXTypeFromName(string name) {

    /* Parse Constrains type based on a name given in table 14.1
     * https://www.algoryx.se/documentation/complete/agx/tags/latest/doc/UserManual/source/constraints.html#id7
     */

    if (name.find("BallJoint")) {
        return typeid(agx::BallJoint).name();
    } else if (name.find("DistanceJoint")) {
        return typeid(agx::DistanceJoint).name();
    } else if (name.find("Hinge")) {
        return typeid(agx::Hinge).name();
    } else if (name.find("UniversalJoint")) {
        return typeid(agx::UniversalJoint).name();
    } else if (name.find("LockJoint")) {
        return typeid(agx::LockJoint).name();
    } else if (name.find("Prismatic")) {
        return typeid(agx::Prismatic).name();
    } else if (name.find("PlaneJoint")) {
        return typeid(agx::PlaneJoint).name();
    } else if (name.find("CylindricalJoint")) {
        return typeid(agx::CylindricalJoint).name();
    } else if (name.find("AngularLockJoint")) {
        return typeid(agx::AngularLockJoint).name();
    } else if (name.find("SplineJoint")) {
        return typeid(agx::SplineJoint).name();
    } else if (name.find("WheelJoint")) {
        return typeid(agxVehicle::WheelJoint).name();
    } else {
        throw "Can not parse agx type from given name";
    }
}

#endif