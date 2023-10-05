#ifndef BOOM_HPP
#define BOOM_HPP

#include <string>
#include <map>
#include <tuple>

#include <agxSDK/Assembly.h>
#include <agxCable/Cable.h>
#include <agxOSG/ExampleApplication.h>

#include "joints.hpp"
#include "cable.hpp"

class Boom : public agxSDK::Assembly
{
    std::vector<Joint> joints;
    std::vector<Cable> cables;

    Boom();
    Joint *const getJoint(const std::string &name);

public:
    void setJointVelocities(const std::map<std::string, std::vector<double>> &velocityMap);
    void setJointVelocity(const std::string &name, const std::vector<double> &velocity);
    void setJointVelocity(const std::size_t &jointIndex, const std::vector<double> &velocity);
    const std::vector<std::vector<double>> getJointPositions();
    const std::vector<double> getLowestCableNode();
    const std::vector<std::string> getJointNames();

    class Builder
    {
        agxSDK::AssemblyRef assemblyReference;
        std::vector<Joint> joints;
        std::vector<Cable> cables;

        bool isInitialized();
        template <typename T>
        Joint createJoint(std::string &name, T *constraintReference);

    public:
        Builder(agxSDK::AssemblyRef _assemblyReference) : assemblyReference(_assemblyReference){};
        template <typename T>
        Boom::Builder *addJoint(agx::Name &name);
        Boom::Builder *addCable(agx::Name &name);
        Boom *build();
    };
};

#endif