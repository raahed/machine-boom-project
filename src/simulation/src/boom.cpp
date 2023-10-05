#include <exception>
#include <sstream>
#include <limits>
#include <type_traits>

#include "boom.hpp"
#include "cable.hpp"

void Boom::setJointVelocities(const std::map<std::string, std::vector<double>> &velocityMap)
{
    for (auto joint : joints)
    {
        auto jointName = joint.getName();
        if (velocityMap.find(jointName) != velocityMap.end())
            joint.setVelocities(velocityMap.at(jointName));
    }
}

void Boom::setJointVelocity(const std::string &name, const std::vector<double> &velocities)
{
    for (auto joint : joints)
    {
        if (name == joint.getName())
        {
            joint.setVelocities(velocities);
            break;
        }
    }
}

void Boom::setJointVelocity(const std::size_t &jointIndex, const std::vector<double> &velocity) {
    joints[jointIndex].setVelocities(velocity);
}

const std::vector<std::vector<double>> Boom::getJointPositions()
{
    std::vector<std::vector<double>> jointPositions;
    for (auto joint : joints)
    {
        jointPositions.push_back(joint.getAngles());
    }
    return jointPositions;
}

const std::vector<std::string> Boom::getJointNames()
{
    std::vector<std::string> names;
    for (auto joint : joints)
    {
        names.push_back(joint.getName());
    }
    return names;
}

Joint *const Boom::getJoint(const std::string &name)
{
    for (auto joint : joints)
    {
        if (joint.getName() == name)
            return &joint;
    }
    std::stringstream errorStream;
    errorStream << "The name: " << name << " is not a valid joint on this boom please check the name again and verify its part of the boom!";
    throw std::invalid_argument(errorStream.str());
}

const std::vector<double> Boom::getLowestCableNode()
{
    std::vector<double> lowestNode{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    for (Cable cable : cables)
    {
        std::vector<double> cableLowestNode = cable.getLowestNode();
        if (cableLowestNode[2] < lowestNode[2])
            lowestNode = cableLowestNode;
    }
    return lowestNode;
}

template <typename T>
Boom::Builder *Boom::Builder::addJoint(agx::Name &name)
{
    static_assert(std::is_base_of<agx::Constraint, T>::value, "The type parameter of Boom::Builder::addJoint() must derive from agx::Constraint");

    agx::Constraint *constraintReference = assemblyReference->getConstraint(name);
    constraintReference->setEnable(true);

    if (constraintReference != NULL)
    {
        T *concreteConstraintRef = constraintReference->as<T>();
        joints.push_back(createJoint(name.str(), constraintReference));
    }

    return this;
}

Boom::Builder *Boom::Builder::addCable(agx::Name &name)
{
    agxSDK::Assembly *tmp = NULL;
    tmp = assemblyReference->getAssembly(name);
    if (tmp != NULL)
    {
        auto cableRef = dynamic_cast<agxCable::Cable *>(tmp);
        auto cable = static_cast<Cable &>(*cableRef);
        cables.push_back(cable);
    }
    return this;
}

Boom *Boom::Builder::build()
{
    if (!isInitialized())
    {
        throw std::invalid_argument("Please provide a boom with cables and joints and make sure they exist in the simulation before building a Boom object!");
    }
    Boom *myBoom = assemblyReference->as<Boom>();
    myBoom->joints = joints;
    myBoom->cables = cables;
    return myBoom;
}

template <typename T>
Joint Boom::Builder::createJoint(std::string &name, T *constraintReference)
{
    Joint joint;
    if (std::is_base_of<agx::Constraint1DOF, T>::value)
    {
        joint = Joint1DOF<T>(name.str(), *constraintReference);
    }
    else
    {
        joint = Joint2DOF<T>(name.str(), *constraintReference);
    }
    return joint;
}

bool Boom::Builder::isInitialized()
{
    return assemblyReference != NULL && joints.size() && cables.size();
}
