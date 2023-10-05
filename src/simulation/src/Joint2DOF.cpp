#include "joints.hpp"

template <class T>
void Joint2DOF<T>::setVelocities(const std::vector<double> &newVelocities)
{
    if (newVelocities.size() != 2)
    {
        throw std::invalid_argument("For two degrees of freedom pass two velocities!");
    }
    this->setVelocity(agx::Constraint2DOF::DOF::FIRST, newVelocities[0]);
    this->setVelocity(agx::Constraint2DOF::DOF::SECOND, newVelocities[1]);
}

template <class T>
void Joint2DOF<T>::setVelocity(agx::Constraint2DOF::DOF dof, const double &newVelocity)
{
    constraint.getMotor1D(dof)->setEnable(true);
    constraint.getMotor1D(dof)->setLocked(false);
    constraint.getMotor1D(dof)->setSpeed(newVelocity);
}

template <class T>
const std::vector<double> Joint2DOF<T>::getVelocities()
{
    std::vector<double> velocities{this->getVelocity(agx::Constraint2DOF::DOF::FIRST), this->getVelocity(agx::Constraint2DOF::DOF::SECOND)};
    return velocities;
}

template <class T>
const double Joint2DOF<T>::getVelocity(agx::Constraint2DOF::DOF dof)
{
    return constraint.getMotor1D(dof)->getSpeed();
}

template <class T>
const std::vector<double> Joint2DOF<T>::getAngles()
{
    std::vector<double> angles{this->getAngle(agx::Constraint2DOF::DOF::FIRST), this->getAngle(agx::Constraint2DOF::DOF::SECOND)};
    return angles;
}

template <class T>
const double Joint2DOF<T>::getAngle(agx::Constraint2DOF::DOF dof)
{
    return constraint.getMotor1D(dof)->getAngle();
}

template <class T>
const std::string Joint2DOF<T>::getName()
{
    return name;
}