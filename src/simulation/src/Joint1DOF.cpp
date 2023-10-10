#include "joints.hpp"

template<class T>
void Joint1DOF<T>::setVelocities(const std::vector<double> &newVelocities) {
    if (newVelocities.size() != 1) {
        throw std::invalid_argument("For one degree of freedom pass one velocity!");
    }
    constraint.getMotor1D()->setEnable(true);
    constraint.getMotor1D()->setLocked(false);
    constraint.getMotor1D()->setSpeed(newVelocities[0]);
};

template<class T>
const std::vector<double> Joint1DOF<T>::getVelocities() {
    return constraint.getMotor1D()->getSpeed();
};

template<class T>
const std::vector<double> Joint1DOF<T>::getAngles() {
    std::vector<double> angles{constraint.getAngle()};
    return angles;
};

template<class T>
const agx::String Joint1DOF<T>::getName() {
    return name;
};