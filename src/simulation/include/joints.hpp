#ifndef JOINTS_HPP
#define JOINTS_HPP

#include <string>
#include <vector>
#include <tuple>
#include <agx/Constraint.h>
#include "constraints.hpp"

class Joint {
public:
    virtual void setVelocities(const std::vector<double> &newVelocities) = 0;

    virtual const std::vector<double> getVelocities() = 0;

    virtual const std::vector<double> getAngles() = 0;

    virtual const std::string getName() = 0;
};

template<class T>
class Joint1DOF : Joint, Convertible_to<T, agx::Constraint1DOF> {
    std::string name;
    T constraint;

public:
    Joint1DOF(std::string _name, T _constraint) : name(_name), constraint(_constraint) {};

    ~Joint1DOF() {
        delete name;
        delete constraint;
    };

    void setVelocities(const std::vector<double> &newVelocities);

    const std::vector<double> getVelocities();

    const std::vector<double> getAngles();

    const std::string getName();
};

template<class T>
class Joint2DOF : Joint, Convertible_to<T, agx::Constraint2DOF> {
    std::string name;
    T constraint;

    void setVelocity(agx::Constraint2DOF::DOF dof, const double &newVelocity);

    const double getVelocity(agx::Constraint2DOF::DOF dof);

    const double getAngle(agx::Constraint2DOF::DOF dof);

public:
    Joint2DOF(std::string _name, T _constraint) : name(_name), constraint(_constraint) {
        constraint.getMotor1D(agx::Constraint2DOF::DOF::FIRST)->setEnable(true);
        constraint.getMotor1D(agx::Constraint2DOF::DOF::FIRST)->setLocked(true);
        constraint.getMotor1D(agx::Constraint2DOF::DOF::SECOND)->setEnable(true);
        constraint.getMotor1D(agx::Constraint2DOF::DOF::SECOND)->setLocked(true);
    };

    ~Joint2DOF() {
        delete name;
        delete constraint;
    };

    void setVelocities(const std::vector<double> &newVelocities);

    const std::vector<double> getVelocities();

    const std::vector<double> getAngles();

    const std::string getName();
};

#endif