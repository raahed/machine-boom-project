/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGXPOWERLINE_ACTUATOR_CONSTRAINTS
#define AGXPOWERLINE_ACTUATOR_CONSTRAINTS

#include <agxPowerLine/PowerLineConstraints.h>

namespace agxPowerLine
{
  class Actuator1DOF;
  class ActuatorConnector;
}

namespace agxPowerLine
{
  class AGXMODEL_EXPORT ElementaryActuatorConstraint :
      public agxPowerLine::ElementaryPhysicalDimensionMultiBodyConstraint
  {
    public:
    ElementaryActuatorConstraint();
    ActuatorConnector* getActuatorConnector();
  };


  class AGXMODEL_EXPORT ActuatorConstraintImplementation :
      public agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation
  {
  public:
    ActuatorConstraintImplementation();
    ActuatorConstraintImplementation(ElementaryPhysicalDimensionConstraint* elementaryConstraint, Actuator1DOF* owningActuator);
    void setOwningActuator(Actuator1DOF* owningActuator);
    virtual bool updateValid() override;
    virtual void prepare() override;

    AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxPowerLine::ActuatorConstraintImplementation);
    virtual void store(agxStream::OutputArchive& out) const override;
    virtual void restore(agxStream::InputArchive& in) override;

    private:
      Actuator1DOF* m_owningActuator;
  };
}

#endif
