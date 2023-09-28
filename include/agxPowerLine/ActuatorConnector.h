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


#ifndef AGXPOWERLINE_ACTUATOR_CONNECTOR_H
#define AGXPOWERLINE_ACTUATOR_CONNECTOR_H

#include <agxPowerLine/Actuator.h>
#include <agxPowerLine/PowerLineUtils.h>
#include <agxPowerLine/detail/ActuatorConstraints.h>

namespace agx
{
  class RigidBody;
}

namespace agxPowerLine
{
  class PhysicalDimension;
}

namespace agxPowerLine
{

  AGX_DECLARE_POINTER_TYPES(ActuatorConnector);

  class AGXMODEL_EXPORT ActuatorConnector : public agxPowerLine::Connector
  {
  public:
    ActuatorConnector( agxPowerLine::Actuator* actuator );

    agxPowerLine::Actuator* getActuator();
    const agxPowerLine::Actuator* getActuator() const;

    // The Actuator could connect its two units in normal or reversed order
    // This function is to find if this specific connector is connected in reversed order
    virtual bool getReverseOrder() const = 0;

    Side getRelativeIndex( Side index ) const;

    ActuatorConstraintImplementation* getActuatorConstraint();

    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxPowerLine::ActuatorConnector);
    void store(agxStream::OutputArchive& out) const override;
    void restore(agxStream::InputArchive& in) override;

  protected:
    ActuatorConnector();
    virtual ~ActuatorConnector();

  protected:
    agx::observer_ptr<agxPowerLine::Actuator> m_actuator;
  };
}

#endif
