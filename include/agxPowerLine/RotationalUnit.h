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

#ifndef AGXPOWERLINE_ROTATIONAL_UNIT_H
#define AGXPOWERLINE_ROTATIONAL_UNIT_H

#include <agxModel/export.h>
#include <agxPowerLine/RotationalDimension.h>
#include <agxPowerLine/Unit.h>

namespace agxPowerLine
{
  /**
  A rotational unit has one degree of freedom. The rotational degree of freedom.
  */
  class AGXMODEL_EXPORT RotationalUnit : public agxPowerLine::Unit
  {
  public:
    /**
    Create a rotational unit.
    */
    RotationalUnit();


    RotationalUnit(agx::RigidBody* body, agx::UInt8 slot);

    /**
    returns the amount of torque delivered from the rotational dimension
    */
    virtual agx::Real getDeliveredTorque() const;

    /**
    returns the number of radians per second
    */
    agx::Real getAngularVelocity() const;

    /**
    returns the number of revolutions per minute
    */
    agx::Real getRPM() const;

    /**
    set the angular velocity damping around the local axis (direction)
    */
    void setAngularVelocityDamping( agx::Real damping );

    /**
    returns a pointer to the rotational dimension
    */
    agxPowerLine::RotationalDimension* getRotationalDimension() const;

    /**
    set the inertia for the rotational dimension of the rotational unit.
    */
    virtual void setInertia( agx::Real inertia );

    virtual agx::Real getInertia() const;

    /**
    returns a vector in world coordinates that defines the rotational axis of the rotational unit.
    */
    virtual agx::Vec3 getDirection() const;

    /**
    Convert between m/s and rpm.
    */
    static agx::Real convertAngularVelocityToRPM( const agx::Real angVel );
    static agx::Real convertRPMtoAngularVelocity( const agx::Real rpm );

    /**

    */
    virtual bool preUpdate(agx::Real timeStep) override;

    /**
     */
    virtual bool postUpdate(agx::Real timeStep) override;

    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::RotationalUnit);

  protected:
    virtual ~RotationalUnit();

  protected:
    RotationalDimensionRef m_rotationalDimension;
  };

  typedef agx::ref_ptr<RotationalUnit> RotationalUnitRef;
}

#endif // AGXMODEL_ROTATIONAL_UNIT_H
