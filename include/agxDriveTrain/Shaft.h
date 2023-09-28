/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once



#include <agxPowerLine/PowerLine.h>
#include <agxPowerLine/RotationalUnit.h>
#include <agxPowerLine/PowerLineConstraints.h>


namespace agxDriveTrain
{
  AGX_DECLARE_POINTER_TYPES( Shaft );

  /**
  A Shaft is the most basic PowerLine Unit. It is used to carry rotational
  motion through the power line.
  */
  class AGXMODEL_EXPORT Shaft : public agxPowerLine::RotationalUnit
  {
  public:
    /**
    Create a shaft.
    */
    Shaft();

  public:

    /**
    Post update will update the accumulated angle
    */
    virtual bool postUpdate(agx::Real timeStep) override;

    /**
    \return The accumulated angle of the shaft in radians.
    */
    agx::Real getAccumulatedAngle() const;

    /**
    Set the accumulated shaft angle in radians.
    */
    void setAccumulatedAngle(agx::Real angle);

    /**
    Functions called by a static function. Do NOT use.
    */
    virtual bool preUpdate(agx::Real timeStep) override;

    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    using agxPowerLine::RotationalUnit::store;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    using agxPowerLine::RotationalUnit::restore;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::Shaft);

  protected:
    agx::Real m_accumulatedAngle;
    virtual ~Shaft();
  };
}
