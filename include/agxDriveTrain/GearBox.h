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

#pragma once

#include <agxPowerLine/PowerLine.h>
#include <agxPowerLine/PowerLineConstraints.h>
#include <agxPowerLine/Connector.h>

namespace agxDriveTrain
{
  /**
  The gear box can have an arbitrary number of gears.
  A negative gear ratio is considered to be a reverse gear.
  */
  class AGXMODEL_EXPORT GearBox : public agxPowerLine::RotationalConnector
  {
  public:
    /**
    Create an empty gear box. Unless gears are added using setGearRatios, this
    gear box will not transmit any torque and getGear will return
    agx::InvalidIndex.
    */
    GearBox();

    /**
    Create a gear box with the given gear ratios.
    \param gears - A vector with gear ratios
    */
    GearBox(const agx::RealVector& gears);

    /**
    Set the gear ratios.
    */
    void setGearRatios(const agx::RealVector& gearRatios);

    /**
    \return The number of gears.
    */
    size_t getNumGears( ) const;

    /**
    Get current load torque, i.e., the torque applied on the output. The torque
    on the input side will be one gear ratio factor lower.
    */
    agx::Real getTorque() const;

    /**
    Change to the next gear, if there is one.
    */
    bool gearUp();

    /**
    Change to the previous gear, if there is one.
    */
    bool gearDown();

    /**
    Change to the gear at the given index.
    */
    bool setGear(int gearIndex);

    /**
    \returns the index of the current gear.
    */
    int getGear() const;

    /**
    \returns The ratio of the gear at the given index.
    */
    agx::Real getGearRatio(int gearIndex) const;

    /**
    \return The gear ratio of the current gear, or 0 if no gear is selected.
    */
    agx::Real getCurrentGearRatio() const;

    virtual void initConstraint() override;

    virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping(
        const agx::Real timeStep, agx::Real& compliance, agx::Real& damping) override;

#ifndef SWIG
    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::GearBox);
#endif

  protected:
    virtual ~GearBox();

    /** \return The inverse of the current gear ratio. */
    agx::Real getCurrentRatio() const;

    agx::RealVector m_gearTable;
    size_t m_currentGear;
  };
  typedef agx::ref_ptr<GearBox> GearBoxRef;
}
