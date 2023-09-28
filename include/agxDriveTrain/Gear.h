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

#include <agx/Hinge.h>

#include <agxPowerLine/PowerLine.h>
#include <agxPowerLine/RotationalUnit.h>
#include <agxPowerLine/PowerLineConstraints.h>


namespace agxDriveTrain
{
  /**
  A gear is a connector between two rotational units. It contains a gear ratio
  that defines the ratio of the number of gears on the output unit to the number
  of gears on the input unit. That is, a large gear ratio will make the output
  rotate slower and a small gear ratio will make the output faster
  */
  class AGXMODEL_EXPORT Gear : public agxPowerLine::RotationalConnector
  {
  public:
    /**
    Create a gear with gear ratio 1.
    */
    Gear();

    /**
    Create a gear with the given gear ratio.
    */
    explicit Gear(agx::Real ratio);

    /**
    Set the gear ratio. A higher gear ratio causes the output to rotate more slowly.
    */
    void setGearRatio(agx::Real ratio);

    /**
    \return The gear ratio.
    */
    agx::Real getGearRatio() const;


  public:
    DOXYGEN_START_INTERNAL_BLOCK()
    virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping(
        const agx::Real timeStep, agx::Real& compliance, agx::Real& damping) override;
    DOXYGEN_END_INTERNAL_BLOCK()

#ifndef SWIG
    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    using agxPowerLine::RotationalConnector::store;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxDriveTrain::Gear );
#endif

  protected:
    virtual ~Gear();
    agx::Real m_gearRatio;
  };

  typedef agx::ref_ptr<Gear> GearRef;



  /**
  Defines a viscous coupling between two rotational dimensions.
  The compliance is set manually. That will lead to a unknown efficiency.
  */
  class AGXMODEL_EXPORT SlipGear : public agxDriveTrain::Gear
  {
  public:
    SlipGear();

    /**
    Set the fixed compliance.
    */
    void setViscousCompliance( agx::Real viscousCompliance );

    /**
    \returns The fixed compliance
    */
    agx::Real getViscousCompliance() const;

  public:
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

    AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::SlipGear);
#endif

  protected:
    virtual ~SlipGear();
    agx::Real m_viscousCompliance;
  };
  typedef agx::ref_ptr<SlipGear> SlipGearRef;



  /**
  A Gear that uses a holonomic constraint instead of the default nonholonomic
  contstraint. This prevents accumulated slipping over time.
  */
  class AGXMODEL_EXPORT HolonomicGear : public agxDriveTrain::SlipGear
  {
  public:
    HolonomicGear();

    /**
    The angle, including winding, of the input unit.
    */
    agx::Real getInputAngle() const;

    /**
    The angle, including winding, of the output unit.
    */
    agx::Real getOutputAngle() const;

    /**
    Will set both input and output angle to zero, to remove all potential energy.
    */
    void rebind();

    /**
    Set the fixed damping.
    */
    void setViscousDamping( agx::Real viscousDamping );

    /**
    \return The fixed damping.
    */
    agx::Real getViscousDamping() const;

  public: // Methods called by AGX.

    virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping(
        const agx::Real timeStep, agx::Real& compliance, agx::Real& damping) override;

    virtual agx::Real calculateViolation() const override;

    virtual bool postUpdate(agx::Real timeStep) override;
    virtual bool removeNotification(agxUtil::ConstraintHolder* holder, agxSDK::Simulation* simulation) override;

#ifndef SWIG
    virtual agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() override;
#endif

#ifndef SWIG
    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::HolonomicGear);
#endif

  protected:
    virtual ~HolonomicGear();

    agx::Real m_inputAngle;
    agx::Real m_outputAngle;

    agx::Real m_viscousDamping;
  };

  typedef agx::ref_ptr<HolonomicGear> HolonomicGearRef;
}
