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
#include <agxPowerLine/RotationalDimension.h>



namespace agxDriveTrain
{

  /**
  Defines a viscous coupling between two rotational dimensions.
  The calculated compliance will differ depending on the input/output velocities
  and the input load.
  */
  class AGXMODEL_EXPORT Clutch : public agxPowerLine::RotationalConnector
  {
  public:
    Clutch();

    explicit Clutch(agx::Real efficiency);

    void setEfficiency(agx::Real efficiency);
    agx::Real getEfficiency() const;

    void setTorqueTransferConstant(agx::Real constant);
    agx::Real getTorqueTransferConstant() const;

    void setForcedCompliance(agx::Real compliance);
    agx::Real getForcedCompliance() const;



    /**
    Calculates a compliance for the clutch constraint given the wanted efficiency and a limitTorque
    */
    agx::Real calculateCompliance(agx::Real limitTorque) const;

    /**
    Calculates the limit torque given the given efficiency (eff) and torque transfer constant ttc(.
    limitTorque = ttc / (1 - eff) - ttc
    */
    agx::Real calculateLimitTorque() const;

    /**
    \returns true if user has chosen to use forced compliance
    */
    bool getUseForcedCompliance() const;




    /**
    Calculates compliance according to a mathematical torque converter.
    */
    virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping(const agx::Real timeStep, agx::Real& compliance, agx::Real& damping) override;


    virtual bool addNotification(agxSDK::Simulation* simulation) override;

#ifndef SWIG
    virtual bool postStore(agxStream::StorageStream& str) const override;
    virtual bool postRestore(agxStream::StorageStream& str) override;

    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::Clutch);
#endif

  protected:
    virtual ~Clutch();

    agx::Real m_efficiency;
    agx::Real m_torqueTransferConstant;
    agx::Real m_forcedCompliance;
  };
  typedef agx::ref_ptr<Clutch> ClutchRef;
}

