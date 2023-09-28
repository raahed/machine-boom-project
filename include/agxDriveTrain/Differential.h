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
#include <agxPowerLine/RotationalUnit.h>

#include <agxDriveTrain/Gear.h>

namespace agxUtil
{
  class ConstraintHolder;
}


namespace agxDriveTrain
{
  typedef std::pair<agx::observer_ptr<agxPowerLine::Unit>, agxPowerLine::Side> UnitObserverSidePair;
  typedef agx::Vector<UnitObserverSidePair> UnitObserverSidePairVector;

  /**
  Connector for locking of the differential.
  Number of outputs from differential minus one of these are needed to lock the differential.
  These will be created automatically by the agxModel::Differential, and the constraints they represent will be enabled when the lock is set active.
  */
  class AGXMODEL_EXPORT DifferentialLockConnector : public agxPowerLine::RotationalConnector
  {
  public:
    /**
    Lock connector between the outputs for the differential.
    */
    DifferentialLockConnector( );

    virtual void initConstraint( ) override;

#ifndef SWIG
    /**
    Stores internal data into stream.
    */
    virtual bool store( agxStream::StorageStream& str ) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore( agxStream::StorageStream& str ) override;


    AGXSTREAM_DECLARE_SERIALIZABLE( agxDriveTrain::DifferentialLockConnector );
#endif

  protected:
    virtual ~DifferentialLockConnector( );


  };

  typedef agx::ref_ptr<DifferentialLockConnector> DifferentialLockConnectorRef;



  /**
  A differential for distributing the torque evenly between an arbitrary number of output shafts.
  */
  class AGXMODEL_EXPORT Differential : public agxDriveTrain::Gear
  {
  public:
    /**
    Create a differential.
    */
    Differential();

    virtual bool connect(agxPowerLine::Side mySide, agxPowerLine::Side unitSide, agxPowerLine::Unit* unit) override;
    virtual bool disconnect(agxPowerLine::Unit* unit) override;

    using Gear::connect;
    using Gear::disconnect;

    /**
    By setting the limited slip torque, the lock will no longer hold infinite torque.
    It will hold at most the limited slip torque.
    This call overrides the lock having infinite force range.
    By enabling the lock again, this call will be overridden, giving the lock infinite possible torque.
    */
    void setLimitedSlipTorque( agx::Real limitedSlipTorque );

    /**
    \returns true if the differential lock is enabled.
    */
    bool getLockEnabled( ) const;

    /**
    \returns true if the differential is using the limited slip torque mode.
    */
    bool getLimitedSlipTorqueEnabled( ) const;

    /**
    enable/disable the differential lock
    */
    void setLock(bool enable);

    virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping(
        const agx::Real timeStep, agx::Real& compliance, agx::Real& damping) override;


#if !defined(SWIG) && !defined(SWIG_CPPWRAPPER_BUILD)

    virtual void getConnectorsRecursive(agxPowerLine::ConnectorPtrVector& connectors) override;

    /**
    Function called by a static function. Do NOT use.
    */
    virtual bool removeNotification(agxUtil::ConstraintHolder* holder, agxSDK::Simulation* simulation) override;

    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::Differential);
#endif

  protected:
    virtual ~Differential();
    void removeDiffLockConnector(size_t connectorIndex);

    agx::Vector<agxDriveTrain::DifferentialLockConnectorRef> m_diffLockConnectors;
    UnitObserverSidePairVector m_children;

  private:
    agx::Bool m_locked;
    agx::Real m_limitedSlipTorque;
  };
  typedef agx::ref_ptr<Differential> DifferentialRef;

}
