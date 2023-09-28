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

#include <agxModel/export.h>
#include <agxPowerLine/PowerLine.h>
#include <agxDriveTrain/Gear.h>

namespace agxDriveTrain
{
  /**
  A torque converter is a connector, which transfers rotating power with the help of a fluid coupling.
  The implementation is based on two tables, one descibing the efficiency of the transfered torque and
  one describing how the geometry of it affects the counter torque present in the torque converter.
  The torque converter also has a lock up clutch, which makes it possible to lock the torque converter,
  and let input and output shafts rotate at the same speed.
  */
  class AGXMODEL_EXPORT TorqueConverter : public agxDriveTrain::Gear
  {
  public:
    /**
    Create the torque converter.
    The torque converter has default values for all parameters, also the efficiency and geometry tables,
    which will give the torque converter typical characteristics.
    */
    TorqueConverter();

    /**
    Get the calculated pump torque, which is the counter torque on the pump side in the model
    \return The pump torque
    */
    agx::Real getPumpTorque();

    /**
    Get the calculated torque output from the turbine side
    \return The turbine torque
    */
    agx::Real getTurbineTorque();

    /**
    Get the value of the oil density, which is the liquid that is used to transfer power
    \return The oil density, in kilograms per cubic meter
    */
    agx::Real getOilDensity() const;

    /**
    Set the oil density, which is the liquid that is used to transfer power
    \param oilDensity - oil density, in kilograms per cubic meter
    \return true of successful, false otherwise
    */
    bool setOilDensity(agx::Real oilDensity);

    /**
    Get the diamerter of the pump
    \return The pump diameter, in meters
    */
    agx::Real getPumpDiameter() const;

    /**
    Set the pump diameter.
    \param pumpDiameter - pump diameter, in meters
    \return true of successful, false otherwise
    */
    bool setPumpDiameter(agx::Real pumpDiameter);

    /**
    Set the values in the efficiency table, will clear the old values from the table. The table describes how
    efficiently the torque can be transfered for a given velocity ratio.
    \param velRatio_efficiency - vector pairs describing the table.
                                 x-values should correspond to velocity ratio and y-values to efficiency
    */
    void setEfficiencyTable(const agx::RealPairVector& velRatio_efficiency);

    /**
    Set the values in the geometry factor table, will clear the old values from the table. The table describes
    how the counter torque changes with the velocity ratio.
    \param velRatio_geometryFactor - vector pairs describing the table.
                                     x-values should correspond to velocity ratio and y-values to geometry factor
    */
    void setGeometryFactorTable(const agx::RealPairVector& velRatio_geometryFactor);

    /**
    Clear the efficiency table from all values.
    */
    void clearEfficiencyTable();

    /**
    Clear the geometry factor table from all values.
    */
    void clearGeometryFactorTable();

    /**
    Insert a value into the efficiency lookup table.
    \param velocityRatio - velocity ratio. Turbine velocity divided by pump velocoty.
    \param efficiency - efficiency of torque converter, for the given velocity ratio
    */
    void insertEfficiencyLookupValue(agx::Real velocityRatio, agx::Real efficiency);

    /**
    Insert a value into the geometry factor lookup table.
    \param velocityRatio - velocity ratio. Turbine velocity divided by pump velocoty.
    \param geometryFactor - effect on counter torque due to geometry, for the given velocity ratio
    */
    void insertGeometryFactorLookupValue(agx::Real velocityRatio, agx::Real geometryFactor);

    /**
    Get the efficiency lookup table.
    \return Table with the efficency values at given velocity ratios.
    */
    const agxPowerLine::LinearLookupTable* getEfficiencyTable() const;

    /**
    Get the geometry factor lookup table.
    \return Table with the geometry factor values at given velocity ratios.
    */
    const agxPowerLine::LinearLookupTable* getGeometryFactorTable() const;

    /**
    Control the lock in the torque converter. A locked torque converter will have the same pump and turbine
    velocities. The lock should be released in a gear change, when braking or when quickly increasing the
    velocity at the pump side.
    \param enable - when true, it will enable the lock. If it is false, the lock will be disabled.
    */
    void enableLockUp(bool enable);

    /**
    Get if the lock is enabled or disabled.
    \return true if lock is enabled, false if it is disabled
    */
    bool getEnableLockUp() const;

    /**
    Set how long time it should take to lock up the torque converter
    \param time - locking time in seconds
    */
    void setLockUpTime(agx::Real time);

    /**
    Get how long time, in seconds, it will take the torque converter to lock up
    \return Lock up time, in seconds
    */
    agx::Real getLockUpTime() const;

    DOXYGEN_START_INTERNAL_BLOCK()
    /** Internal method */
    virtual void initConstraint() override;
    /** Internal method */
    virtual bool preUpdate(agx::Real timeStep) override;
    /** Internal method */
    virtual bool store(agxStream::StorageStream& str) const override;
    /** Internal method */
    virtual bool restore(agxStream::StorageStream& str) override;
    AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::TorqueConverter);
    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    virtual ~TorqueConverter();
    agxPowerLine::LinearLookupTableRef m_efficiencyTable;
    agxPowerLine::LinearLookupTableRef m_geometryFactorTable;

    agx::Real m_oilDensity;
    agx::Real m_diameter;
    agx::Real m_m0;
    agx::Real m_pumpTorque;
    agx::Real m_turbineTorque;

    bool m_lockEnabled;
    agx::Real m_lockTime;
    agx::Real m_timeSinceLock;
    agx::Real m_ratioAtLock;

  private:
    void calculateInternalTorques(agx::Real inputVel, agx::Real outputVel);
    void controlLock(agx::Real timeStep);

  };
  AGX_DECLARE_POINTER_TYPES(TorqueConverter);
}
