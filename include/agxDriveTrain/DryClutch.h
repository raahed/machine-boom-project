#pragma once
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
  Single disk plate clutch with dry friction.  Input and output shafts are
  connected to plates with rough surfaces.  A spring loaded mechanism increases
  the normal force between the plates between a minimum and maximum, giving rise
  to the concept of torque capacity.  When the plates are at maximum distance
  from each other, no force is transmitted.  When they are at zero distance,
  maximum pressure is applied.

  Together with Coulomb friction, this keeps the plates rotating at nearly the
  same angular velocity untill the stick-slip transition is reached,
  alternately, when the torque required to move the output shaft at the same
  velocity as the input shaft reaches the torque capacity.  This happens when
  the load on the output shaft is too difficult to move.  If the two disks move
  relative to each other, this is called slipping so one can read the slip
  velocity at any time.

  The "opening fraction" here represents the value of the torque capacity in
  proportion to the maximum capacity, i.e., when the plates are fully closed,
  i.e., maximum normal force is achieved.  As for most clutches, this model
  includes a "locked" mode in which a clamp is activated to prevent any slip.
  This stays "ON" until the clutch is disengaged, or if the output shaft starts
  to move faster but in the same direction as the input shaft, as in the case of
  dynamic breaking.

  This model has two modes, namely, manual and automatic.  In the manual mode,
  the user is responsible to adjust the opening fraction, or set the lock
  mechanism directly. This can be done using a control algorithm and an event
  listener.  For the automatic mode, a time constant determines how long it
  takes to go from fully open to fully engaged.  The function "engage"
  determines whether we are ramping up or down.  Setting auto lock mode
  activates logic to activate or deactivate the lock according to simple logic.
  If the slip velocity is small in proportion of the input shaft velocity, the
  lock is activated.  If either "engage(false)" function is called or the torque
  between the places is negative -- dynamic braking -- then the lock is
  deactivated.


  The maximum available torque before slipping for a given opening fraction
  \f$f\f$ is `\f$fT_c\f$' where \f$T_c\f$ is the torque capacity defined by

  \f[
    T_c = 2\frac{2}{3}\mu\frac{r_2^3-r_1^3}{r_2^2-r_1^2}F
  \f]

  where \f$r_i\f$ are the outer and inner radii.  This defaults to a `reasonable'
  number corresponding to a passenger car, namely, around 40 Nm

  Automatic engagement makes the fraction increase linearly from the time the
  "engage(true)" function is called, or decreases linearly from the time the
  "engage(false)" is called.

  We assume that the pressure plate is a linear spring so that $F$ is
  directly proportional to it's position.

  */
  class AGXMODEL_EXPORT DryClutch : public agxPowerLine::RotationalConnector
  {
  public:
    DryClutch();

    /**
     Puts the clutch in assisted mode so it will fully engage or disengage in a
     time determined by setTimeConstant below.  Using engage(true) will turn
     manual mode to false as per the rule: respect the users's intention.
    \param engage - true to engage, false to disengage from current fraction
    */
    void setEngage(bool engage);

    /**
    \return - Returns whether the clutch is engaged or not.
    */
    bool isEngaged() const;

    /**
    \param fraction - Set the fraction of the clutch opening in range [0,1]
    */
    void setFraction(agx::Real fraction);

    /**
    \return - Returns the fraction of the clutch opening (fraction of maximum normal
    force betwen the plates).
    */
    agx::Real getFraction() const;

    /**
    Set the time constant for clutch engagement and disengagement. Engagement
    is linear and proceeds by a fraction of +/- step/tau per step.   The default
    time constant is in seconds.
    \param tau - the time constant. The unit of tau is the same as the unit of time step size defined by the user.
    */
    void setTimeConstant(agx::Real tau);

    /**
    When manual mode is false, user must control all aspects of the clutch
    engagement and disengagement.  Using the engage method will turn this flag to false.
    \param manual - Set or unset manual mode.
    */
    void setManualMode(bool manual);

    /**
    \return - Returns true in manual mode, false otherwise
    */
    bool isManualMode() const;

    /**
    \return - Returns the time constant of the clutch for engaging or disengaging.
    */
    agx::Real getTimeConstant() const;

    /**
    Lock or unlock the clutch. In locked mode, the two plates are fixed rigidly together.
    \param setLock - desired state
    */
    void setLock(bool setLock);

    /**
    Returns the lock status.
    */
    bool isLocked() const;

    /**
    This is the maximum torque that the clutch can transfer: it will slip if
     this is exceeded.  Set directly or by setting the physical parameters below.
     */
    void setTorqueCapacity(agx::Real capacity);

    /**
    \return - Returns the torque capacity of the clutch.
    */
    agx::Real getTorqueCapacity() const;

    /**
    Calculate the torque capacity based on the geometric and physical parameters of the clutch.
    \note This will overwrite torque capacity set via setTorqueCapacity() method. The users are referred to the reference (Shaver R. Manual transmission clutch systems. Warrendale, Pennsylvania: SAE International, 1997.) for more information of the clutch parameters.
    \param maxForce - the maximum normal force exerted onto the clutch discs
    \param mu - the Coulomb friction coefficient
    \param radii - a 2D vector of the inner and outer radii of the clutch discs
    */
    void calculateTorqueCapacity(agx::Real maxForce, agx::Real mu, agx::Real radii[2]);

    /**
    In auto lock mode, the clutch will lock when the slip is small enough in
    proportion to the velocity of the input shaft.
    \param autoLock - true or false to enable or disable autoLock, respectively
    */
    void setAutoLock(bool autoLock);

    /**
    \return - State of auto lock
    */
    bool isAutoLock() const;

    /**
    \return - Torque between the plates
    */
    agx::Real getTorque() const;

    /**
    \return - Relative velocity between the plates
    */
    agx::Real getSlip() const;

    /**
    \return - Angular velocity of input shaft
    */
    agx::Real getInputVelocity() const;

    /**
    \return - Angular velocity of output shaft
    */
    agx::Real getOutputVelocity() const;

    /**
    In autolock mode, the clutch will lock with the ratio of slip to input
    shaft velocity is less than this threshold
    */
    void  setMinRelativeSlip(agx::Real minRelativeSlip);

    /**
    \return - Minimum relative velocity for lock
    */
    agx::Real getMinRelativeSlip() const;

    DOXYGEN_START_INTERNAL_BLOCK()

#ifndef SWIG
    agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() override;
#endif

    virtual agx::RegularizationParameters::VariableType calculateComplianceAndDamping(const agx::Real timeStep, agx::Real& compliance, agx::Real& damping) override;

    virtual bool preUpdate(agx::Real timeStep) override;

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

    AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::DryClutch);
#endif

    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    virtual ~DryClutch();

    agx::Real m_torqueCapacity;
    bool      m_isEngaged;
    bool      m_isAutoLock;
    bool      m_isManual;
    agx::Real m_timeConstant;
    agx::Real m_minRelativeSlip;
    agx::Real m_fraction;
    bool      m_isLocked;

  };
  typedef agx::ref_ptr<DryClutch> DryClutchRef;

} // namespace agxDriveTrain
