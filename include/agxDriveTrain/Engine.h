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

#include <agxDriveTrain/Shaft.h>
#include <agxDriveTrain/VelocityConstraint.h>
#include <agxPowerLine/PowerLineConstraints.h>

namespace agxDriveTrain
{
  AGX_DECLARE_POINTER_TYPES(EnginePowerGenerator);

  /**
  Power generator that adds torque to the engine according to a lookup table.
  */
  class AGXMODEL_EXPORT EnginePowerGenerator : public agxPowerLine::TorqueGenerator
  {
  public:
    /**
    Create an engine power generator.
    \param dimension - rotational dimension, to which body a torque will be added each timestep
    \param minTorque
    */
    EnginePowerGenerator(agxPowerLine::RotationalDimension* dimension, agx::Real minTorque);

    /**
    Return the current torque to add (time integral of power)
    */
    virtual agx::Real getCurrentPowerTimeIntegral() const override;

    /**
    The min torque is the torque an engine produces at idle.
    */
    void setMinTorque(agx::Real minTorque);

    agx::Real getMinTorque() const;

    /**
    Finds the current RPM (and not angular velocity)
    */
    virtual agx::Real variableLookupFunction( ) const override;

    /**
    Scales the resulting torque with the throttle input to the engine.
    */
    virtual agx::Real resultScalerFunction() const override;

    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::EnginePowerGenerator);

  protected:
    EnginePowerGenerator();
    virtual ~EnginePowerGenerator();
    agx::Real m_minTorque;
  };



  AGX_DECLARE_POINTER_TYPES( Engine );

  /**
  An engine that is driven by an EnginePowerGenerator.
  */
  class AGXMODEL_EXPORT Engine : public agxDriveTrain::Shaft
  {
  public:
    /**
    Create an engine with an EnginePowerGenerator.
    */
    Engine();

    /**
    \return The currently active power generator.
    */
    agxPowerLine::PowerGenerator* getPowerGenerator() const;

    /**
    Set the power generator to use for this engine.
    */
    void setPowerGenerator(agxPowerLine::PowerGenerator* generator);

    /**
    Get revolutions per minute.
    */
    agx::Real getRPM() const;

    /**
    Set the throttle input.
    \param throttle - 0 means no gas, 1 means full gas.
    */
    void setThrottle(agx::Real throttle);

    /**
    \return The current throttle.
    */
    virtual agx::Real getThrottle() const;

    /**
    Set the wanted RPM when there is no throttle.
    */
    void setIdleRPM(agx::Real idleRPM);

    /**
    \return The wanted RPM when there is no throttle.
    */
    agx::Real getIdleRPM() const;

  public:
    /**
    Function called by static functions. Do NOT use them.
    */
    virtual bool preUpdate(agx::Real timeStep) override;

    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::Engine);

  protected:
    virtual ~Engine();

  protected:
    agxPowerLine::PowerGeneratorRef m_generator;
    agx::Real m_throttle;
    agx::Real m_idleRPM;
  };



  class PidControlledEngine;

  /**
  Inherit from the ThrottleCalculator and implement the calculateThrottle()
  function to control the torque of a PidControlledEngine.

  The calculator has to be set to the engine:
  PidControlledEngine::setThrottleCalculator(ThrottleCalculator* throttleCalculator);
  */
  class AGXMODEL_EXPORT ThrottleCalculator : public agx::Referenced, public agxStream::Serializable
  {
  public:
    /**
    Create a ThrottleCalculator given a PidControlledEngine.
    */
    explicit ThrottleCalculator( PidControlledEngine* engine );

    /**
    Classes inheriting from ThrottleCalculator has to implement the
    calculateThrottle function.
    */
    virtual agx::Real calculateThrottle() = 0;

    /**
    Set an engine to the controller.
    */
    void setEngine( PidControlledEngine* engine );

    /**
    \return The engine currently controlled by this ThrottleCalculator.
    */
    PidControlledEngine* getEngine();
    const PidControlledEngine* getEngine() const;

    /**
    Stores internal data into stream.
    */
    virtual bool store( agxStream::StorageStream& str ) const;

    /**
    Restores internal data from stream.
    */
    virtual bool restore( agxStream::StorageStream& str );

    void store( agxStream::OutputArchive& out ) const override;

    void restore( agxStream::InputArchive& in ) override;

    AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxDriveTrain::ThrottleCalculator );

    ThrottleCalculator( );

  protected:
    virtual ~ThrottleCalculator( );
    PidControlledEngine* m_engine;
  };

  typedef agx::ref_ptr<ThrottleCalculator> ThrottleCalculatorRef;



  /**
  ThrottleCalculator that strives to keep the engine at some given RPM.

  The pGain, dGain and iGain parameters specifies how the controller should
  behave when not at the target RPM. These are expressed in terms of torque and
  RPM.

  The p in pGain stands for proportional. The pGain produces a contribution to
  the throttle that is proportional to the current error in the RPM. The
  further from the target RPM the more throttle. For every RPM that is missing,
  the controller will add one pGain Nm worth of throttle.

  The d in dGain stands for derivative. It uses the differences between the
  current RPM and the RPM in the previous time step to calculate it's
  contribution to the throttle.

  The i in iGain stands for integrated. It is similar to the pGain parameter,
  but considers errors in previous time steps as well. The error in the RPM is
  accumulated over time and the iGain parameter determines how much this
  integrated value should contribute to the throttle. To prevent this number
  for becoming unreasonable large the iRange parameter is used to clamp the
  integrated value to the range [-iRange, iRange].
  */
  class AGXMODEL_EXPORT RpmController : public agxDriveTrain::ThrottleCalculator
  {
    public:
      /**
      Create a RpmController that operates on the given engine. Call
      PidControlledEngine::setThrottleCalculator to activate the controller.

      \param engine - The engine to control the RPM of.
      \param targetRpm - The RPM we wish to maintain.
      \param pGain - Gain of the proportional term.
      \param dGain - Gain of the derivative term.
      \param iGain - Gain of the integrated term.
      \param iRange - Limits on the RPM error integration.
      */
      RpmController(
          PidControlledEngine* engine, agx::Real targetRpm,
          agx::Real pGain, agx::Real dGain, agx::Real iGain,
          agx::Real iRange);

      agx::Real getTargetRpm() const;
      void setTargetRpm(agx::Real targetRpm);

      agx::Real getLastThrottle() const;


    // Methods called other classes in the DriveTrain framework.
    public:
      virtual agx::Real calculateThrottle() override;

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::RpmController);

    protected:
      RpmController();
      virtual ~RpmController();

    private:
      agx::Real m_targetRpm;

      agx::Real m_pGain;
      agx::Real m_dGain;
      agx::Real m_iGain;
      agx::Real m_iRange;

      agx::Real m_iState;
      agx::Real m_lastRpmDiff;
      agx::Real m_lastThrottle; // \todo Should this be in ThrottleCalculator? If so, how should it be updated?
  };


  typedef agx::ref_ptr<RpmController> RpmControllerRef;



  /**
  Implementation of a combustion engine.
  Characteristics of a real engine are applied through one lookup table.

  The lookup table tells how much torque the engine is able to apply given its rpm.

  To set the rpm/torque table use:
  void setRPMTorqueTable(agx::Vector<std::pair<agx::Real, agx::Real> > const& rpm_torques);

  If you only know the power it produces you may set the rpm/power table instead:

  void setRPMPowerTable(agx::Vector<std::pair<agx::Real, agx::Real> > const& rpm_power);
  */
  class AGXMODEL_EXPORT PidControlledEngine : public agxDriveTrain::Engine
  {
  public:
    /**
    Create an engine with an empty torque table and no ThrottleCalculator.
    */
    PidControlledEngine( );

    /**
    If \p ignition is true - start the engine. Else turn it off.
    */
    void ignition( bool ignition );

    /**
    Turn off the ignition key
    */
    void shutdown( bool shutdown );

    /**
    \returns true if the engine is running.
    */
    bool isRunning( ) const;

    /**
    \returns current rpm of engine.
    */
    agx::Real getRPM( ) const;

    /**
    \returns the current load of engine
    */
    agx::Real getLoadTorque( ) const;

    /**
    Give the engine output torque at different engine RPMs.
    */
    void setRPMTorqueTable( const agx::RealPairVector& rpm_torques );

    /**
    Give the engine output power at different engine RPMs.
    */
    void setRPMPowerTable( const agx::RealPairVector& rpm_power );

    /**
    \returns the output torque at a given rpm.
    */
    agx::Real getTorqueAtRPM( agx::Real rpm ) const;

    /**
    Use the extra load to be able to set an extra torque loss to simulate
    friction, electrical systems load, tuning.
    */
    void setExtraLoad( agx::Real torque );

    agx::Real getExtraLoad( ) const;

    /**
    Set a throttle controller that will be asked for a throttle every time step
    while the engine is turned on.
    */
    void setThrottleCalculator( ThrottleCalculator* controller );

    /**
    \returns The current ThrottleCalculator, if any, or nullptr.
    */
    const ThrottleCalculator* getThrottleCalculator() const;
    ThrottleCalculator* getThrottleCalculator();

    /**
    \returns The current possible max torque for the current RPM.
    */
    agx::Real getTorque( ) const;

  public:
    /**
    Function that has to be public since it is called by static functions. Do NOT use it.
    */
    virtual bool preUpdate( agx::Real timeStamp ) override;

    /**
    Stores internal data into stream.
    */
    virtual bool store( agxStream::StorageStream& str ) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore( agxStream::StorageStream& str ) override;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxDriveTrain::PidControlledEngine );

  protected:
    virtual ~PidControlledEngine( );
    bool m_isOn;
    agx::Real m_extraLoad;
    ThrottleCalculatorRef m_pidController;
  };

  typedef agx::ref_ptr< PidControlledEngine > PidControlledEngineRef;


  /**
  An engine that tries to hold a fixed velocity.
  */
  class AGXMODEL_EXPORT FixedVelocityEngine : public agxDriveTrain::Shaft
  {
  public:
    /**
    Create an engine with target velocity set to zero.
    */
    FixedVelocityEngine();

    /**
    Get the torque applied on the engine during the last time step. This is
    different from getDeliveredTorque in that getCurrentTorque include torque
    consumed by the engine itself, while getDeliveredTorque only include torque
    delivered out of the engine.

    \return The torque applied on the engine during the last time step.
    */
    agx::Real getCurrentTorque();

    /**
    \returns the current target velocity
    */
    agx::Real getTargetVelocity();

    /**
    \return The torque range available to the Engine
    */
    agx::RangeReal getTorqueRange();

    /**
    Set the target velocity of the engine.

    \param velocity - the desired velocity
    */
    void setTargetVelocity(agx::Real velocity);

    /**
    Set the RPM that the engine should try to hold.
    This sets the target velocity.
    */
    void setTargetRpm(agx::Real rpm);

    /**
    Set the torque range to limit the available torque of the engine.
    The available torque is infinite by default.
    If the default value is overriden it might not be able to
    reach the target velocity set.

    \param range - the available torque range
    */
    void setTorqueRange(agx::RangeReal range);

    /**
    * \return The VelocityConstraint of the motor.
    */
    const agxDriveTrain::VelocityConstraint* getConstraint() const;

    DOXYGEN_START_INTERNAL_BLOCK()
    /**
    */
    virtual bool addNotification(agxSDK::Simulation* simulation) override;

    /**
    */
    virtual bool removeNotification(agxUtil::ConstraintHolder* constraintHolder, agxSDK::Simulation* simulation) override;

    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    /**
    Store velocity constraint data to stream.
    */
    virtual bool postStore(agxStream::StorageStream& str) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    /**
    Restore velocity constraint data from stream.
    */
    virtual bool postRestore(agxStream::StorageStream& str) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::FixedVelocityEngine);
    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    virtual ~FixedVelocityEngine();

    agxDriveTrain::VelocityConstraintRef m_velocityConstraint;
  };

  typedef agx::ref_ptr< FixedVelocityEngine > FixedVelocityEngineRef;
}
