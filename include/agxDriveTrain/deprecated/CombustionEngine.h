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

#include <agxModel/export.h>

#include <agxPowerLine/RotationalUnit.h>


namespace agxDriveTrain
{
  namespace deprecated {

  class AGXMODEL_EXPORT CombustionEngine : public agxPowerLine::RotationalUnit
  {
    public:
      /**
      Create an combustion engine.
      \param displacementVolume - the total displacement volume, in m^3, of the engine, which is the volume the pistons 
                                  displace in the cylinder.
                                  Throttle bore, inlet volume are derived from the displacement volume as a default value.
                                  If you want more specific values for those parameters, you have to set them specifically.
      */
      CombustionEngine(agx::Real displacementVolume);

      /**
      Get revolutions per minute.
      */
      agx::Real getRPM() const;

      /**
      Get the current pressure in the inlet manifold, i.e.the pressure right before
      the cylinders.
      \return The inlet manifold pressure.
      */
      agx::Real getInletPressure() const;

      /**
      Set the total Displacement volume of engine cylinders in m^3
      */
      void setDisplacementVolume(agx::Real displacementVolume);

      /**
      \return Displacement volume of engine cylinders
      */
      agx::Real getDisplacementVolume() const;

      /**
      Set the percentage that the throttle is open.
      \param throttle - 0 means no gas (idle), 1 means full gas.
      \return true if successful, false otherwise
      */
      bool setThrottle(agx::Real throttle);

      /**
      \return The current throttle.
      */
      agx::Real getThrottle() const;

      /**
      Set the throttle plate angle directly, instead of percentage of throttle opening.
      \param throttleAngle - angle in radians. Cannot be smaller than idle throttle and must be smaller than pi/2.
      \return true if successful, false otherwise
      */
      bool setThrottleAngle(agx::Real throttleAngle);

      /**
      \return The angle of the throttle plate
      */
      agx::Real getThrottleAngle() const;

      /**
      Set the idle throttle angle for the engine, i.e. the smallest angle the throttle can have when the engine is
      running.
      \param throttleAngle - angle for idle throttle, must be larger than 0 and smaller than maximum throttle angle
      \return true if successful, false otherwise
      */
      bool setIdleThrottleAngle(agx::Real throttleAngle);

      /**
      \return Idle throttle angle, the smallest angle the throttle plate can have when the engine is running.
      */
      agx::Real getIdleThrottleAngle() const;

      /**
      Set the largest angle the throttle plate is allowed to have, i.e. the angle is has when the engine has full gas.
      \param throttleAngle - maximum throttle angle, must be smaller than pi/2 rad and larger than idle throttle angle.
      \return true if successful, false otherwise
      */
      bool setMaxThrottleAngle(agx::Real throttleAngle);

      /**
      \return The maximum throttle angle the throttle plate is allowed to have.
      */
      agx::Real getMaxThrottleAngle() const;

      /**
      Set the smallest and largest angle the throttle plate is allowed to have, i.e. the angle at idle and the angle
      at full gas.
      \param min - angle for idle throttle, must be larger than 0 and smaller than maximum angle
      \param max - maximum throttle angle, must be smaller than pi/2 rad and larger than minimum angle
      \return true if successful, false otherwise
      */
      bool setThrottleAngleRange(agx::Real min, agx::Real max);

      /**
      Set the diameter of the throttle plate, which is assumed to be circular.
      \param throttleBore - throttle bore given in meters, which must be larger than 0.
      \return true of successful, false otherwise
      */
      bool setThrottleBore(agx::Real throttleBore);

      /**
      \return Throttle bore, i.e. throttle plate diameter in meters.
      */
      agx::Real getThrottleBore() const;

      /**
      Set how many revolutions per cycle the engine cylinders have.
      \param nrRevolutionsPerCycle - 1 for two-stroke engine, 2 for four-stroke engine
      \return true if successful, false otherwise
      */
      bool setNrRevolutionsPerCycle(agx::Real nrRevolutionsPerCycle);

      /**
      \return The number of revolutions per engine cylinder cycle
      */
      agx::Real getNrRevolutionsPerCycle() const;

      /**
      \return The current torque the engine can supply
      */
      agx::Real getOutputTorque();

      /**
      Set the volumetric efficiency. It describes the ratio of air volume drawn into the cylinders to the volume the
      cylinder sweeps, the displacement volume.
      \param volumetricEfficiency - the volumetric efficiency. Should be a value between 0 and 1.
      */
      bool setVolumetricEfficiency(agx::Real volumetricEfficiency);

      /**
      \return The volumetric efficiency of the engine.
      */
      agx::Real getVolumetricEfficiency() const;

      /**
      Set the discharge coefficient of the flow past the throttle. It describes the ratio between the mass flow rate at
      the discharge end to that of the ideal mass flow at the discharge.
      \param dischargeCoefficient - value between 0 and 1 that. 1 means that the flow is ideal.
      \return true of successful, false otherwise
      */
      bool setDischargeCoefficient(agx::Real dischargeCoefficient);

      /**
      \return The discharge coefficient of the throttle flow.
      */
      agx::Real getDischargeCoefficient() const;

      /**
      Set the inlet manifold volume of the engine.
      \param inletVolume - inlet manifold volume in m^2, which must be larger than 0.
      \return true of successful, false otherwise
      */
      bool setInletVolume(agx::Real inletVolume);

      /**
      \return Volume of the inlet manifold of the engine in m^2
      */
      agx::Real getInletVolume() const;

      /**
      Must be called to start the engine.
      \param enable - true to start the engine, false turns it off.
      */
      void setEnable(bool enable);

      /**
      \return true if the engine is running, false otherwise.
      */
      bool getEnable() const;

      /**
      Set the inertia for the engine
      */
      virtual void setInertia(agx::Real inertia) override;

      /**
      Get the engine inertia
      */
      virtual agx::Real getInertia() const override;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      /** Internal method */
      virtual bool preUpdate(agx::Real timeStep) override;

      /** Internal method */
      virtual bool store(agxStream::StorageStream& str) const override;

      /** Internal method */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::deprecated::CombustionEngine);

      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      CombustionEngine() = default;

      virtual ~CombustionEngine();

    private:
      /**
      Computes the air mass flow past the throttle into the inlet manifold
      \return Air mass flow past throttle.
      */
      agx::Real computeThrottleMassFlow();

      /**
      Computes the derivative of the air mass flow past the throttle into the inlet manifold
      \return Derivative of air mass flow past throttle.
      */
      agx::Real computeThrottleMassFlowDer();

      /**
      Function for integrating the pressure derivative to calculate the current inlet manifold pressure.
      Also calulates the update to the engine torque.
      \param timeStep - simulation time step
      \param angularVel - current engine angular velocity
      */
      void stepPressure(agx::Real timeStep, agx::Real angularVel);

      /**
      Function to be called just when starting up the engine. It integrates pressure and velocity of the engine to an
      initial value, i.e. it starts the engine.
      \param timeStep - simulation time step
      */
      void startEngine(agx::Real timeStep);


    private:
      agx::Real m_inletPressure;
      agx::Real m_torque;

      agx::Real m_idleThrottle;
      agx::Real m_maxThrottleAngle;
      agx::Real m_throttleAngle;
      agx::Real m_throttleBore;
      agx::Real m_displacementVolume;
      agx::Real m_loadConstant;
      agx::Real m_nrRevolutionsPerCycle;

      agx::Real m_inletVolume;
      agx::Real m_inletTemp;
      agx::Real m_ambientPressure;
      agx::Real m_volumetricEfficiency;
      agx::Real m_airGasConstant;
      agx::Real m_dischargeCoefficient;

      bool m_enable;
      bool m_started;

      agx::Real m_c1;
      agx::Real m_c2;
      agx::Real m_startingTime;
      agx::Real m_restInertia;
  };

  AGX_DECLARE_POINTER_TYPES(CombustionEngine);
}
  }
