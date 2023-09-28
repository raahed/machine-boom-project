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
#include <agxDriveTrain/Shaft.h>
#include <agxDriveTrain/CombustionEngineParameters.h>


namespace agxDriveTrain
{
  /**
  Parameters controlling the mean effective friction torque. This corresponds to
  page 722, section 13.5.1 in "Internal combustion engine fundamentals" by John B. Heywood, 1988.
  */
  struct AGXMODEL_EXPORT FrictionTorqueParameters
  {
    agx::Real cfr0;
    agx::Real cfr1;
    agx::Real cfr2;
    FrictionTorqueParameters();
  };


  class AGXMODEL_EXPORT CombustionEngine : public agxDriveTrain::Shaft
  {
    public:
      /**
      Create a combustion engine.
      The model of the combustion engine consists of two subsystems: namely, 
      the intake manifold dynamics and the rotational dynamics.

      The model of the intake manifold dynamics implemented here is described in example 7.2,
      in "Modeling and control of engines and drivelines" by Lars Eriksson and Lars Nielsen,
      2014. Regarding the model of the rotational dynamics, it is defined in the section 10.1.5 
      of the same book.

      Construct a combustion engine.
      \param params - The typical engine parameters including the displacement volume, 
      rated torque, rated power, rated engine RPM, etc.
      */
      CombustionEngine(CombustionEngineParameters params);

      /**
      Return directly a reference to allow the constructs like
      engine.getFrictionTorqueParameters().Cfr1 = 3;
      */
      FrictionTorqueParameters& getFrictionTorqueParameters();

      /**
      Return directly a reference to allow the constructs like
      engine.getCombustionEngineParameters().displacementVolume;
      engine.getCombustionEngineParameters().maxTorque;
      Warning: there is no setCombustionEngineParameters() method, since that requires reinitialization.
      It is best to delete and create new instance.
      */
      CombustionEngineParameters& getCombustionEngineParameters();

      /**
      Get the current pressure in the inlet manifold.
      \return The inlet manifold pressure.
      */
      agx::Real getInletPressure() const;

      /**
      Set the fraction or the percentage of the throttle openning.
      \param throttle - 0 is idle/fully closed, 1 means fully open.
      \return True if successful, false otherwise.
      */
      bool setThrottle(agx::Real throttle);

      /**
      \return The current throttle fraction between 0 (idling) and 1 (full openning).
      */
      agx::Real getThrottle() const;

      /**
      Set the opening angle of the throttle plate directly, as opposed to the fraction/percentage of the throttle opening.
      \param throttleAngle - Angle in radians. It should be larger than the idle throttle angle and smaller than pi/2.
      \return True if successful, false otherwise
      */
      bool setThrottleAngle(agx::Real throttleAngle);

      /**
      \return The openning angle of the throttle plate.
      */
      agx::Real getThrottleAngle() const;

      /**
      Set the idle throttle angle, i.e. the smallest angle that the throttle can have when the engine is
      running at idle condition.
      \param throttleAngle - Angle of idle throttle in radians, must be larger than 0 and smaller than pi/2.
      \return True if successful, false otherwise.
      */
      bool setIdleThrottleAngle(agx::Real throttleAngle);

      /**
      \return Idle throttle angle, the smallest angle the throttle plate can have when the engine is running at idle.
      */
      agx::Real getIdleThrottleAngle() const;

      /**
      Set the maximum angle that the throttle plate is allowed to have, i.e. the angle when the engine has full gas.
      \param throttleAngle - maximum throttle angle in radians, must be smaller than pi/2 rad and larger than idle throttle angle.
      \return True if successful, false otherwise.
      */
      bool setMaxThrottleAngle(agx::Real throttleAngle);

      /**
      \return The maximum throttle angle that the throttle plate is allowed to have.
      */
      agx::Real getMaxThrottleAngle() const;

      /**
      Set the minimum and maximum angle that the throttle plate is allowed to have, i.e. the angle at idle and the angle
      at full gas.
      \param min - Angle for idle throttle in radians, must be larger than 0 and smaller than pi/2.
      \param max - Maximum throttle angle in radians, must be smaller than pi/2 rad and larger than minimum angle.
      \return True if successful, false otherwise.
      */
      bool setThrottleAngleRange(agx::Real min, agx::Real max);

      /**
      Set the ratio between throttle pin diameter and throttle bore diameter.
      \param throttlePinBoreRatio - the ratio between throttle pin and throttle bore diameters, which should be 
      varying within the range of [1/12, 1/6].
      \return True of successful, false otherwise.
      */
      bool setThrottlePinBoreRatio(agx::Real throttlePinBoreRatio);

      /**
      \return The ratio between throttle pin and throttle bore diameters.
      */
      agx::Real getThrottlePinBoreRatio() const;

      /**
      Set the number of crankshaft revolutions during one engine operating cycle.
      \param nrRevolutionsPerCycle - 1 for two-stroke engine, 2 for four-stroke engine.
      \return True if successful, false otherwise
      */
      bool setNrRevolutionsPerCycle(agx::Real nrRevolutionsPerCycle);

      /**
      \return The number of crankshaft revolutions per engine operating cycle
      */
      agx::Real getNrRevolutionsPerCycle() const;

      /**
      Set the maximum volumetric efficiency of the engine. The volumetric efficiency refers to the ratio of air volume
      drawn into the cylinders to the volume the cylinder sweeps.
      \param maxVolumetricEfficiency - The maximum value of the volumetric efficiency. It is usually within the range of 0.8 to 0.95.
      */
      bool setMaxVolumetricEfficiency(agx::Real maxVolumetricEfficiency);

      /**
      \return The maximum volumetric efficiency of the engine.
      */
      agx::Real getMaxVolumetricEfficiency() const;

      /**
      \return The volumetric efficiency at the specified engine speed and inlet pressure.
      \param N - The engine speed in radian per second.
      \param p - The inlet pressure in pa.
      */
      agx::Real getVolumetricEfficiency(agx::Real N, agx::Real p) const;

      /**
      Compute the engine air mass flow rate at current engine speed N and inlet pressure p.
      \param N - The engine speed in radian per second.
      \param p - The inlet pressure in pa.
      */
      agx::Real computeCylinderAirMassFlowRate(agx::Real N, agx::Real p);

      /**
      \return The thermal efficiency of the engine.
      */
      agx::Real getThermalEfficiency() const;

      /**
      Set the stoichiometric air-fuel ratio. Ideal ratio of air to fuel needed for complete combustion, e.g. gasoline 14.6.
      \param ratio - The stoichiometric air-fuel ratio.
      */
      bool setAirFuelRatio(agx::Real ratio);

      /**
      \return The stoichiometric air-fuel ratio.
      */
      agx::Real getAirFuelRatio() const;

      /**
      Set the heat value of the fuel, Unit: J/kg.
      \param heatValue - the energy that the fuel can deliver per kg of fuel mass.
      */
      bool setHeatValue(agx::Real heatValue);

      /**
      \return The heat value of the fuel air mixture.
      */
      agx::Real getHeatValue() const;

      /**
      Compute the engine indicated torque.
      \param N - The engine speed in radian per second.
      \param p - The inlet pressure in pa.
      */
      agx::Real computeIndicatedTorque(agx::Real N, agx::Real p);

      /**
      Compute the engine friction torque losses.
      \param N - The engine speed in radian per second.
      */
      agx::Real computeFrictionTorque(agx::Real N);

      /**
      Compute the pumping torque losses.
      \param p - The inlet manifold pressure in pa.
      */
      agx::Real computePumpTorque(agx::Real p);

      /**
      \return The current indicated torque that the engine has.
      */
      agx::Real getIndicatedTorque() const;

      /**
      \return The torque of friction loss that the engine components consumes.
      */
      agx::Real getFrictionTorque() const;
      
      /**
      \return The torque of pumping loss that the engine breaths air in and out.
      */
      agx::Real getPumpTorque() const;

      /**
      \return The current torque that the engine can supply
      */
      agx::Real getOutputTorque() const;

      /**
      Set the inlet manifold volume of the engine.
      \param inletVolume - The inlet manifold volume in m^3, which must be larger than 0.
      \return True of successful, false otherwise
      */
      bool setInletVolume(agx::Real inletVolume);

      /**
      \return The volume of the inlet manifold of the engine in m^3.
      */
      agx::Real getInletVolume() const;

      /**
      Must be called to start the engine.
      \param enable - True to start the engine, false turns it off.
      */
      void setEnable(bool enable);

      /**
      \return True if the engine is running, false otherwise.
      */
      bool getEnable() const;

      /**
      The inlet manifold is where the fuel and air to be mixed. The lower temperature yields more power.
      In the current model, this is a constant.
      \param inletTemperature - The temperature of the inlet manifold in K.
      */
      void setInletTemperature(agx::Real inletTemperature);

      /**
      \return The temperature of the inlet manifold.
      */
      agx::Real getInletTemperature() const;

      /**
      \return The cylinder air mass flow rate.
      */
      agx::Real getCylinderAirMassFlow();

      /**
      \return The throttle air mass flow rate.
      */
      agx::Real getThrottleAirMassFlow();

      DOXYGEN_START_INTERNAL_BLOCK()
      /** Internal method:
      Compute the maximum of the throttle openning area. 
      */
      void computeMaxThrottleArea();

      /** Internal method: 
      Compute the volumetric efficiency coefficients of the engine. 
      */
      void computeVolumetricEfficiencyCoefficients();
      
      /** Internal method: 
      Compute the gain of the compressible flow. Here, the gain refers to 
      a non-dimensional scaling factor to increase of decrease the air mass flow.
      \param r - the ratio between the inlet pressure and ambient pressure.
      */
      agx::Real computeFlowGain(agx::Real r);

      /** Internal method: 
      Compute the derivative of the gain of the compressible flow.
      \param r - the ratio between the inlet pressure and ambient pressure.
      */
      agx::Real computeFlowGainDerivative(agx::Real r);

      /** Internal method: 
      Compute the gain of the effective area of the throttle openning. The gain refers to a 
      non-dimensional scaling factor to increase or decrease the throttle area.
      \param alpha - the throttle openning angle.
      */
      agx::Real computeThrottleAreaGain(agx::Real alpha);

     public:
      /** Internal method */
      virtual bool preUpdate(agx::Real timeStep) override;

      /** Internal method */
      virtual bool store(agxStream::StorageStream& str) const override;

      /** Internal method */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::CombustionEngine);

      DOXYGEN_END_INTERNAL_BLOCK()

      CombustionEngine() = default;

    protected:
      virtual ~CombustionEngine();

    private:

      /**
      Function for integrating the pressure derivative to calculate the current inlet manifold pressure.
      Also calulates the update to the engine torque.
      \param timeStep - simulation time step
      */
      void stepPressure(agx::Real timeStep);

      /**
      Function to be called just when starting up the engine. It prescribes the inlet pressure and the engine speed
      to appropriate initial values.
      */
      void startEngine();

    private:

      agx::Real m_inletVolume;
      agx::Real m_inletPressure;

      agx::Real m_maxVolumetricEfficiency;
      agx::Real m_idleThrottleAngle;

      agx::Real m_maxThrottleAngle;
      agx::Real m_throttleAngle;

      agx::Real m_throttlePinBoreRatio;
      agx::Real m_maxThrottleArea;
      
      agx::Real m_nrRevolutionsPerCycle;

      agx::Real m_inletTemperature;
      agx::Real m_ambientPressure;

      agx::Real m_airGasConstant;

      bool m_enable;
      bool m_started;

      agx::Real m_idleInletPressure;

      agx::Real m_c0;
      agx::Real m_c1;
      agx::Real m_c2;
      agx::Real m_c3;

      agx::Real m_airFuelRatio;
      agx::Real m_heatValue;
      agx::Real m_thermalEfficiency;

      agx::Real m_indicatedTorque;
      agx::Real m_frictionTorque;
      agx::Real m_pumpTorque;

      agx::Real m_outputTorque;

      CombustionEngineParameters m_combustionEngineParameters;
      FrictionTorqueParameters m_frictionTorqueParameters;
  };

  AGX_DECLARE_POINTER_TYPES(CombustionEngine);
}
