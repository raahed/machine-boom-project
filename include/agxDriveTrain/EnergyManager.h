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
#include <agx/Real.h>

// Forward declarations

namespace agxSDK {
  class Simulation;
}

#ifndef SWIG
namespace agxDriveTrain
{
  class ElectricMotor;
  class CombustionEngine;
  class Engine;
  class FixedVelocityEngine;
}
#endif

namespace agxDriveTrain
{
  class AGXMODEL_EXPORT EnergyManager
  {
  public:

    /**
    Adds an electric motor to the EnergyManager.
    \param motor - The electric motor to be added.
    \param simulation - The simulation. This parameter is only necessary if the PowerLine of the motor is not added to
    a simulation.
    */
    static void add( const agxDriveTrain::ElectricMotor* motor, const agxSDK::Simulation* simulation = nullptr );

    /**
    Adds an combustion engine to the EnergyManager.
    \param motor - The combustion engine to be added.
    \param simulation - The simulation. This parameter is only necessary if the PowerLine of the motor is not added to
    a simulation.
    */
    static void add( const agxDriveTrain::CombustionEngine* motor, const agxSDK::Simulation* simulation = nullptr );

    /**
    Adds an engine to the EnergyManager.
    \param motor - The engine to be added.
    \param simulation - The simulation. This parameter is only necessary if the PowerLine of the motor is not added to
    a simulation.
    */
    static void add( const agxDriveTrain::Engine* motor, const agxSDK::Simulation* simulation = nullptr );

    /**
    Adds an fixed velocity engine to the EnergyManager.
    \param motor - The fixed velocity engine to be added.
    \param simulation - The simulation. This parameter is only necessary if the PowerLine of the motor is not added to
    a simulation.
    */
    static void add( const agxDriveTrain::FixedVelocityEngine* motor, const agxSDK::Simulation* simulation = nullptr );

    /**
    Removes an electric motor from the EnergyManager.
    \param motor - The electric motor to be removed.
    \param simulation - The simulation. This parameter is only necessary if the PowerLine of the motor is not added to
    a simulation.
    */
    static void remove( const agxDriveTrain::ElectricMotor* motor, const agxSDK::Simulation* simulation = nullptr );

    /**
    Removes a combustion engine from the EnergyManager.
    \param motor - The combustion engine to be removed.
    \param simulation - The simulation. This parameter is only necessary if the PowerLine of the motor is not added to
    a simulation.
    */
    static void remove( const agxDriveTrain::CombustionEngine* motor, const agxSDK::Simulation* simulation = nullptr );

    /**
    Removes an engine from the EnergyManager.
    \param motor - The engine to be removed.
    \param simulation - The simulation. This parameter is only necessary if the PowerLine of the motor is not added to
    a simulation.
    */
    static void remove( const agxDriveTrain::Engine* motor, const agxSDK::Simulation* simulation = nullptr );

    /**
    Removes an fixed velocity engine from the EnergyManager.
    \param motor - The fixed velocity engine to be removed.
    \param simulation - The simulation. This parameter is only necessary if the PowerLine of the motor is not added to
    a simulation.
    */
    static void remove( const agxDriveTrain::FixedVelocityEngine* motor,
                        const agxSDK::Simulation* simulation = nullptr );

    /**
    Get the work done by an electric motor.
    \param motor - The electric motor.
    \return The work done.
    */
    static agx::Real getWorkDone( const agxDriveTrain::ElectricMotor* motor );

    /**
    Get the work done by a combustion engine.
    \param motor - The combustion engine.
    \return The work done.
    */
    static agx::Real getWorkDone( const agxDriveTrain::CombustionEngine* motor );

    /**
    Get the work done by an engine.
    \param motor - The engine.
    \return The work done.
    */
    static agx::Real getWorkDone( const agxDriveTrain::Engine* motor );

    /**
    Get the work done by a fixed velocity engine.
    \param motor - The fixed velocity engine.
    \return The work done.
    */
    static agx::Real getWorkDone( const agxDriveTrain::FixedVelocityEngine* motor );

    /**
    Get the power for an electric motor.
    \param motor - The electric motor.
    \return The power.
    */
    static agx::Real getPower( const agxDriveTrain::ElectricMotor* motor );

    /**
    Get the power for a combustion engine.
    \param motor - The combustion engine.
    \return The power.
    */
    static agx::Real getPower( const agxDriveTrain::CombustionEngine* motor );

    /**
    Get the power for an engine.
    \param motor - The engine.
    \return The power.
    */
    static agx::Real getPower( const agxDriveTrain::Engine* motor );

    /**
    Get the power for a fixed velocity engine.
    \param motor - The fixed velocity engine.
    \return The power.
    */
    static agx::Real getPower( const agxDriveTrain::FixedVelocityEngine* motor );

  private:
    EnergyManager() = default;
  };
}
