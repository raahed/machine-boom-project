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
#include <agxDriveTrain/ElectricMotorConstraint.h>

namespace agxDriveTrain
{
  /**
  An electric motor is a roational unit that is a source of power in a drive train.
  It simulates the conversion of electrical energy to mechanical energy. The angular
  velocity of the motor is changed by changing the input voltage to the motor, this
  can be done at each time step.
  */
  class AGXMODEL_EXPORT ElectricMotor : public agxPowerLine::RotationalUnit
  {
    public:
      /**
      Create an electrical motor with the properties of the given parameters
      \param resistance - electrical resistance of the motor
      \param torqueConstant - torque constant of the motor
      \param emfConstant - back EMF constant of the motor
      \param inductance - inductance of the motor
      */
      ElectricMotor(agx::Real resistance, agx::Real torqueConstant, agx::Real emfConstant, agx::Real inductance);

      /**
      Set the input voltage to the electrical motor. Higher voltage will cause
      the motor to spin faster.
      */
      void setVoltage(agx::Real voltage);
      /**
      /return The current input voltage of the motor.
      */
      agx::Real getVoltage() const;

      /**
      Set the electrical reistance of the motor.
      */
      void setResistance(agx::Real resistance);
      /**
      \return The reistance of the motor.
      */
      agx::Real getResistance() const;

      /**
      Set the torque constant of the motor. Sets how the torque changes with the current
      in the motor.
      */
      void setTorqueConstant(agx::Real torqueConstant);
      /**
      \return The torque constant of the motor.
      */
      agx::Real getTorqueConstant() const;

      /**
      Set the back electromotive force constant. The back EMF constant describes the
      relationship between the motor's back EMF and its angular velocity.
      */
      void setEmfConstant(agx::Real emfConstant);
      agx::Real getEmfConstant() const;

      /**
      Set the inductance of the motor. Larger inductance means the motor will take
      more time to accelerate to its maximum speed when the voltage is changed.
      */
      void setInductance(agx::Real inductance);
      /**
      \return The inductance of the motor.
      */
      agx::Real getInductance() const;

      /**
      \return The current through the motor.
      */
      agx::Real getCurrent() const;
      /**
      \return The back electromotive force.
      */
      agx::Real getEmfVoltage() const;

      /**
       * \return The ElectricMotorConstraint of the motor.
       */
      const agxDriveTrain::ElectricMotorConstraint* getConstraint() const;

    public:
      /** \internal */
      virtual bool addNotification(agxSDK::Simulation* simulation) override;
      /** \internal */
      virtual bool removeNotification(agxUtil::ConstraintHolder* constraintHolder, agxSDK::Simulation* simulation) override;

      /** \internal */
      virtual bool preUpdate(agx::Real timeStep) override;

      /** \internal */
      virtual bool store(agxStream::StorageStream& str) const override;
      /** \internal */
      virtual bool postStore(agxStream::StorageStream& str) const override;

      /** \internal */
      virtual bool restore(agxStream::StorageStream& str) override;
      /** \internal */
      virtual bool postRestore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::ElectricMotor);

    protected:
      ElectricMotor() = default;

      using base = agxPowerLine::RotationalUnit;
      virtual ~ElectricMotor();

    private:
      agx::Real m_resistance;
      agx::Real m_torqueConstant;
      agx::Real m_emfConstant;
      agx::Real m_inductance;
      agx::Real m_voltage;

      ElectricMotorConstraintRef m_motor;
  };

  AGX_DECLARE_POINTER_TYPES(ElectricMotor);
}


