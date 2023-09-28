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

#include <agx/Constraint.h>
#include <agx/ConstraintImplementation.h>

#include <agxPowerLine/RotationalUnit.h>

namespace agxDriveTrain {
  /**
  \internal
  ElectricMotorConstraintImplementation is the implementation of the constraint for
  the electric motor. It contains a secondary constraint, a TargetSpeedController,
  which is used to drive the motor.
  */
  class ElectricMotorConstraintImplementation : public agx::HighLevelConstraintImplementation
  {
    public:
      /**
      \internal
      \param rotationalDimension - the rotational dimension of the motor
      */
      ElectricMotorConstraintImplementation(agxPowerLine::RotationalDimensionRef rotationalDimension);

      virtual void storeLightData(agxStream::StorageStream& str) const override;

      virtual void restoreLightData(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::ElectricMotorConstraintImplementation);

    protected:
      ElectricMotorConstraintImplementation();

    private:
      ~ElectricMotorConstraintImplementation();
      agx::TargetSpeedController* m_electricMotorConstraint;
      agxPowerLine::RotationalDimension* m_rotationalDimension;
  };


  /**
  \internal
  The constraint of the electric motor, which can drive the motor to rotate.
  */
  class ElectricMotorConstraint : public agx::Constraint
  {
    public:
      /**
      \internal
      \param rotationalDimension - the rotational dimension of the motor
      */
      ElectricMotorConstraint(agxPowerLine::RotationalDimensionRef rotationalDimension);

      /**
      \internal
      Set the rotational speed of the secondary constraint, the TargetSpeedController
      \param speed - speed of the motor rotation
      */
      void setSpeed(agx::Real speed);

      /**
      \internal
      Set the compliance of the secondary constraint, the TargetSpeedController
      \param compliance - the compliance of the motor constraint
      */
      void setCompliance(agx::Real compliance);
      using agx::Constraint::setCompliance;

      virtual void render(agxRender::RenderManager* /*canvas*/, float /*scale*/) const override {}

      AGXSTREAM_DECLARE_SERIALIZABLE(agxDriveTrain::ElectricMotorConstraint);

    protected:
      ElectricMotorConstraint();
      virtual ~ElectricMotorConstraint();
      virtual int getNumDOF() const override;

    private:
      ElectricMotorConstraintImplementation* m_electricMotorConstraint;
  };

  AGX_DECLARE_POINTER_TYPES(ElectricMotorConstraint);
}


