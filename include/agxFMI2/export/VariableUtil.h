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

#ifndef AGXFMI2_EXPORT_VARIABLEUTIL_H
#define AGXFMI2_EXPORT_VARIABLEUTIL_H

#include <agxFMI2/export/Variable.h>
#include <agx/Constraint.h>
#include <agx/ParticleEmitter.h>

namespace agxFMI2
{
  namespace Export
  {
    namespace Constraint
    {
      // Constraint
      AGXFMI_EXPORT InputVariable_Real *setCompliance(agx::Constraint* constraint);
      AGXFMI_EXPORT InputVariable_Real *setDamping(agx::Constraint* constraint);

      // Motor
      // AGXFMI_EXPORT InputVariable_Bool *enableMotor(agx::Constraint1DOF* constraint);
      // AGXFMI_EXPORT InputVariable_Bool *enableMotor(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

      AGXFMI_EXPORT InputVariable_Real *setMotorSpeed(agx::Constraint1DOF* constraint);
      AGXFMI_EXPORT InputVariable_Real *setMotorSpeed(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

      AGXFMI_EXPORT InputVariable_Real *setMotorMinForceRange(agx::Constraint1DOF* constraint);
      AGXFMI_EXPORT InputVariable_Real *setMotorMinForceRange(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

      AGXFMI_EXPORT InputVariable_Real *setMotorMaxForceRange(agx::Constraint1DOF* constraint);
      AGXFMI_EXPORT InputVariable_Real *setMotorMaxForceRange(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

      AGXFMI_EXPORT InputVariable_Real *setMotorCompliance(agx::Constraint1DOF* constraint);
      AGXFMI_EXPORT InputVariable_Real *setMotorCompliance(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

      // Range
      // AGXFMI_EXPORT InputVariable_Bool *enableRange(agx::Constraint1DOF* constraint);
      // AGXFMI_EXPORT InputVariable_Bool *enableRange(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

      AGXFMI_EXPORT InputVariable_Real *setRangeMinRange(agx::Constraint1DOF* constraint);
      AGXFMI_EXPORT InputVariable_Real *setRangeMinRange(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

      AGXFMI_EXPORT InputVariable_Real *setRangeMaxRange(agx::Constraint1DOF* constraint);
      AGXFMI_EXPORT InputVariable_Real *setRangeMaxRange(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

      AGXFMI_EXPORT InputVariable_Real *setRangeMinForceRange(agx::Constraint1DOF* constraint);
      AGXFMI_EXPORT InputVariable_Real *setRangeMinForceRange(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

      AGXFMI_EXPORT InputVariable_Real *setRangeMaxForceRange(agx::Constraint1DOF* constraint);
      AGXFMI_EXPORT InputVariable_Real *setRangeMaxForceRange(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

      AGXFMI_EXPORT InputVariable_Real *setRangeCompliance(agx::Constraint1DOF* constraint);
      AGXFMI_EXPORT InputVariable_Real *setRangeCompliance(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

      // Lock
      // AGXFMI_EXPORT InputVariable_Bool *enableLock(agx::Constraint1DOF* constraint);
      // AGXFMI_EXPORT InputVariable_Bool *enableLock(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

      AGXFMI_EXPORT InputVariable_Real *setLockPosition(agx::Constraint1DOF* constraint);
      AGXFMI_EXPORT InputVariable_Real *setLockPosition(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

      AGXFMI_EXPORT InputVariable_Real *setLockMinForceRange(agx::Constraint1DOF* constraint);
      AGXFMI_EXPORT InputVariable_Real *setLockMinForceRange(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

      AGXFMI_EXPORT InputVariable_Real *setLockMaxForceRange(agx::Constraint1DOF* constraint);
      AGXFMI_EXPORT InputVariable_Real *setLockMaxForceRange(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

      AGXFMI_EXPORT InputVariable_Real *setLockCompliance(agx::Constraint1DOF* constraint);
      AGXFMI_EXPORT InputVariable_Real *setLockCompliance(agx::Constraint2DOF* constraint, agx::Constraint2DOF::DOF dof);

    }

    namespace ParticleEmitter
    {
      AGXFMI_EXPORT InputVariable_Real *setEmitterRate(agx::ParticleEmitter* emitter);
    }

  }

}


#endif /* AGXFMI2_EXPORT_VARIABLEUTIL_H */
