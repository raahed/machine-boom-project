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

#include <agx/agx.h>

namespace agxUtil {
  namespace convert {

    /// Converts spring constant to compliance. This is done by taking the inverse (compliance = 1/springConstant).
    AGXPHYSICS_EXPORT agx::Real convertSpringConstantToCompliance(agx::Real springConstant);

    /// Converts spring constant to compliance. This is done by taking the inverse (springConstant = 1/compliance).
    AGXPHYSICS_EXPORT agx::Real convertComplianceToSpringConstant(agx::Real compliance);

    /// Converts elasticity to compliance. This is done by taking the inverse (compliance = 1/elasticity).
    AGXPHYSICS_EXPORT agx::Real convertElasticityToCompliance(agx::Real elasticity);

    /// Converts compliance to elasticity. This is done by taking the inverse (elasticity = 1/compliance).
    AGXPHYSICS_EXPORT agx::Real convertComplianceToElasticity(agx::Real compliance);

    /**
    Converts (viscous) damping coefficient to spook damping.
    This is done by dividing them:
    spookDamping = dampingCoefficient / springConstant
    \param dampingCoefficient - the damping coefficient (for linear dimensions, in force*time/distance; for rotational dimensions, in torque*time/radians).
    \param springConstant - the spring constant (for linear dimensions, in force/distance; for rotational dimensions, in torque/radians).
    \retval damping as used in agx-constraints and contacts (in time)
    */
    AGXPHYSICS_EXPORT agx::Real convertDampingCoefficientToSpookDamping(
      agx::Real dampingCoefficient,
      agx::Real springConstant);

    /**
    Converts spook damping to (viscous) damping coefficient .
    This is done by multiplying them:
    dampingCoefficient = springConstant * spookDamping
    \param spookDamping as used in agx-constraints and contacts (in time)
    \param springConstant - the spring constant (for linear dimensions, in force/distance; for rotational dimensions, in torque/radians).
    \retval dampingCoefficient - the damping coefficient (for linear dimensions, in force*time/distance; for rotational dimensions, in torque*time/radians).
    */
    AGXPHYSICS_EXPORT agx::Real convertSpookDampingToDampingCoefficient(
      agx::Real spookDamping,
      agx::Real springConstant);
  }
}

