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

#include <agx/LockJoint.h>

namespace agxModel
{
  /**
  Stiffness and damping data for constraints related to beam models. The compliance
  of a constraint is 1.0 / stiffness.
  */
  struct AGXMODEL_EXPORT BeamStiffnessDamping
  {
    /**
    Sane default values for stiffness and damping that can be used
    when inputs are invalid but something has to be returned. Print
    warnings when these values are used.
    */
    static BeamStiffnessDamping createDefault();

    /**
    Translational stiffness where x and y are shear stiffness and
    z is stretch. Compliance is 1.0 / stiffness.
    */
    agx::Vec3 translationalStiffness;

    /**
    Rotational stiffness where x and y are bend stiffness and
    z is torsion. Compliance is 1.0 / stiffness.
    */
    agx::Vec3 rotationalStiffness;

    /**
    Translational damping time where x and y are shear damping and
    z is stretch.
    */
    agx::Vec3 translationalDampingTime;

    /**
    Rotational damping time where x and y are bend damping and
    z is torsion.
    */
    agx::Vec3 rotationalDampingTime;

    /**
    Apply stiffness as compliance to the given lock joint. The scale is multiplied
    with stiffness translational.z() (stretch) and all rotational DOFs.
    \param lock - lock joint to apply stiffness and damping data to
    \param scale - stiffness scale of translational z and all rotational DOFs (default and normally: 1.0)
    */
    void apply( agx::LockJoint* lock, agx::Real scale = 1.0 ) const;
  };

  inline void BeamStiffnessDamping::apply( agx::LockJoint* lock, agx::Real scale /*= 1.0*/ ) const
  {
    if ( lock == nullptr )
      return;

    constexpr agx::Real minStiffness = 1.0E-5;
    constexpr agx::Real minDamping = 0.0;

    lock->setCompliance( 1.0 / std::max( translationalStiffness.x(), minStiffness ), agx::LockJoint::TRANSLATIONAL_1 );
    lock->setCompliance( 1.0 / std::max( translationalStiffness.y(), minStiffness ), agx::LockJoint::TRANSLATIONAL_2 );
    lock->setCompliance( 1.0 / std::max( scale * translationalStiffness.z(), minStiffness ), agx::LockJoint::TRANSLATIONAL_3 );

    lock->setCompliance( 1.0 / std::max( scale * rotationalStiffness.x(), minStiffness ), agx::LockJoint::ROTATIONAL_1 );
    lock->setCompliance( 1.0 / std::max( scale * rotationalStiffness.y(), minStiffness ), agx::LockJoint::ROTATIONAL_2 );
    lock->setCompliance( 1.0 / std::max( scale * rotationalStiffness.z(), minStiffness ), agx::LockJoint::ROTATIONAL_3 );

    lock->setDamping( std::max( translationalDampingTime.x(), minDamping ), agx::LockJoint::TRANSLATIONAL_1 );
    lock->setDamping( std::max( translationalDampingTime.y(), minDamping ), agx::LockJoint::TRANSLATIONAL_2 );
    lock->setDamping( std::max( translationalDampingTime.z(), minDamping ), agx::LockJoint::TRANSLATIONAL_3 );

    lock->setDamping( std::max( rotationalDampingTime.x(), minDamping ), agx::LockJoint::ROTATIONAL_1 );
    lock->setDamping( std::max( rotationalDampingTime.y(), minDamping ), agx::LockJoint::ROTATIONAL_2 );
    lock->setDamping( std::max( rotationalDampingTime.z(), minDamping ), agx::LockJoint::ROTATIONAL_3 );
  }
}
