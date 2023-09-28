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

#include <agx/Real.h>
#include <agx/Vector.h>

#include <agxCable/CableDamageTypes.h>
#include <agxCable/export.h>

#include <cstddef>


namespace agxCable
{
  class SegmentDamage;
  using SegmentDamageVector = agx::Vector<SegmentDamage>;
  typedef agx::VectorPOD<SegmentDamage*> SegmentDamagePtrVector;

  /**
  A SegmentDamage instance records damages estimates computed for a single cable
  segment. It can hold either the contributions for a single time step, or
  accumulated contributions for several time steps. The SegmentDamage instances
  are held by a CableDamageImplementation instance.
  */
  class AGXCABLE_EXPORT SegmentDamage
  {
    public:
      /**
       * Create a SegmentDamage with all damages set to zero.
       */
      SegmentDamage();

      agx::Real bendDeformation() const;
      agx::Real twistDeformation() const;
      agx::Real stretchDeformation() const;

      agx::Real bendRate() const;
      agx::Real twistRate() const;
      agx::Real stretchRate() const;

      agx::Real bendTension() const;
      agx::Real twistTension() const;
      agx::Real stretchTension() const;

      agx::Real contactNormalForce() const;
      agx::Real contactFrictionForce() const;

      agx::Real& bendDeformation();
      agx::Real& twistDeformation();
      agx::Real& stretchDeformation();


      agx::Real& bendRate();
      agx::Real& twistRate();
      agx::Real& stretchRate();

      agx::Real& bendTension();
      agx::Real& twistTension();
      agx::Real& stretchTension();

      agx::Real& contactNormalForce();
      agx::Real& contactFrictionForce();

      /**
      \return The sum of all damage estimates.
      */
      agx::Real total() const;

      /**
      \param type - The type of damage to access.
      \return The damage recorded for the given type.
      */
      agx::Real operator[](DamageTypes::DamageType type) const;
      agx::Real& operator[](DamageTypes::DamageType type);

      /**
      Access damages based on index instead of type name.
      Convenience operator to make looping over all damages easier.
      The indices correspond to the enum literals in agxCable::DamageTypes.
      \p index must be less than \p agxCable::DamageTypes::NUM_CABLE_DAMAGE_TYPES.

      \param index - The index of the damage type to access.
      \return The damage recorded for the given index.
      */
      agx::Real operator[](size_t index) const;
      agx::Real& operator[](size_t index);

    private:
      agx::Real m_damages[DamageTypes::NUM_CABLE_DAMAGE_TYPES];
  };

  AGXCABLE_EXPORT SegmentDamage operator+(const SegmentDamage& lhs, const SegmentDamage& rhs);
  AGXCABLE_EXPORT SegmentDamage& operator+=(SegmentDamage& lhs, const SegmentDamage& rhs);
}
