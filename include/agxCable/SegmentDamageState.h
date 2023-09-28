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

#include <agxCable/CableDamageStateTypes.h>
#include <agxCable/export.h>
#include <cstddef>
#include <iosfwd>


namespace agxCable
{
  class CableDamageStateFilter;
}

namespace agxCable
{
  class SegmentDamageState;
  using SegmentDamageStateVector = agx::Vector<SegmentDamageState>;

  /**
  A SegmenDamageState instance records a collection of states computed from a
  cable for later use by a cable damage estimation model. The SegmentDamageState
  instances are held by a CableDamageState instance.
  */
  class AGXCABLE_EXPORT SegmentDamageState
  {
    public:
      /**
      Create a SegmentDamageState with all states set to NaN.
      */
      SegmentDamageState();

      agx::Real stretchTension() const;
      agx::Real bendTension() const;
      agx::Real twistTension() const;
      agx::Real frictionForce() const;
      agx::Real normalForce() const;
      agx::Real impactSpeed() const;
      agx::Real normalImpactSpeed() const;
      agx::Real tangentialImpactSpeed() const;
      agx::Real stretch() const;
      agx::Real stretchRate() const;
      agx::Real bend() const;
      agx::Real bendRate() const;
      agx::Real twist() const;
      agx::Real twistRate() const;

      agx::Real& stretchTension();
      agx::Real& bendTension();
      agx::Real& twistTension();
      agx::Real& frictionForce();
      agx::Real& normalForce();
      agx::Real& impactSpeed();
      agx::Real& normalImpactSpeed();
      agx::Real& tangentialImpactSpeed();
      agx::Real& stretch();
      agx::Real& stretchRate();
      agx::Real& bend();
      agx::Real& bendRate();
      agx::Real& twist();
      agx::Real& twistRate();


      agx::Real total() const;
      agx::Real tension() const;
      agx::Real contact() const;

      agx::Real& operator[](size_t i);
      agx::Real operator[](size_t i) const;

      /**
      Check if this SegmenDamageState has computed all states indicated by the
      given filter.

      \param filter - The filter to test against.
      \return True if and only if all states indicated by the given filter has been computed.
      */
      bool isValid(const CableDamageStateFilter& filter) const;

      SegmentDamageState& operator+=(const SegmentDamageState& rhs);

    private:
      agx::Real m_damages[DamageStateTypes::NUM_CABLE_DAMAGE_DATA_TYPES];
  };


  AGXCABLE_EXPORT std::ostream& operator<<(std::ostream& stream, const SegmentDamageState& damage);
}
