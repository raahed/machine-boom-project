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

#include <agxCable/CableDamageStateTypes.h>
#include <agxCable/export.h>

namespace agxCable
{
  /**
  A bit set used by the cable damage framework to control which parts of the
  cable damage input that a CableDamageState instance should calculate.
  */
  class AGXCABLE_EXPORT CableDamageStateFilter
  {
    public:
      CableDamageStateFilter();

      void set(agxCable::DamageStateTypes::StateType type, bool enable);
      bool get(agxCable::DamageStateTypes::StateType type) const;

    private:
      bool m_filter[agxCable::DamageStateTypes::NUM_CABLE_DAMAGE_DATA_TYPES];
  };
}
