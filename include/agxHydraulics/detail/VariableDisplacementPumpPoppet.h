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

#ifndef AGXHYDRAULICS_DETAIL_VARIABLE_DISPLACEMENT_PUMP_POPPET_H
#define AGXHYDRAULICS_DETAIL_VARIABLE_DISPLACEMENT_PUMP_POPPET_H

#include <agxPowerLine/TranslationalUnit.h>
#include <agxHydraulics/export.h>

namespace agxHydraulics
{
  /// \cond INTERNAL_DOCUMENTATION
  namespace detail
  {
    /**
    Simple translational unit used by the VariableDisplacementPump to determine
    how much to pump. Operated by the various variable displacement pump
    constraints.
    */
    class VariableDisplacementPumpPoppet : public agxPowerLine::TranslationalUnit
    {
      public:
        VariableDisplacementPumpPoppet();

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::VariableDisplacementPumpPoppet);
        using agxPowerLine::TranslationalUnit::store;
        using agxPowerLine::TranslationalUnit::restore;

      protected:
        virtual ~VariableDisplacementPumpPoppet() {}
    };
  }
  /// \endcond
}

#endif
