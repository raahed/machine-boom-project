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



#ifndef AGXHYDRAULICS_DETAIL_SPOOL_VALVE_STATIC_FLOW_UNIT_H
#define AGXHYDRAULICS_DETAIL_SPOOL_VALVE_STATIC_FLOW_UNIT_H

#include <agxHydraulics/FlowUnit.h>

namespace agxHydraulics
{
  namespace detail
  {
    /**
    \internal
    Custom FlowUnit used by the spool valve to prevent flow in connected but
    unlinked external FlowUnits. Always has STATIC motion control.
    */
    class AGXHYDRAULICS_EXPORT SpoolValveStaticFlowUnit : public agxHydraulics::FlowUnit
    {
    public:
      SpoolValveStaticFlowUnit(agxHydraulics::FlowUnit* blockedFlowUnit);

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::SpoolValveStaticFlowUnit);
      using agxHydraulics::FlowUnit::store;
      using agxHydraulics::FlowUnit::restore;

    protected:
      SpoolValveStaticFlowUnit();
      virtual ~SpoolValveStaticFlowUnit();
    };
  }
}

#endif
