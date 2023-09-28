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


#ifndef AGXHYDRAULICS_DETAIL_SPOOL_VALVE_CONNECTION_INFO_H
#define AGXHYDRAULICS_DETAIL_SPOOL_VALVE_CONNECTION_INFO_H

#include <agxHydraulics/FlowUnit.h>
#include <agxPowerLine/Sides.h>
#include <agx/Referenced.h>
#include <agxStream/Serializable.h>

namespace agxHydraulics
{
  /// \cond INTERNAL_DOCUMENTATION
  namespace detail
  {

    /**
    SpoolValveConnectionInfo is a state handling class used by the SpoolValve.
    There is one SpoolValveConnectionInfo for every external FlowUnit connected
    to the SpoolValve and it details the manner in which that external FlowUnit
    is connected to the SpoolValve.
    */
    struct AGXHYDRAULICS_EXPORT SpoolValveConnectionInfo : public agxStream::Serializable
    {
      public:
        /**
        \param staticFlowUnit - The spool valve's own internal blocker unit.
        \param pipeSide - The side of the external FlowUnit that the spool valve is connected to.
        \param staticPipeSide - The side of the internal blocker unit that the external FlowUnit may connect to.
        */
        SpoolValveConnectionInfo(
            FlowUnit* staticFlowUnit,
            agxPowerLine::Side pipeSide,
            agxPowerLine::Side staticPipeSide
        );

        /// Create an invalid SpoolValveConnectionInfo. Used to signal invalid states or unconnected FlowUnits.
        SpoolValveConnectionInfo();

        /// \return True if the SpoolValveConnectionInfo has been initialized.
        bool isValid() const;

        /**
        \param pipe - The external FlowUnit for which this SpoolValveConnectionInfo was created.
        \return The FlowConnector on the SpoolValve side pipe.
        */
        agxHydraulics::FlowConnector* currentConnector(agxHydraulics::FlowUnit* pipe);
        const agxHydraulics::FlowConnector* currentConnector(const agxHydraulics::FlowUnit* pipe) const;

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::SpoolValveConnectionInfo);

      public:
        // A static FlowUnit that the external FlowUnit is connected to when not linked with another external FlowUnit.
        agxHydraulics::FlowUnitRef staticFlowUnit;

        // The side of the external FlowUnit that is connected to the SpoolValve. The SpoolValve may only manipulate this side.
        agxPowerLine::Side pipeSide;

        // The side of the static FlowUnit that the external FlowUnit should be connected to when not linked.
        agxPowerLine::Side staticPipeSide;
    };
  }
  /// \endcond
}

#endif
