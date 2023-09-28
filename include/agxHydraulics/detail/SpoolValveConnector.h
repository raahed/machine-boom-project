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



#ifndef AGXHYDRAULICS_DETAIL_SPOOL_VALVE_CONNECTOR_H
#define AGXHYDRAULICS_DETAIL_SPOOL_VALVE_CONNECTOR_H

#include <agxHydraulics/FlowConnector.h>

namespace agxHydraulics
{
  class SpoolValve;

  namespace detail
  {

    /**
    \internal
    Custom FlowConnector used by the SpoolValve to connect external FlowUnits
    to internal static FlowUnits. Places restrictions on how Units may be
    connected, and only allows a SpoolValve to do the connecting.
    */
    class AGXHYDRAULICS_EXPORT SpoolValveConnector : public agxHydraulics::FlowConnector
    {
    public:
      SpoolValveConnector();

      /**
      Does nothing and returns false. Use internalConnect instead, which may
      only be called by the SpoolValve.

      \see internalConnect
      \return false
      */
      virtual bool connect(
          agxPowerLine::Side mySide,
          agxPowerLine::Side unitSide,
          agxPowerLine::Unit* unit) override;

      // Inherint the versions of connect that eventually call the two above.
      using agxHydraulics::FlowConnector::connect;

      /// \return A regular FlowConstraint, with a custom name.
      virtual agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() override;

      /**
      Will reject connections if the SpoolValveConnector is currently connected
      to an internal static FlowUnit.
      */
      virtual bool vetoConnect(agxPowerLine::Unit* unit) const override;


      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;


      /**
      Set values of constraints. Must be called after the constraints are created.
      */
      virtual bool postStore(agxStream::StorageStream& str) const override { return FlowConnector::postStore(str); }
      virtual bool postRestore(agxStream::StorageStream& str) override { return FlowConnector::postRestore(str); }

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::SpoolValveConnector);

    protected:
      virtual ~SpoolValveConnector();
      bool isConnectedToInternalStaticFlowUnit() const;

      friend class agxHydraulics::SpoolValve;
      bool internalConnect(agxPowerLine::Unit* unit, agxPowerLine::Side mySide, agxPowerLine::Side unitSide);
    };
  }
}

#endif
