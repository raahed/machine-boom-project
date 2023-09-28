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


#ifndef AGXHYDRAULICS_SPOOL_VALVE_CONNECTOR_H
#define AGXHYDRAULICS_SPOOL_VALVE_CONNECTOR_H

#include <agxHydraulics/FlowConnector.h>
#include <agxHydraulics/FlowUnit.h>

#include <agxHydraulics/detail/SpoolValveConnectionInfo.h>


namespace agxHydraulics
{
  AGX_DECLARE_POINTER_TYPES(SpoolValve);


  /**
  Spool valves are used to dynamically redirect flow during a simulation. It
  connects to any number of FlowUnits and allows arbitrary linking of the
  connected Units. Flow can only pass between FlowUnits that are linked.

  The FlowUnits form disjoint sets of linked Units, where fluid can flow freely
  to and from any FlowUnit within each set. Connecting a FlowUnit to the
  SpoolValve creates a new set containing that FlowUnit only.

  A FlowUnit that is the sole member of a FlowUnit set is blocked since there is
  nowhere for the fluid to go.
  */
  class AGXHYDRAULICS_EXPORT SpoolValve : public agxPowerLine::Connector
  {
    public:
      SpoolValve();

      /**
      Create a link between the two given FlowUnits, in effect merging the two
      FlowUnit sets into a single set.

      Linking will fail if either argument is nullptr, the operation would
      link a FlowUnit to itself, either of the FlowUnits is not connected to
      the SpoolValve and if the given FlowUnits are already connected.

      \return True if the link was created. False otherwise.
      */
      bool link(agxHydraulics::FlowUnit* input, agxHydraulics::FlowUnit* output);

      /**
      Remove the given FlowUnit from its current FlowUnit set, making it the
      sole member of a new FlowUnit set.

      Notice that linking two FlowUnits and then unlinking one of them may
      produce a configuration different from the pre-link system.
      */
      bool unlink(agxHydraulics::FlowUnit* flowUnit);

      /**
      \return True if the two given FlowUnits are part of the same FlowUnit set.
              False otherwise.
      */
      bool doesLinkExist(agxHydraulics::FlowUnit* input, agxHydraulics::FlowUnit* output) const;

      /**
      \return The number of FlowUnit sets in the SpoolValve. Does not count unit sets.
      */
      size_t getNumLinks() const;


      /**
      Connect the given Unit, which must be a FlowUnit, to the SpoolValve. The
      newly connected Unit will become the sole member of a new FlowUnit set
      and therefore be blocked.

      The connect will fail if the given Unit is not a FlowUnit, if the given
      FlowUnit is already connected to the SpoolValve, if the given unitSide is
      not a valid side and if the given FlowUnit already has a flow connection
      on the given unitSide.

      The mySide argument is ignored.

      \return True if the Unit was connected to the SpoolValve. False otherwise.
      */
      virtual bool connect(
          agxPowerLine::Side mySide,
          agxPowerLine::Side unitSide,
          agxPowerLine::Unit* unit) override;

      using Connector::connect;

      virtual bool disconnect(agxPowerLine::Unit* unit) override;

      virtual bool disconnect() override;



    // Methods called by the rest of the PowerLine/Hydraulics frame work.
    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      virtual agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* createConstraint() override;

#ifndef SWIG
      /**
      */
      virtual bool removeNotification(
          agxUtil::ConstraintHolder* constraintHolder,
          agxSDK::Simulation* simulation) override;


      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const override;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::SpoolValve);
#endif
      DOXYGEN_END_INTERNAL_BLOCK()

      typedef agx::HashVector<agxHydraulics::FlowUnit*, detail::SpoolValveConnectionInfo> UnitPtrSpoolValveInfoTable;

    protected:
      virtual ~SpoolValve();

    private:
      void removeInternalFlowConnector(agxHydraulics::FlowConnector* connector);
      void removeInternalFlowUnit(agxHydraulics::FlowUnit* unit);

    private:
      UnitPtrSpoolValveInfoTable m_connectionInfo;
  };
}


#endif
