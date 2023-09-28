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


#ifndef AGXHYDRAULICS_FLOW_DIMENSION_H
#define AGXHYDRAULICS_FLOW_DIMENSION_H

#include <agxHydraulics/export.h>

#include <agxPowerLine/PhysicalDimension.h>
#include <agxPowerLine/detail/DimensionState.h>


#ifndef SWIG
namespace agxHydraulics
{
  const agx::Real complianceScaling(1);
  const agx::Real dampingScaling(1);
}
#endif


namespace agxHydraulics
{
  class FlowUnit;
  class FlowConnector;
}



namespace agxHydraulics
{
  AGX_DECLARE_POINTER_TYPES(FlowDimension);
  AGX_DECLARE_VECTOR_TYPES(FlowDimension);


  /**
  A FlowDimension represents the flow of fluid through a FlowUnit.
  */
  class AGXHYDRAULICS_EXPORT FlowDimension : public agxPowerLine::PhysicalDimension
  {
    public:
      FlowDimension(agxHydraulics::FlowUnit* unit);

      FlowDimension(agxHydraulics::FlowUnit* unit, agx::RigidBody* body, agx::UInt8 slot);


      /**
      \return The FlowConnector occupying the given side of the FlowDimension,
              or nullptr if there is no such FlowConnector.
      */
      agxHydraulics::FlowConnector* getFlowConnector(agxPowerLine::Side side);
      const agxHydraulics::FlowConnector* getFlowConnector(agxPowerLine::Side side) const;

      /**
      \return The FlowConnector occupying the input side of the FlowDimension,
              or nullptr if there is no such FlowConnector.
      */
      agxHydraulics::FlowConnector* getInputFlowConnector();
      const agxHydraulics::FlowConnector* getInputFlowConnector() const;

      /**
      \return The FlowConnector occupying the output side of the FlowDimension,
              or nullptr if there is no such FlowConnector.
      */
      agxHydraulics::FlowConnector* getOutputFlowConnector();
      const agxHydraulics::FlowConnector* getOutputFlowConnector() const;


      /**
      \return The FlowUnit that owns this FlowDimension.
      */
      FlowUnit* getFlowUnit();
      const FlowUnit* getFlowUnit() const;


      /**
      Returns the sum of the constraint forces of all constraints owned by
      Connectors connected to the output side of this FlowDimension. The
      constraint force is often a pressure, but may be something else as well.
      Use with caution.
      */
      virtual agx::Real getOutputLoad() const override;

      /**
      Returns the sum of the constraint forces of all constraints owned by
      Connectors connected to the input side of this FlowDimension. The
      constraint force is often a pressure, but may be something else as well.
      Use with caution.
      */
      virtual agx::Real getInputLoad() const override;

      /**
      Compute the combined power, i.e., pressure * flow rate, generated at all
      output connections. The returned power will be wrong if there is a
      connection to a constraint whose constraint force is not in units of
      pressure.
      */
      virtual agx::Real getPowerOut() const override;

      /**
      Compute the combined power, i.e., pressure * flow rate, generated at all
      input connections. The returned power will be wrong if there is a
      connection to a constraint whose constraint force is not in units of
      pressure.
      */
      virtual agx::Real getPowerIn() const override;






    // Methods called by the rest of the PowerLine/Hydraulics framework.
    public:
      /**
      \return The name of the flow dimension type.
      */
      static std::string getStaticTypeName();

      /**
      \return The name of the flow dimension type.
      */
      virtual std::string getTypeName() const override;

      /**
      The ID of a physical dimension type is always a number between zero and
      the total number of types of physical dimensions in the system, which can
      be queried using PowerLineController::getNumberOfPhysicalDimensions.

      \return The ID of the flow dimension type.
      */
      static agxPowerLine::PhysicalDimension::Type getStaticType();

      agxPowerLine::detail::Translational1DofState* getFlowState();
      const agxPowerLine::detail::Translational1DofState* getFlowState() const;

      agx::UInt8 getElementIndex() const;


      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::FlowDimension);

#ifndef SWIG

      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& out) const override;

      using agxPowerLine::PhysicalDimension::store;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& in) override;

      using agxPowerLine::PhysicalDimension::restore;
#endif


    protected:
      FlowDimension();
      virtual ~FlowDimension() {}

    private:
      typedef agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* (agxHydraulics::FlowDimension::*GetConstraint)(size_t) const;
      typedef const agxPowerLine::Connection* (agxHydraulics::FlowDimension::*GetConnection)(size_t) const;
      agx::Real sumPower(
        const agxPowerLine::ConnectionRefVector& connections,
        GetConstraint getConstraintFunction,
        GetConnection getConnectionFunction) const;

    private:
      agxPowerLine::detail::Translational1DofState* m_flowState;
  };
}
#endif
