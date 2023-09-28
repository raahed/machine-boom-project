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


#ifndef AGXHYDRAULICS_DETAIL_FLOW_RATE_CONSTRAINT_H
#define AGXHYDRAULICS_DETAIL_FLOW_RATE_CONSTRAINT_H

#include <agx/Constraint.h>
#include <agx/ConstraintImplementation.h>
#include <agxHydraulics/FlowDimension.h>


/*
This file contains various constraint related classes used by the ConstantFlowValve.
*/


namespace agxHydraulics
{
  class FlowUnit;

  /// \cond INTERNAL_DOCUMENTATION
  namespace detail
  {
    /**
    Communication interface between the FlowRateConstraint and the FlowUnit
    using it. The FlowUnit owns and maintains an instance of the FlowRate-
    Constraint data and the constraints have a pointer to the same instance.
    */
    struct FlowRateConstraintData
    {
      agx::Real targetFlowRate;
      agx::Real currentFlowRate;
      agx::UInt8 elementIndex;
      bool allowPumping;

      FlowRateConstraintData(agx::Real targetFlowRate, agx::UInt8 elementIndex, bool allowPumping);
    };



    /**
    \internal
    A one-sided velocity constraint that acts to prevent flow rates in the
    given element index higher than the target flow rate.
    */
    class ElementaryFlowRateConstraint : public agx::ElementaryConstraintN<1>
    {
      public:
        ElementaryFlowRateConstraint(const FlowRateConstraintData& data);

        /**
        Updates the force range to match the current flow rate and the allow
        pumping setting.
        */
        virtual void prepare() override;

        /**
        Writes 1 to the element index of G[row].
        */
        virtual agx::UInt getJacobian(
            agx::Jacobian6DOFElement* G, agx::UInt numBlocks,
            agx::UInt row, agx::GWriteState::Enum writeState) override;

        /**
        Writes the target flow rate to v[row].
        */
        virtual agx::UInt getVelocity(agx::Real* v, agx::UInt row) const override;

        /**
        Writes 0 to v[row].
        */
        virtual agx::UInt getViolation(agx::Real* b, agx::UInt row) override;

        /**
        Change the FlowRateConstraintData store to use. Used after a restore
        since we cannot safely store pointers to non-serializable objects across
        a store/restore boundary.
        */
        void setDataLocation(FlowRateConstraintData* data);

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::ElementaryFlowRateConstraint);

      protected:
        ElementaryFlowRateConstraint();
        virtual ~ElementaryFlowRateConstraint() {}

      private:
        // To avoid warning C4512. And these really should not be copyable.
        void operator=(const ElementaryFlowRateConstraint&) {}

      private:
        const FlowRateConstraintData* m_data;

        // The velocity used by getVelocity. Updated in prepare.
        agx::Real m_constraintVelocity;
    };


    ElementaryFlowRateConstraint* asElementaryFlowRateConstraint(agx::Constraint* constraint);



    /**
    Constraint implementation holding a single ElementaryFlowRateConstraint.
    */
    class FlowRateConstraintImplementation : public agx::ConstraintImplementation, public agxStream::Serializable
    {
      public:
        FlowRateConstraintImplementation(const FlowRateConstraintData& data, FlowUnit* flowUnit);
        void setFlowDimension(agxHydraulics::FlowDimension* flowDim);

        virtual void prepare() override;

        virtual void storeLightData(agxStream::StorageStream& str) const override;
        virtual void restoreLightData(agxStream::StorageStream& str) override;

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::FlowRateConstraintImplementation);

      protected:
        virtual ~FlowRateConstraintImplementation() {}
        FlowRateConstraintImplementation();

      private:
        agxHydraulics::FlowDimension* m_flowDimension;
    };



    AGX_DECLARE_POINTER_TYPES(FlowRateConstraint);

    /**
    A constraint holding a single FlowConstraintImplementation.

    The FlowRateConstraint acts to limit the absolute value of the flow rate of
    the associated FlowUnit. Whenever the flow rate want to exceed the target
    flow rate the constraint will kick in to prevent that. Flow rates within the
    [-targetFlowRate, targetFlowRate] range are allowed.

    There is a one time step delay on direction changes. The constraint can
    only work in one direction at the time and the current direction is
    determined during constraint preparation. If something in the system causes
    the fluid to change direction and accelerate past the target flow rate in a
    single time step, then the constraint will be unable to prevent a too high
    flow rate until the next time step.
    */
    class FlowRateConstraint : public agx::Constraint
    {
      public:
        /**
        Create a FlowRateConstraint for the given flow unit.
        */
        FlowRateConstraint(const FlowRateConstraintData& data, FlowUnit* flowUnit);
        void setFlowDimension(agxHydraulics::FlowDimension* flowDim);
        virtual void render(agxRender::RenderManager* canvas, float scale) const override;

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::FlowRateConstraint);

      protected:
        FlowRateConstraint();
        virtual ~FlowRateConstraint();
        virtual int getNumDOF() const override;

      private:
        FlowRateConstraintImplementation* m_flowRateConstraint;
    };


  }
  /// \endcond
}


#endif
