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

#ifndef AGXHYDRAULICS_DETAIL_FLOW_CONSTRAINT_H
#define AGXHYDRAULICS_DETAIL_FLOW_CONSTRAINT_H

#include <agxPowerLine/PowerLineConstraints.h>
#include <agxHydraulics/export.h>

namespace agxHydraulics
{
  /// \cond INTERNAL_DOCUMENTATION
  namespace detail
  {
    /**
    The ElementaryFlowConstraint maintains a zero net flow through a junction. It
    is behaviorally identical to the the base power line constraint, which means
    that it acts to remove any constraint velocity and the constraint force is
    evenly distributed over the connected pipes.

    The constraint can operate in two modes: flexing or leaking. In the flexing
    mode the constraint integarates the flow in and out of the junction in
    order to track the current amount of fluid held by the junction itself. It
    acts as a spring loaded reservoir. The higher the pressure in the system
    the more fluid will be stored in the junction.

    In the leaking mode the junction will lose fluid instead of expanding. The
    higher the pressure the more fluid will be lost from the system.

    The mode of operation is controlled using the holonomic constraint parameter,
    where holonomic means flexing and non-holonomic means leaking.
    */
    class AGXHYDRAULICS_EXPORT ElementaryFlowConstraint : public agxPowerLine::ElementaryPhysicalDimensionMultiBodyConstraint
    {
      public:
        /**
        \parameter holonomic - True to create a holonomic (flexing) junction.
                               False to create a non-holonomic (leaking) junction.
        */
        ElementaryFlowConstraint(bool holonomic);

        /**
        \return The amount of fluid stored in the junction. Only valid for holonomic junctions.
        */
        agx::Real getStoredFluid() const;

        /**
        Set the amount of fluid stored in the junction. Only valid for
        holonomic junction.

        Use with caution since this translates directly to constraint violation
        and potentially large pressures in the following time steps.
        */
        void setStoredFluid(agx::Real storedFluid);



      // Methods called by the rest of the PowerLine/Hydraulics framework.
      public:
        virtual size_t getJacobian(
          agx::Jacobian6DOFElement* G,
          const agxPowerLine::RigidBodyPtrIntHashVector& bodyToIndexTable,
          const agx::ConstraintRigidBodyContainer& bodies,
          const agxPowerLine::PhysicalDimensionPtrIntTable& dimensions,
          const agx::Real dt) override;


        virtual void getJacobian(
          const agxPowerLine::PhysicalDimension* dimension,
          agx::Jacobian6DOFElement& G) const override;

        using agxPowerLine::ElementaryPhysicalDimensionMultiBodyConstraint::getJacobian;

        virtual agx::UInt getViolation(agx::Real* g, agx::UInt row) override;

        void integrateStoredFluid(
            const agx::Jacobian6DOFElement* G,
            const agx::Real solution,
            const agxPowerLine::RigidBodyPtrIntHashVector& bodyToIndexTable,
            const agxPowerLine::PhysicalDimensionPtrIntTable& dimensions,
            const agx::Real dt);



        virtual agx::Bool isImpacting() const override;
        void setIsImpacting(bool impact);

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::ElementaryFlowConstraint);

      protected:
        ElementaryFlowConstraint();
        virtual ~ElementaryFlowConstraint() {}

      private:
        bool m_holonomic;
        agx::Real m_storedFluid;
        bool m_isImpacting;
    };

    typedef agx::ref_ptr<ElementaryFlowConstraint> ElementaryFlowConstraintRef;



    /**
    ConstraintImplementation that contains a single ElementaryFlowConstraint.
    */
    class AGXHYDRAULICS_EXPORT FlowConstraintImplementation : public agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation
    {
      public:
        FlowConstraintImplementation(bool holonomic);

        /**
        Create a FlowConstraintImplementation from a custom ElementaryFlowConstraint.
        Used e.g. by the relief valve connector.
        */
        explicit FlowConstraintImplementation(ElementaryFlowConstraint* elementaryConstraint);

        virtual void postSolveCallback(
            const agx::Jacobian6DOFElement* jacobians, const agx::Real* solution, agx::Real dt) override;

        ElementaryFlowConstraint* getElementaryFlowConstraint();
        const ElementaryFlowConstraint* getElementaryFlowConstraint() const;

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::FlowConstraintImplementation);

      protected:
        FlowConstraintImplementation();
        virtual ~FlowConstraintImplementation() {}

      protected:
        // Owned by the base class.
        ElementaryFlowConstraint* m_flowConstraint;
    };

  }
  /// \endcond

}

#endif
