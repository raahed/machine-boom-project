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


#ifndef AGXHYDRAULICS_RELIEF_VALVE_CONSTRAINTS_H
#define AGXHYDRAULICS_RELIEF_VALVE_CONSTRAINTS_H


#include <agxHydraulics/detail/FlowConstraint.h>
#include <agxPowerLine/PowerLineConstraints.h>
#include <agxHydraulics/export.h>



namespace agxHydraulics
{
  class FlowDimension;

  /// \cond INTERNAL_DOCUMENTATION
  namespace detail
  {
    AGX_DECLARE_POINTER_TYPES(ElementaryLimitedFlowBlockConstraint);

    /**

    Elementary constraint that prevents flow through a FlowUnit. Has a force
    range, called the cracking pressure, so that the constraint will start to
    leak through the FlowUnit once the pressure difference over the FlowUnit is
    greater than the cracking pressure. The constraint will still be active in
    that case, so the pressure loss over the pipe will be large.

    Used in the relief valve to prevent flow to tank via the drain pipes until
    the pressure is high enough.
    */
    class AGXHYDRAULICS_EXPORT ElementaryLimitedFlowBlockConstraint : public agx::ElementaryConstraintN<1>
    {
      public:
        /**
        \param crackingPressure - The pressure difference at which the constraint cannot hold back the fluid.
        \param elementIndex - The dimension index in the RigidBody owned by the blocked FlowUnit this constraint blocks.
        */
        ElementaryLimitedFlowBlockConstraint(agx::Real crackingPressure, agx::UInt8 elementIndex);

        void setCrackingPressure(agx::Real crackingPressure);
        agx::Real getCrackingPressure() const;

        virtual agx::UInt getJacobian(
            agx::Jacobian6DOFElement* G,
            agx::UInt numBlocks,
            agx::UInt row,
            agx::GWriteState::Enum writeState ) override;

        virtual agx::UInt getViolation(agx::Real* g, agx::UInt row) override;

        void setElementIndex(agx::UInt8 elementIndex);

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::ElementaryLimitedFlowBlockConstraint);

      protected:
        ElementaryLimitedFlowBlockConstraint();
        virtual ~ElementaryLimitedFlowBlockConstraint() {}

      private:
        agx::UInt8 m_elementIndex;
    };


    /**
    ConstraintImplementation holding a ElementaryLimitedFlowBlockConstraint.
    */
    class AGXHYDRAULICS_EXPORT LimitedFlowBlockConstraintImplementation : public agx::ConstraintImplementation, public agxStream::Serializable
    {
      public:
        /**
        \param crackingPressure - The pressure differance at which the constraint cannot hold back the fluid.
        \param flowDimension - The FlowDimension that should be blocked.
        */
        LimitedFlowBlockConstraintImplementation(agx::Real crackingPressure, agxHydraulics::FlowDimension* flowDimension);

        void setCrackingPressure(agx::Real crackingPressure);
        agx::Real getCrackingPressure() const;

        void setFlowDimension(agxHydraulics::FlowDimension* flowDim);

        virtual void prepare() override;

        virtual void storeLightData(agxStream::StorageStream& str) const override;
        virtual void restoreLightData(agxStream::StorageStream& str) override;

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::LimitedFlowBlockConstraintImplementation);

      protected:
        LimitedFlowBlockConstraintImplementation();

      protected:
        ElementaryLimitedFlowBlockConstraintRef m_flowBlockConstraint;
        agxHydraulics::FlowDimension* m_flowDimension;
    };


    AGX_DECLARE_POINTER_TYPES(LimitedFlowBlockConstraint);
    AGX_DECLARE_VECTOR_TYPES(LimitedFlowBlockConstraint);


    /**
    Constraint that prevents flow through a FlowUnit until the pressure
    difference over the FlowUnit is larger than the cracking pressure, at which
    point the constraint starts to leak.
    */
    class AGXHYDRAULICS_EXPORT LimitedFlowBlockConstraint : public agx::Constraint
    {
      public:
        /**
        \param crackingPressure - The pressure difference at which the constraint cannot hold back the fluid.
        \param flowDimension - The FlowDimension that should be blocked.
        */
        LimitedFlowBlockConstraint(agx::Real crackingPressure, agxHydraulics::FlowDimension* flowDimension);

        /**
        Set the maximum pressure that the constraint can hold.
        */
        void setCrackingPressure(agx::Real crackingPressure);
        agx::Real getCrackingPressure() const;

        void setFlowDimension(agxHydraulics::FlowDimension* flowDim);

        virtual void render(agxRender::RenderManager* /*canvas*/, float /*scale*/) const override {}

        AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::detail::LimitedFlowBlockConstraint);

      protected:
        LimitedFlowBlockConstraint();
        virtual ~LimitedFlowBlockConstraint();
        virtual int getNumDOF() const override;

      private:
        LimitedFlowBlockConstraintImplementation* m_flowBlockConstraint;
    };
  }
  /// \endcond
}

#endif
