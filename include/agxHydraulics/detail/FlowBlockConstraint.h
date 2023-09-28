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

#ifndef AGXHYDRAULICS_FLOW_BLOCK_CONSTRAINT_H
#define AGXHYDRAULICS_FLOW_BLOCK_CONSTRAINT_H

#include <agx/Constraint.h>
#include <agx/ConstraintImplementation.h>
#include <agx/ElementaryConstraint.h>

#include <agxHydraulics/FlowUnit.h>
#include <agxHydraulics/FlowDirection.h>

namespace agxHydraulics
{

  /// \cond INTERNAL_DOCUMENTATION
  /**
  Constraint that prevents flow in either direction, or both, or none. Most of
  the logic is handled by the FlowBlockConstraintImplementation.
  */
  class AGXHYDRAULICS_EXPORT ElementaryFlowBlockConstraint : public agx::ElementaryConstraintN<1>
  {
    public:
      /**
      \param allowedFlowDirection - The direction of flow that the flow block
                                    constraint should allow. Flow in the opposite
                                    direction is blocked.
      \param elementIndex - The index in the rigid body's dimensions is owned by the blocked FlowDimension.
      */
      ElementaryFlowBlockConstraint(
          agxHydraulics::FlowDirection::Literal allowedFlowDirection,
          agx::UInt8 elementIndex);

      /**
      \return The flow direction that the constraint will not block.
      */
      agxHydraulics::FlowDirection::Literal getAllowedFlowDirection() const;

      /**
      \param allowedFlowDireciton - The flow direction that the constraint will not block.
      */
      void setAllowedFlowDirection(agxHydraulics::FlowDirection::Literal allowedFlowDirection);

      void setElementIndex(agx::UInt8 elementIndex);

      virtual agx::UInt getJacobian(
             agx::Jacobian6DOFElement* G, agx::UInt numBlocks,
             agx::UInt row, agx::GWriteState::Enum writeState) override;

      /**
      Always sets the violation to zero.
      */
      virtual agx::UInt getViolation(agx::Real* g, agx::UInt row) override;

      /**
      Always sets the constraint velocity to zero.
      */
      virtual agx::UInt getVelocity(agx::Real* v, agx::UInt row) const override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::ElementaryFlowBlockConstraint);

    protected:
      ElementaryFlowBlockConstraint();
      virtual ~ElementaryFlowBlockConstraint() {}

    private:
      agx::UInt8 m_elementIndex;
      agxHydraulics::FlowDirection::Literal m_allowedFlowDirection;
  };


  /**
  Non-PowerLine ConstraintImplementation holding an ElementaryFlowBlockConstraint.
  */
  class AGXHYDRAULICS_EXPORT FlowBlockConstraintImplementation : public agx::ConstraintImplementation, public agxStream::Serializable
  {
    public:
      /**
      \param allowedFlowDirection - The flow direction that the constraint should not block.
      \param flowUnit - The FlowUnit that should have all other flow directions blocked.
      */
      FlowBlockConstraintImplementation(
          agxHydraulics::FlowDirection::Literal allowedFlowDirection,
          agxHydraulics::FlowUnit* flowUnit);

      /**
      \return The flow direction that the constraint will not block.
      */
      agxHydraulics::FlowDirection::Literal getAllowedFlowDirection() const;

      /**
      \param allowedFlowDireciton - The flow direction that the constraint will not block.
      */
      void setAllowedFlowDirection(agxHydraulics::FlowDirection::Literal allowedFlowDirection);

    // Methods called by the rest of the PowerLine/Hydraulics framework.
    public:
      ElementaryFlowBlockConstraint* getElementaryFlowBlockConstraint();

      virtual void prepare() override;

      virtual void storeLightData(agxStream::StorageStream& str) const override;
      virtual void restoreLightData(agxStream::StorageStream& str) override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::FlowBlockConstraintImplementation);

    protected:
      FlowBlockConstraintImplementation();

    private:
      ~FlowBlockConstraintImplementation();
      ElementaryFlowBlockConstraint* m_flowBlockConstraint;
      agxHydraulics::FlowDimension* m_flowDimension;
  };


  AGX_DECLARE_POINTER_TYPES(FlowBlockConstraint);

  /**
  A constraint that allows flow in one direction only.
  */
  class AGXHYDRAULICS_EXPORT FlowBlockConstraint : public agx::Constraint
  {
    public:
      /**
      \param allowedFlowDirection - The flow direction that the constraint should not block.
      \param flowUnit - The FlowUnit that should have all other flow directions blocked.
      */
      FlowBlockConstraint(agxHydraulics::FlowDirection::Literal allowedFlowDirection, agxHydraulics::FlowUnit* flowUnit);

      /**
      \return The flow direction that the constraint will not block.
      */
      agxHydraulics::FlowDirection::Literal getAllowedFlowDirection() const;

      /**
      \param allowedFlowDirection - The flow direction that the constraint should not block.
      */
      void setAllowedFlowDirection(agxHydraulics::FlowDirection::Literal allowedFlowDirection);

      virtual void render(agxRender::RenderManager* /*canvas*/, float /*scale*/) const override {}

      AGXSTREAM_DECLARE_SERIALIZABLE(agxHydraulics::FlowBlockConstraint);

    protected:
      FlowBlockConstraint();
      virtual ~FlowBlockConstraint();
      virtual int getNumDOF() const override;

    private:
      FlowBlockConstraintImplementation* m_flowBlockConstraint;
  };
  /// \endcond

}


#endif
