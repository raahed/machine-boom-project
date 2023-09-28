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

#ifndef AGX_STRONGINTERACTIONCONSTRAINT_H
#define AGX_STRONGINTERACTIONCONSTRAINT_H

#include <agx/StrongInteraction.h>
#include <agx/Constraint.h>
#include <agx/agx_valarray_types.h>

namespace agx
{
  AGX_DECLARE_POINTER_TYPES( StrongInteractionConstraint );

  /**
  EXPERIMENTAL
  */
  class AGXPHYSICS_EXPORT StrongInteractionConstraint : public agx::StrongInteraction
  {
    public:
      /**
      Construct given constraint. This constraint will be handled as a 'strong interaction'
      with the jacobian written in the mass matrix.
      */
      StrongInteractionConstraint( agx::Constraint* constraint );

    public:
      /**
      \return first rigid body
      */
      virtual agx::RigidBody* getRigidBody1() const override;

      /**
      \return second rigid body
      */
      virtual agx::RigidBody* getRigidBody2() const override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::StrongInteractionConstraint );

    protected:
      /**
      Default constructor used by serialization.
      */
      StrongInteractionConstraint();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~StrongInteractionConstraint();

      /**
      Render the interaction.
      */
      virtual void render( agxRender::RenderManager* mgr, float scale ) override;

      /**
      Prepares for solve.
      */
      virtual void prepare() override;

      /**
      Write data for dynamic bodies.
      */
      virtual void writeMatrixData( agx::StrongInteraction::MatrixData rb1DiagonalBlock,
                                    agx::StrongInteraction::MatrixData rb2DiagonalBlock,
                                    agx::StrongInteraction::MatrixData offDiagonalBlock ) override;

      /**
      Write data to the right hand side, for dynamic bodies.
      */
      virtual void writeRhs( agx::Physics::RigidBodyPtr rb1,
                             agx::Real* rb1Rhs,
                             agx::Physics::RigidBodyPtr rb2,
                             agx::Real* rb2Rhs,
                             agx::Bool isRestingSolve ) override;

    protected:
      agx::ConstraintRef m_constraint;
      agx::RealValarray  m_rhs;
  };
}

#endif
