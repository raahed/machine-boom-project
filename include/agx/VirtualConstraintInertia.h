/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material without having a written signed agreement with Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGX_VIRTUALCONSTRAINTINERTIA_H
#define AGX_VIRTUALCONSTRAINTINERTIA_H

#include <agx/StrongInteraction.h>
#include <agx/agx_valarray_types.h>
#include <agx/Constraint.h>

namespace agx
{
  AGX_DECLARE_POINTER_TYPES(VirtualConstraintInertia);

  /**
  This class will add a virtual inertia along or around the constraint axis (N-axis), which will affect the resulting
  rotational or translational acceleration on the affected rigid bodies. This virtual inertia is added as a strong interaction that modifies the
  mass and inertia matrix of the coupled bodies right before the solver to achieve this effect. Virtual inertias are applied on
  the rigid bodies in the constraint separately which will affect their movement around and/or along the constraint axis.

  This class can be used to add virtual inertia to a rotational constraint, such as a hinge or add a translational inertia
  along the central axis of a prismatic constraint.
  */
  class AGXPHYSICS_EXPORT VirtualConstraintInertia : public agx::StrongInteraction
  {
  public:
    /**
    Basic constructor that takes a constraint and the virtual inertias that should be applied in the central axis of the constraint.

    \param constraint The constraint that the virtual inertia will be added to.
    \param rigidBody1TranslationInertia The translational inertia/mass that will be for movements along the the constraint axis on on Rigid Body 1.
    \param rigidBody1RotationalInertia The virtual inertia that will be added to movements around the constraint axis on Rigid Body 1.
    \param rigidBody2TranslationInertia The translational inertia/mass that will be for movements along the the constraint axis on Rigid Body 2.
    \param rigidBody2RotationalInertia The virtual inertia that will be added to movements around the constraint axis on Rigid Body 2.
    */
    VirtualConstraintInertia(agx::Constraint* constraint,
      agx::Real rigidBody1TranslationInertia,
      agx::Real rigidBody1RotationalInertia,
      agx::Real rigidBody2TranslationInertia,
      agx::Real rigidBody2RotationalInertia);

  public:
    /**
    Sets the virtual translational inertia that will be applied in the constraint interaction applied on rigid body 1.
    \param translationalInertia - The virtual translational inertia to set in the constraint.
    */
    void setRigidBody1TranslationalInertia(agx::Real translationalInertia);

    /**
    Sets the virtual translational inertia that will be applied in the constraint interaction applied on rigid body 2.
    \param translationalInertia - The virtual translational inertia to set in the constraint.
    */
    void setRigidBody2TranslationalInertia(agx::Real translationalInertia);

    /**
    Sets the virtual rotational inertia that will be applied in the constraint interaction upon rigid body 1
    in the constraint.
    \param rotationalInertia - The virtual rotational inertia to set in the constraint applied on rigid body 1.
    */
    void setRigidBody1RotationalInertia(agx::Real rotationalInertia);

    /**
    Sets the virtual rotational inertia that will be applied in the constraint interaction upon rigid body 2
    in the constraint.
    \param rotationalInertia - The virtual rotational inertia to set in the constraint applied on rigid body 2.
    */
    void setRigidBody2RotationalInertia(agx::Real rotationalInertia);

    /**
    Gets the virtual translational inertia that will be applied in the constraint interaction applied on rigid body 1.
    \return The virtual translational inertia to set in the constraint.
    */
    agx::Real getRigidBody1TranslationalInertia() const;

    /**
    Gets the virtual rotational inertia that will be applied in the constraint interaction upon rigid body 1
    in the constraint.
    \return The virtual rotational inertia to set in the constraint applied on rigid body 1.
    */
    agx::Real getRigidBody1RotationalInertia() const;

    /**
    Gets the virtual translational inertia that will be applied in the constraint interaction applied on rigid body 2.
    \return The virtual translational inertia to set in the constraint applied on rigid body 2.
    */
    agx::Real getRigidBody2TranslationalInertia() const;

    /**
    Gets the virtual rotational inertia that will be applied in the constraint interaction upon rigid body 1
    in the constraint.
    \return The virtual rotational inertia to set in the constraint applied on rigid body 2.
    */
    agx::Real getRigidBody2RotationalInertia() const;

    /**
    \return first rigid body
    */
    virtual agx::RigidBody* getRigidBody1() const override;

    /**
    \return second rigid body
    */
    virtual agx::RigidBody* getRigidBody2() const override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agx::VirtualConstraintInertia);

  protected:
    /**
    Default constructor used by serialization.
    */
    VirtualConstraintInertia();

    /**
    Reference counted object, protected destructor.
    */
    virtual ~VirtualConstraintInertia();

    /**
    Prepares for solve.
    */
    virtual void prepare() override;

    /**
    Write data for dynamic bodies.
    */
    virtual void writeMatrixData(agx::StrongInteraction::MatrixData rb1DiagonalBlock,
                                 agx::StrongInteraction::MatrixData rb2DiagonalBlock,
                                 agx::StrongInteraction::MatrixData offDiagonalBlock) override;

    /**
    Write data to the right hand side, for dynamic bodies.
    */
    virtual void writeRhs(agx::Physics::RigidBodyPtr rb1,
                          agx::Real* rb1Rhs,
                          agx::Physics::RigidBodyPtr rb2,
                          agx::Real* rb2Rhs,
                          agx::Bool isRestingSolve) override;

  protected:
    struct Transformed
    {
      agx::Matrix3x3 addedMassMatrix1;
      agx::Matrix3x3 addedMassMatrix2;
      agx::Matrix3x3 addedInertiaMatrix1;
      agx::Matrix3x3 addedInertiaMatrix2;

      void reset()
      {
        addedMassMatrix1.set( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
        addedMassMatrix2.set( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
        addedInertiaMatrix1.set( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
        addedInertiaMatrix2.set( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
      }
    };

  protected:
    agx::ConstraintRef m_constraint;
    agx::Real          m_rigidBody1TranslationalInertia;
    agx::Real          m_rigidBody1RotationalInertia;
    agx::Real          m_rigidBody2TranslationalInertia;
    agx::Real          m_rigidBody2RotationalInertia;
    Transformed        m_transformed;
  };
}

#endif /*AGX_VIRTUALINERTIAINTERACTION_H*/