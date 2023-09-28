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

#pragma once

#include <agx/ElementaryConstraint.h>
#include <agx/Frame.h>
#include <agx/Interaction.h>
#include <agx/SetVector.h>

#if defined(_MSC_VER)
#pragma warning( push )
#pragma warning( disable : 4589 ) // Disable: 4589: Constructor of abstract class 'agx::FrictionModel' ignores initializer for virtual base class 'agxStream::Serializable'
#endif


namespace agxSDK
{
  class Simulation;
}

namespace agxRender
{
  class RenderManager;
}

namespace agx
{
  // forward declaration
  class Constraint;
  class DynamicsSystem;
  class ConstraintImplementation;
  class DebugRendererCache;
  class Attachment;
  class AttachmentPair;
  class RigidBodyAttachment;
  class RigidBody;
  class RegularizationParameters;

  AGX_DECLARE_POINTER_TYPES( Constraint );
  AGX_DECLARE_VECTOR_TYPES( Constraint );
  typedef agx::SetVector<ref_ptr<Constraint> >   ConstraintRefSetVector;


  /** The base class for a constraint.
  This is an interface class for all constraints.  The virtual
  functions are place holders for the computation of Jacobians and
  constraint value.  In addition, there are interfaces defined here to
  setup the interacting bodies.

  Consider the case of two constrained rigid bodies.  Before the
  constraint is applied, the relative motion is the manifold
  \f$R^3\times SO(3)\f$.  If the constraint is a prismatic joint, the
  remaining manifold of relative motion is just \f$R\f$, the Real line.
  We put a chart on that manifold so we can put a value on the
  extension of the prismatic joint.  On the resulting joint coordinate,
  we might then enforce a secondary constraint such as joint limits, a
  position lock, or a driver that imposes a velocity.

  The constraint processing consists of the following stages:
  -perform necessary geometric computation from rigid bodies' kinematic
  data
  -evaluate the constraint value and basic Jacobians
  -configure dynamical parameters (compliance and damping)
  -compute joint angles according to manifold chart if needed
  -if there are secondary constraints:
  -# compute the value of the constraint
  -# compute the secondary Jacobian by collapsing recursion
  -# add rows to the constraint Jacobian
  -# configure dynamics parameters (compliance and damping)
  -if there are effort constraints
  -# for simple limits, add these to parameters
  -# for additional limits, register these with corresponding Jacobians
  */
  class CALLABLE AGXPHYSICS_EXPORT Constraint : public agx::Interaction
  {
    public:
      /// Specifies in what solvers the constraint will be solved.
      enum SolveType
      {
        DIRECT = 1,                                 /**< Solved only in the DIRECT solver */
        ITERATIVE = (1<<2),                         /**< Solved only in the ITERATIVE solver */
        DIRECT_AND_ITERATIVE = (DIRECT | ITERATIVE) /**< Solved both in the ITERATIVE and the DIRECT solver */
      };

    public:
      /**
      Rebind this constraint, i.e., use current state as the initial configuration.
      \return true if the rebinding is successful - false otherwise
      */
      virtual agx::Bool rebind();

      /// returns the number of bodies involved
      agx::UInt getNumBodies() const;

      /**
      \return rigid body at index i (the bodies are ordered as they were added)
      */
      agx::RigidBody* getBodyAt( agx::UInt i );

      /**
      \return rigid body at index i (the bodies are ordered as they were added)
      */
      const agx::RigidBody* getBodyAt( agx::UInt i ) const;

      /**
      Set the compliance of this constraint for the i:th DOF.
      \param compliance - value of compliance
      \param dof - index of the requested DOF (-1 indicates all)
      */
      virtual void setCompliance( agx::Real compliance, agx::Int dof );

      /**
      Set the compliance of this constraint for all DOFs.
      \param compliance - value of compliance
      */
      void setCompliance( agx::Real compliance );

      /**
      Set the elasticity of this constraint for the i:th DOF. Elasticity is
      1 / compliance.
      \param elasticity - value of elasticity
      \param dof - index of the requested DOF (-1 indicates all)
      */
      virtual void setElasticity(agx::Real elasticity, agx::Int dof);

      /**
      Set the elasticity of this constraint for all DOFs.
      \param elasticity - value of elasticity
      */
      virtual void setElasticity(agx::Real elasticity);


      /**
      Get the compliance for DOF i. Note that indices > number of DOF for this constraint is not defined, so
      compliance of index 0 will be returned.
      \param dof - index of the requested DOF
      \return compliance for the given DOF
      */
      virtual agx::Real getCompliance( agx::UInt dof ) const;

      /**
      Get the elasticity for DOF i. Note that indices > number of DOF for this constraint is not defined, so
      elasticity of index 0 will be returned.
      \param dof - index of the requested DOF
      \return elasticity for the given DOF
      */
      virtual agx::Real getElasticity(agx::UInt dof) const;

      /**
      Set the damping of this constraint for the i:th DOF
      \param damping - value of damping
      \param dof - index of the requested DOF (-1 indicates all).
      */
      virtual void setDamping( agx::Real damping, agx::Int dof );

      /**
      Set the damping of this constraint for all DOFs.
      \param damping - value of damping
      */
      void setDamping( agx::Real damping );

      /**
      Get the damping for DOF \p dof. Note that indices > number of DOF for this constraint is not defined, so
      damping of index 0 will be returned.
      \param dof - index of the requested DOF
      \return damping for the given DOF
      */
      virtual Real getDamping( agx::UInt dof ) const;

      /**
      Assign force range for all DOF of the elementary constraints.
      \note Using this method will NOT change the force range of the controllers (secondary constraints),
            if present. Use their specific methods instead, e.g., constraint->getMotor1D()->setForceRange( fRange ).
      \param forceRange - value of force range for all elementary DOFs
      */
      void setForceRange( agx::RangeReal forceRange );

      /**
      Assign force range, of an elementary constraint, for a given DOF.
      \note For controllers (secondary constraints), use their specific methods, e.g., constraint->getMotor1D()->setForceRange( fRange ).
      \param forceRange - value of force range for the given DOF
      \param dof - index of the DOF (-1 indicates all)
      */
      void setForceRange( agx::RangeReal forceRange, agx::Int dof );

      /**
      Assign force range, of an elementary constraint, for a given DOF.
      \note For controllers (secondary constraints), use their specific methods, e.g., constraint->getMotor1D()->setForceRange( fLower, fUpper ).
      \param lower - lower value of force range for the given DOF
      \param upper - lower value of force range for the given DOF
      \param dof - index of the DOF (-1 indicates all)
      */
      void setForceRange( agx::Real lower, agx::Real upper, agx::Int dof );

      /**
      Get the force range for DOF \p dof.
      \note Indices > number of DOF for this constraint is not defined, an infinite force range will be returned.
      \note For controllers (secondary constraints), use their specific method, e.g., constraint->getMotor1D()->getForceRange().
      \param dof - index of the requested DOF
      */
      agx::RangeReal getForceRange( agx::UInt dof = 0 ) const;

      /**
      Consider using \p getLastForce instead. This method returns the magnitude of the force in
      a given degree of freedom (DOF). Enabled controllers in the given DOF will NOT be included,
      agx::ElementaryConstraint::getCurrentForce has to be used for that.
      \return the current force magnitude (i.e., the force magnitude last time step) in given DOF >= 0 && <= NUM_DOFS
      \sa setEnableComputeForces and getLastForce
      */
      agx::Real getCurrentForce( agx::UInt dof ) const;

      /**
      \return a pointer to the regularization parameter object of this constraint for the i:th DOF
      */
      virtual agx::RegularizationParameters* getRegularizationParameters( agx::UInt i );

      /**
      \return a pointer to the regularization parameter object of this constraint for the i:th DOF
      */
      virtual const agx::RegularizationParameters* getRegularizationParameters( agx::UInt i ) const;

      /**
      \return true if this constraints is valid
      It is possible for a constraint to declare itself invalid during a simulation.
      This may happen if for example a body that is included in this constraint is
      removed from the simulation.
      */
      agx::Bool getValid() const;

      /// Inherited method of how to render this constraint into DebugRenderer
      virtual void render( class agxRender::RenderManager* mgr, float scale ) const = 0;

      /// \return a pointer to the internal constraint implementation
      agx::ConstraintImplementation* getRep();

      #ifndef SWIG
      /// \return a pointer to the internal constraint implementation
      const agx::ConstraintImplementation* getRep() const;
      #endif

      /// \return the entity id of the constraint
      agx::UInt32 getEntityId() const;

      /**
      Enable/disable a constraint.
      \param enable - If true the constraint will be enabled.
      */
      virtual void setEnable( agx::Bool enable );

      /**
      \return true if the constraint is enabled.
      */
      agx::Bool getEnable() const;

      /**
      \return true if the constraint is enabled.
      */
      agx::Bool isEnabled() const;

      /**
      \return the number of DOF for this constraint, not including secondary constraints. -1 if num DOF is undefined for the constraint.
      */
      virtual int getNumDOF() const = 0;

      /**
      Specify the solve type for this constraint. Valid is DIRECT (default for non-iterative solvers), ITERATIVE or DIRECT_AND_ITERATIVE
      where DIRECT_AND_ITERATIVE means that this constraint will be solved both direct and iterative.
      \param solveType - new solve type for this constraint
      \note Solve type is ignored by iterative solvers.
      */
      void setSolveType( agx::Constraint::SolveType solveType );

      /**
      \return the current solve type for this constraint
      \note Solve type is ignored by iterative solvers.
      */
      agx::Constraint::SolveType getSolveType() const;

      /**
      \return the attachment frame for \p rb if present - otherwise 0
      */
      agx::RigidBodyAttachment* getAttachment( const agx::RigidBody* rb ) const;

      /**
      \return the attachment frame for body with index \p i (call equal to constraint->getAttachment( constraint->getBodyAt( i ) ))
      */
      agx::Attachment* getAttachment( agx::UInt i ) const;

      /**
      \return the attachment pair if this constraint supports it - otherwise 0
      */
      agx::AttachmentPair* getAttachmentPair() const;

      /**
      Enable/disable debug rendering of this constraint.
      */
      void setEnableDebugRendering( agx::Bool enable );

      /**
      \return true if this constraints debug rendering is enabled - otherwise false
      */
      bool getEnableDebugRendering() const;

      /**
      Enable (or disable) computation of the forces applied to the dynamic bodies in this constraint.
      Matrix-vector operation to compute the forces after solve.
      \sa getLastForce
      \param enable - true to enable, false to disable
      */
      void setEnableComputeForces( agx::Bool enable );

      /**
      \return true if this constraint has been enabled to compute the forces applied to its bodies - otherwise false
      */
      agx::Bool getEnableComputeForces() const;

      /**
      If 'compute forces' is enabled, returns the last force and torque applied by this constraint on
      the body with \p bodyIndex. The force is given in world coordinates and is the one applied
      at the anchor position of this constraint.

      The result includes force and torque from this constraint including all enabled
      controllers (e.g., motors, locks, ranges).
      \sa setEnableComputeForces
      \param bodyIndex      - index of body, if number of bodies = 1 and bodyIndex = 1, the force and
                              torque applied to "the world body" is returned
      \param retForce       - the force applied by this constraint on body at \p bodyIndex last solve
      \param retTorque      - the torque applied by this constraint on body at \p bodyIndex last solve
      \param giveForceAtCm  - this parameter affects the resulting torque. The default behavior (giveForceAtCm = false)
                              calculates the force applied by this constraint at the anchor position.
                              Letting giveForceAtCm = true, the force will be applied at center of mass,
                              affecting the torque as \f$ T_{new} = T - r \times F \f$ where r is the vector
                              from the anchor point to the center of mass of the body.
      \return true if resulting force and torque was written to \p retForce and \p retTorque - otherwise false
      */
      agx::Bool getLastForce( agx::UInt bodyIndex, agx::Vec3& retForce, agx::Vec3& retTorque, agx::Bool giveForceAtCm = false ) const;

      /**
      If 'compute forces' is enabled, returns the last force and torque applied by this constraint on
      the body \p rb. The force is given in world coordinates and is the one applied at the anchor
      position of this constraint.

      The result includes force and torque from this constraint including all enabled
      controllers (e.g., motors, locks, ranges).
      \sa setEnableComputeForces
      \param rb             - body in this constraint. If this constraint is attached in world, nullptr can be used.
      \param retForce       - the force applied by this constraint on body \p rb at the last solve
      \param retTorque      - the torque applied by this constraint on body \p rb at the last solve
      \param giveForceAtCm  - this parameter affects the resulting torque. The default behavior (giveForceAtCm = false)
                              calculates the force applied by this constraint at the anchor position.
                              Letting giveForceAtCm = true, the force will be applied at center of mass,
                              affecting the torque as \f$T_{new} = T - r \times F\f$ where r is the vector
                              from the anchor point to the center of mass of the body.
      \return true if resulting force and torque was written to \p retForce and \p retTorque - otherwise false
      */
      agx::Bool getLastForce( const agx::RigidBody* rb, agx::Vec3& retForce, agx::Vec3& retTorque, agx::Bool giveForceAtCm = false ) const;

      /**
      If 'compute forces' is enabled, returns the last force and torque applied by this constraint on
      the body with \p bodyIndex. The force is given in the frame of the constraint.

      The result includes force and torque from this constraint including all enabled
      controllers (e.g., motors, locks, ranges).
      \sa setEnableComputeForces
      \param bodyIndex      - index of body, if number of bodies = 1 and bodyIndex = 1, the force and
      torque applied to "the world body" is returned
      \param retForce       - the force applied by this constraint on body at \p bodyIndex last solve
      \param retTorque      - the torque applied by this constraint on body at \p bodyIndex last solve
      \param giveForceAtCm  - this parameter affects the resulting torque. The default behavior (giveForceAtCm = false)
                              calculates the force applied by this constraint at the anchor position.
                              Letting giveForceAtCm = true, the force will be applied at center of mass,
                              affecting the torque as \f$T_{new} = T - r \times F\f$ where r is the vector
                              from the anchor point to the center of mass of the body.
      \return true if resulting force and torque was written to \p retForce and \p retTorque - otherwise false
      */
      agx::Bool getLastLocalForce( agx::UInt bodyIndex, agx::Vec3& retForce, agx::Vec3& retTorque, agx::Bool giveForceAtCm = false ) const;

      /**
      If 'compute forces' is enabled, returns the last force and torque applied by this constraint on
      the body \p rb. The force is given in the frame of the constraint.

      The result includes force and torque from this constraint including all enabled
      controllers (e.g., motors, locks, ranges).
      \sa setEnableComputeForces
      \param rb             - body in this constraint. If this constraint is attached in world, nullptr can be used.
      \param retForce       - the force applied by this constraint on body \p rb at the last solve
      \param retTorque      - the torque applied by this constraint on body \p rb at the last solve
      \param giveForceAtCm  - this parameter affects the resulting torque. The default behavior (giveForceAtCm = false)
                              calculates the force applied by this constraint at the anchor position.
                              Letting giveForceAtCm = true, the force will be applied at center of mass,
                              affecting the torque as \f$T_{new} = T - r \times F\f$ where r is the vector
                              from the anchor point to the center of mass of the body.
      \return true if resulting force and torque was written to \p retForce and \p retTorque - otherwise false
      */
      agx::Bool getLastLocalForce( const agx::RigidBody* rb, agx::Vec3& retForce, agx::Vec3& retTorque, agx::Bool giveForceAtCm = false ) const;

      /**
      Pass \p true to enable linearization of constraint compliance. This makes
      constraint violation a linear function of the force or torque used to
      violate the constraint also for the elementary constraint types for which
      this isn't already the case, such as QuatLock. The effect of this is that
      the parameters sent to the solver by the constraint are tweaked to make
      the constrain more closely follow Hook's law.

      Setting this to true has no effect on most constraints. It does have an
      effect on the rotational part of AngularLockJoint, LockJoint, and
      Prismatic.

      When false, the default, the resulting violation of any QuatLock in the
      constraint is linear in quaternion space instead of degrees and radians.
      */
      void setEnableLinearization(bool enable);

      /**
      \return true if constraint compliance linearization is enabled for this
      constraint. False otherwise.
      */
      bool getEnableLinearization() const;

      /**
      Add elementary constraint (like Spherical, Dot1, Dot2 etc) given name.
      General for elementary constraints is that their input is based on
      body relative data (far from general).
      \return true if this constraint supports the elementary constraint and the elementary constraint will be used - otherwise false
      */
      agx::Bool addElementaryConstraint( const agx::Name& name, agx::ElementaryConstraint* elementaryConstraint );

      /**
      Add secondary constraint (like motor, range and/or lock etc) given name.
      General for secondary constraints is that their input is based on the
      current constraint angle.
      \return true if this constraint supports the secondary constraint and the secondary constraint will be used - otherwise false
      */
      agx::Bool addSecondaryConstraint( const agx::Name& name, agx::ElementaryConstraint* secondaryConstraint );

      /**
      Remove elementary constraint.
      \param elementaryConstraint - elementary constraint to remove
      \return true if removed
      */
      agx::Bool removeElementaryConstraint( agx::ElementaryConstraint* elementaryConstraint );

      /**
      Remove secondary constraint.
      \param secondaryConstraint - secondary constraint to remove
      \return true if removed
      */
      agx::Bool removeSecondaryConstraint( agx::ElementaryConstraint* secondaryConstraint );

      /**
      Remove elementary constraint.
      \param name - name of the elementary constraint to remove
      \return true if removed
      */
      agx::Bool removeElementaryConstraint( const agx::Name& name );

      /**
      Remove secondary constraint.
      \param name - name of the secondary constraint to remove
      \return true if removed
      */
      agx::Bool removeSecondaryConstraint( const agx::Name& name );

      /**
      \param index - index of the elementary constraint
      \return elementary constraint given index
      */
      agx::ElementaryConstraint* getElementaryConstraint( const agx::UInt index ) const;

      /**
      Find elementary constraint given name.
      \param name - name of the elementary constraint
      \return elementary constraint given name
      */
      agx::ElementaryConstraint* getElementaryConstraintGivenName( const agx::Name& name ) const;

      /**
      \param index - index of the secondary constraint
      \return secondary constraint given index
      */
      agx::ElementaryConstraint* getSecondaryConstraint( const agx::UInt index ) const;

      /**
      Find secondary constraint given name.
      \param name - name of the elementary constraint
      \return elementary constraint given name
      */
      agx::ElementaryConstraint* getSecondaryConstraintGivenName( const agx::Name& name ) const;

      /**
      \return the number of elementary constraints
      */
      agx::UInt getNumElementaryConstraints() const;

      /**
      \return the number of secondary constraints
      */
      agx::UInt getNumSecondaryConstraints() const;

      /**
      Calculates the current number of active rows, including both elementary and secondary
      constraints. The result can be used to allocate temporary buffers for additional calls
      to solver related methods, e.g., getViolation.
      \return the (current) total number of active rows in this constraint
      */
      agx::UInt calculateNumActiveRows() const;

      /**
      Given a point and an axis in world, this function calculates each local attachment frame for one or two bodies.
      The first body, rb1, has to be valid. If the second body, rb2, is zero the second frame, rb2Frame, will be given in world frame.
      \param worldPoint - point in world
      \param worldAxis - axis in world (for example prismatic- or hinge axis)
      \param rb1 - first rigid body, invalid if null
      \param rb1Frame - frame to calculate for rb1, invalid if null
      \param rb2 - second rigid body, if null rb2Frame will be given in world
      \param rb2Frame - frame to calculate for rb2, invalid if null
      \return true if the calculations succeeded - otherwise false (if false, rb1Frame and rb2Frame are not touched)
      */
      static agx::Bool calculateFramesFromWorld( agx::Vec3 worldPoint, agx::Vec3 worldAxis, const agx::RigidBody* rb1, agx::Frame* rb1Frame, const agx::RigidBody* rb2, agx::Frame* rb2Frame );

      /**
      Given a point and two axes in world, this function calculates each local attachment frame for one or two bodies.
      The frames' z-axis will be aligned with the worldAxis and its y-axis will be normal to the worldAxis-secondWorldAxis plane.
      The first body, rb1, has to be valid. If the second body, rb2, is zero the second frame, rb2Frame, will be given in world frame.
      \param worldPoint - point in world
      \param worldAxis - axis in world (for example prismatic- or hinge axis)
      \param secondWorldAxis - axis in world (must be non-parallel to worldAxis)
      \param rb1 - first rigid body, invalid if null
      \param rb1Frame - frame to calculate for rb1, invalid if null
      \param rb2 - second rigid body, if null rb2Frame will be given in world
      \param rb2Frame - frame to calculate for rb2, invalid if null
      \return true if the calculations succeeded - otherwise false (if false, rb1Frame and rb2Frame are not touched)
      */
      static agx::Bool calculateFramesFromWorld(agx::Vec3 worldPoint, agx::Vec3 worldAxis, agx::Vec3 secondWorldAxis, const agx::RigidBody* rb1, agx::Frame* rb1Frame, const agx::RigidBody* rb2, agx::Frame* rb2Frame);

      /**
      Calculates the constraint attachment frames given point and axis in body coordinates of \p body.
      \param bodyPoint - point in body coordinates of reference rigid body "body"
      \param bodyAxis - axis in body coordinates of reference rigid body "body"
      \param body - reference body, invalid if null
      \param bodyFrame - reference body attachment frame, invalid if null
      \param otherBody - other rigid body, if null otherFrame will be calculated in world frame
      \param otherFrame - other frame, invalid if null
      \return true if the calculations succeeded - otherwise false
      */
      static agx::Bool calculateFramesFromBody( agx::Vec3 bodyPoint, agx::Vec3 bodyAxis, const agx::RigidBody* body, agx::Frame* bodyFrame, const agx::RigidBody* otherBody, agx::Frame* otherFrame );

      /**
      Calculates the constraint attachment frames given point and axes in body coordinates of \p body.
      The frames' z-axis will be aligned with the bodyAxis and its y-axis will be normal to the bodyAxis-secondBodyAxis plane.
      \param bodyPoint - point in body coordinates of reference rigid body "body"
      \param bodyAxis - axis in body coordinates of reference rigid body "body"
      \param secondBodyAxis - non-parallel axis to bodyAxis in body coordinates of reference rigid body "body"
      \param body - reference body, invalid if null
      \param bodyFrame - reference body attachment frame, invalid if null
      \param otherBody - other rigid body, if null otherFrame will be calculated in world frame
      \param otherFrame - other frame, invalid if null
      \return true if the calculations succeeded - otherwise false
      */
      static agx::Bool calculateFramesFromBody(agx::Vec3 bodyPoint, agx::Vec3 bodyAxis, agx::Vec3 secondBodyAxis, const agx::RigidBody* body, agx::Frame* bodyFrame, const agx::RigidBody* otherBody, agx::Frame* otherFrame);

      /**
      Creates a constraint given a point and an axis in world frame. Both relative attachment frames
      will coincide at the given world point with their z-axes pointing in the given world axis
      direction.
      \param worldPoint - constraint center point given in the world frame
      \param worldAxis - constraint axis given in world frame
      \param rb1 - first rigid body (not defined if null)
      \param rb2 - second rigid body, if null, \p rb1 will be attached in world
      \return the constraint of type T
      */
      template< typename T >
      static agx::ref_ptr< T > createFromWorld( agx::Vec3 worldPoint, agx::Vec3 worldAxis, agx::RigidBody* rb1, agx::RigidBody* rb2 = nullptr );

      /**
      Creates a constraint given a point and an axis in \p rb1 model frame. Seen from body one frame - both
      relative attachment frames will coincide at the given point with their z-axes pointing in the axis direction.
      \param bodyPoint - constraint center point given in first body's frame
      \param bodyAxis - constraint axis given in first body's frame
      \param rb1 - first rigid body (not defined if null)
      \param rb2 - second rigid body, if null, \p rb1 will be attached in world
      \return the constraint of type T
      */
      template< typename T >
      static agx::ref_ptr< T > createFromBody( agx::Vec3 bodyPoint, agx::Vec3 bodyAxis, agx::RigidBody* rb1, agx::RigidBody* rb2 = nullptr );

      /// Try to do dynamic_cast on the constraint and return a pointer to the casted constraint
      template< typename T >
      T *as();

      /// Try to do dynamic_cast on the constraint and return a pointer to the casted constraint
      template< typename T >
      const T *as() const;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agx::Constraint );

      void store( agxStream::OutputArchive& out ) const override;
      void restore( agxStream::InputArchive& in ) override;

      virtual void storeLightData( agxStream::StorageStream& str ) const override;
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      Constraint();
      virtual ~Constraint();

      friend class DynamicsSystem;
      virtual void setSystem( agx::DynamicsSystem* ) {}

      friend class agxSDK::Simulation;
      virtual void setSimulation( agxSDK::Simulation* simulation ) override;
      /*
      Called when this Constraint is added to a simulation
      (given that it not already is in the simulation).
      */
      virtual void addNotification() override;
      virtual void removeNotification() override;

#ifndef SWIG
      virtual void preSystemCallback( agx::DynamicsSystem* dynamicsSystem ) override;
      virtual void postSystemCallback( agx::DynamicsSystem* dynamicsSystem ) override;
#endif

      void setRep( agx::ConstraintImplementation* _rep );

    protected:
      agx::ConstraintImplementation* m_implementation;

    private:
      DOXYGEN_START_INTERNAL_BLOCK()
      friend class InternalData;
      /**
      \return internal data for merge split
      */
      agx::Referenced* getInternalData() const;

      /**
      Assign merge split data for this body.
      */
      void setInternalData( agx::Referenced* data );
      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      agx::Bool m_debugRenderingEnable;
  };

  // Backward compatibility. In export to Lua use:
  // class Motor1D : public agx::TargetSpeedController {};
  // class Lock1D  : public agx::LockController {};
  // class Range1D : public agx::RangeController {};
  // class Screw1D : public agx::ScrewController {};
  typedef agx::TargetSpeedController Motor1D;
  typedef agx::LockController        Lock1D;
  typedef agx::RangeController       Range1D;
  typedef agx::ScrewController       Screw1D;

  AGX_DECLARE_POINTER_TYPES(Constraint1DOF);
  /**
  Specialization for constraints that have only one degree of freedom
  such as Hinge and Prismatic.
  */
  class CALLABLE AGXPHYSICS_EXPORT Constraint1DOF : public agx::Constraint
  {
    public:
      /**
      Utility method for not so type safe languages.
      */
      static agx::Constraint1DOF* safeCast( const agx::Constraint* constraint );

    public:
      /**
      \return the secondary constraint of type Lock1D
      */
      agx::Lock1D* getLock1D();

      /**
      \return the secondary constraint of type Lock1D
      */
      const agx::Lock1D* getLock1D() const;

      /**
      \return the secondary constraint of type Motor1D
      */
      agx::Motor1D* getMotor1D();

      /**
      \return the secondary constraint of type Motor1D
      */
      const agx::Motor1D* getMotor1D() const;

      /**
      \return the electric motor controller
      */
      agx::ElectricMotorController* getElectricMotorController();

      /**
      \return the electric motor controller
      */
      const agx::ElectricMotorController* getElectricMotorController() const;

      /**
      \return the secondary constraint of type Range1D
      */
      agx::Range1D* getRange1D();

      /**
      \return the secondary constraint of type Range1D
      */
      const agx::Range1D* getRange1D() const;

      /**
      \return the friction controller
      */
      agx::FrictionController* getFrictionController();

      /**
      \return the friction controller
      */
      const agx::FrictionController* getFrictionController() const;

      /**
      This method return the current angle for the 1D constraint.
      For a prismatic, this is the position of the constraint, if it is a hinge, then it
      is the angle of rotation including the winding.
      \return the current joint angle including the winding (if available).
      */
      agx::Real getAngle() const;

      /**
      This methods return the current speed for the 1D constraint.
      For a prismatic this is the linear velocity along the axis, if it is a hinge, then it is
      the current angular rotation around the hinge axis.
      \return the current speed for the 1D constraint.
      */
      agx::Real getCurrentSpeed() const;

    protected:
      Constraint1DOF();
      virtual ~Constraint1DOF();
  };

  AGX_DECLARE_POINTER_TYPES(Constraint2DOF);
  /**
  Specialization for constraints that have only one degree of freedom
  such as Hinge and Prismatic.
  */
  class CALLABLE AGXPHYSICS_EXPORT Constraint2DOF : public agx::Constraint
  {
    public:
      /**
      Utility method for not so type safe languages.
      */
      static agx::Constraint2DOF* safeCast( const agx::Constraint* constraint );

    public:
      /// Specifies a selected degree of freedom
      enum DOF
      {
        FIRST = 0,
        SECOND
      };

      /**
      \return the secondary constraint of type Lock1D for the given degree of freedom
      */
      agx::Lock1D* getLock1D( agx::Constraint2DOF::DOF dof );

      /**
      \return the secondary constraint of type Lock1D for the given degree of freedom
      */
      const agx::Lock1D* getLock1D( agx::Constraint2DOF::DOF dof ) const;

      /**
      \return the secondary constraint of type Motor1D for the given degree of freedom
      */
      agx::Motor1D* getMotor1D( agx::Constraint2DOF::DOF dof );

      /**
      \return the secondary constraint of type Motor1D for the given degree of freedom
      */
      const agx::Motor1D* getMotor1D( agx::Constraint2DOF::DOF dof ) const;

      /**
      \return the electric motor controller
      */
      agx::ElectricMotorController* getElectricMotorController( agx::Constraint2DOF::DOF dof );

      /**
      \return the electric motor controller
      */
      const agx::ElectricMotorController* getElectricMotorController( agx::Constraint2DOF::DOF dof ) const;

      /**
      \return the secondary constraint of type Range1D for the given degree of freedom
      */
      agx::Range1D* getRange1D( agx::Constraint2DOF::DOF dof );

      /**
      \return the secondary constraint of type Range1D for the given degree of freedom
      */
      const agx::Range1D* getRange1D( agx::Constraint2DOF::DOF dof ) const;

      /**
      \return the friction controller for the given degree of freedom
      */
      agx::FrictionController* getFrictionController( agx::Constraint2DOF::DOF dof );

      /**
      \return the friction controller for the given degree of freedom
      */
      const agx::FrictionController* getFrictionController( agx::Constraint2DOF::DOF dof ) const;

      /**
      Screw controller connects rotation and translation.
      \return the secondary constraint of type Screw1D
      */
      agx::Screw1D* getScrew1D();

      /**
      Screw controller connects rotation and translation.
      \return the secondary constraint of type Screw1D
      */
      const agx::Screw1D* getScrew1D() const;

      /**
      \return the angle given degree of freedom
      */
      agx::Real getAngle( agx::Constraint2DOF::DOF dof ) const;

      /**
      \return the current speed given degree of freedom
      */
      agx::Real getCurrentSpeed( agx::Constraint2DOF::DOF dof ) const;

    protected:
      Constraint2DOF();
      virtual ~Constraint2DOF();
  };

  DOXYGEN_START_INTERNAL_BLOCK()
  class AGXPHYSICS_EXPORT ConstraintFrame
  {
    public:
      ConstraintFrame();
      virtual ~ConstraintFrame();

      Vec3& operator [] ( const unsigned int& i );
      const Vec3& operator [] ( const unsigned int& i ) const;

      void setTranslate( const Vec3& translate );
      const Vec3& getTranslate() const;

    protected:
      Vec3 m_data[ 4 ];
  };
  DOXYGEN_END_INTERNAL_BLOCK()

  inline ConstraintImplementation* Constraint::getRep()
  {
    return m_implementation;
  }

  inline const ConstraintImplementation* Constraint::getRep() const
  {
    return m_implementation;
  }

  template< typename T >
  ref_ptr< T > Constraint::createFromWorld( Vec3 worldPoint, Vec3 worldAxis, RigidBody* rb1, RigidBody* rb2 /* = nullptr */ )
  {
    FrameRef f1 = new Frame();
    FrameRef f2 = new Frame();
    Constraint::calculateFramesFromWorld( worldPoint, worldAxis, rb1, f1, rb2, f2 );
    ref_ptr< T > constraint = new T( rb1, f1, rb2, f2 );
    return constraint;
  }

  template< typename T >
  ref_ptr< T > Constraint::createFromBody( Vec3 bodyPoint, Vec3 bodyAxis, RigidBody* rb1, RigidBody* rb2 /* = nullptr */ )
  {
    FrameRef f1 = new Frame();
    FrameRef f2 = new Frame();
    Constraint::calculateFramesFromBody( bodyPoint, bodyAxis, rb1, f1, rb2, f2 );
    ref_ptr< T > constraint = new T( rb1, f1, rb2, f2 );
    return constraint;
  }

  /// Try to do dynamic_cast on the constraint and return a pointer to the casted constraint
  template< typename T >
  T *Constraint::as()
  {
    return dynamic_cast<T*>(this);
  }

  /// Try to do dynamic_cast on the constraint and return a pointer to the casted constraint
  template< typename T >
  const T *Constraint::as() const
  {
    return dynamic_cast<const T*>(this);
  }

} //namespace agx

#if defined(_MSC_VER)
#pragma warning( pop )
#endif

