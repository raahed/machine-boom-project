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

#ifndef AGX_HINGE_H
#define AGX_HINGE_H

#include <agx/Constraint.h>

namespace agx
{
  /**
  Specialization for the constraint frame class that add convenient accessors/mutator methods
  for manipulation of the geometry of the frame for a Hinge.
  */
  class CALLABLE AGXPHYSICS_EXPORT HingeFrame : public agx::ConstraintFrame
  {
    public:
      /**
      Default constructor: rotational axis is along Z.
      */
      HingeFrame();

      /**
      \param center - Hinge center position in world coordinates
      \param axis - Hinge axis in world coordinates
      */
      HingeFrame( const Vec3& center, const Vec3& axis );

      ~HingeFrame() {}

      /**
      Sets the axis of the hinge in world coordinates.
      \param axis - Hinge axis in world coordinates
      */
      void setAxis( const Vec3& axis );

      /**
      Sets the center of the hinge in world coordinates.
      \param center - Hinge center position in world coordinates
      */
      void setCenter( const Vec3& center );

      /**
      Sets all parameters needed to create a hinge.
      \param center - Hinge center position in world coordinates
      \param axis - Hinge axis in world coordinates
      */
      void setHinge( const Vec3& center, const Vec3& axis );

      /**
      Get the hinge axis
      \return Hinge axis
      */
      const Vec3& getAxis() const;
  };

  /**
  The hinge constraint between two rigid bodies or one rigid body
  and the world.

  The hinge constraint maintains a constant rotational axis between
  two rigid bodies and allows rotation about this axis.  By default,
  there is no control on the relative angle between the two bodies and
  in fact, the relative rotational angle is not computed.  By using
  the DOFController object and setting flags and other parameters, the
  relative angle can be computed.  It can also be limited to a given
  range, locked at a given value, or motorized using a velocity or a
  position constraint.  When a joint limit is reached, an inequality
  constraint is added to the system to enforce the joint limit.  If
  the motor or lock control is already turned ON, both constraints can
  be active, and it is up to the solver to compute the correct
  solution.

  When setting up the hinge constraint, the user can simply pass a
  HingeFrame object to the constraint.  The "Z" axis of the frame will then
  be used as the hinge axis.  This is not to say that the constraint
  will keep that axis in world coordinates, only that at the instant
  following configuration, the hinge axis will be coincident with the
  "Z" axis of the given frame.  Correspondingly, by default, the joint
  angle will be computed in the "XY" plane of the given frame so that
  zero angle corresponds to the hinge frame aligned along the original
  "X" axis.

  In addition, the configuration of the hinge can be done by setting
  the attachment frames of each of the bodies.  When doing that, the
  constraint will then operate to make the two "Z" axes parallel, and
  the origins of the two frames coincident.

  */
  class CALLABLE AGXPHYSICS_EXPORT Hinge : public Constraint1DOF
  {
    public:
      /**
      Default constructor. Should be used with the addRigidBody method.
      */
      Hinge();

      /**
      Create hinge given a hinge frame (world coordinates) and one or two rigid bodies. The
      hinge axis and attachment point are given in world coordinates.
      \param hf - HingeFrame defining the hinge
      \param rb1 - First body (invalid if null)
      \param rb2 - Second body connected with the hinge (if null, first rigid body will be attached in world)
      \note If this constructor fails to create this constraint, a warning is printed and this
            constraint is cleared.
      */
      Hinge( const HingeFrame& hf, RigidBody* rb1, RigidBody* rb2 = 0 );

      /**
      Create hinge given attachment frames and one or two rigid bodies. The attachment frame is
      relative its body with the hinge axis pointing in the z-direction of the attachment frame.
      If rb2 == rb2AttachmentFrame == 0 the first body will be attached to the world given
      rb1AttachmentFrame. If rb2 == 0 and rb2AttachmentFrame != 0, then rb2AttachmentFrame will
      be used as the world frame attachment.
      \param rb1 - First rigid body (invalid if null)
      \param rb1AttachmentFrame - First rigid body attachment frame (invalid if null)
      \param rb2 - Second rigid body (if null, first rigid body will be attached in world)
      \param rb2AttachmentFrame - Second rigid body attachment frame (invalid if rb2 != 0 and this is null)
      \note If this constructor fails to create this constraint, a warning is printed and this
            constraint is cleared.
      */
      Hinge( RigidBody* rb1, Frame* rb1AttachmentFrame, RigidBody* rb2 = nullptr, Frame* rb2AttachmentFrame = nullptr );

      /**
      Create hinge given attachment frame and one body at the time. This is only valid for two
      bodies, i.e., it's not possible to use this method with bodies attached to the world.
      When the second body has been added, this constraint will calculate its initial state
      and is after that ready to be added to the simulation.
      \param rb - Current rigid body to be added to this constraint
      \param rbAttachmentFrame - Attachment frame of the current rigid body
      */
      bool addRigidBody( RigidBody* rb, Frame* rbAttachmentFrame );

      /**
      Enum used for specifying which Degree of Freedom (DOF) that should be accessed in calls to for example:
      constraint->getRegularizationParameters( dof ); constraint->setDamping( damping, dof );
      */
      enum DOF
      {
        ALL_DOF=-1,        /**< Select all degrees of freedom */
        TRANSLATIONAL_1=0, /**< Select DOF for the first translational axis */
        TRANSLATIONAL_2=1, /**< Select DOF for the second translational axis */
        TRANSLATIONAL_3=2, /**< Select DOF for the third translational axis */
        ROTATIONAL_1=3,    /**< Select DOF corresponding to the first rotational axis */
        ROTATIONAL_2=4,    /**< Select DOF corresponding to the second rotational axis */
        NUM_DOF=5          /**< Number of DOF available for this constraint */
      };

      /**
      \return the number of DOF for this constraint, not including secondary constraints. -1 if num DOF is undefined for the constraint.
      */
      int getNumDOF() const override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::Hinge );

    protected:
      virtual ~Hinge();

      /**
      Used internally for debug rendering.
      */
      virtual void render( class agxRender::RenderManager *mgr, float scale  ) const override;

    private:
      class HingeImplementation *m_implementation;
  };

  typedef ref_ptr<Hinge> HingeRef;

} //namespace agx

#endif
