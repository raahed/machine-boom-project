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

#ifndef AGX_BALLJOINT_H
#define AGX_BALLJOINT_H

#include <agx/Constraint.h>

namespace agx
{
  /**
  Class for storing the geometric information of a BallJoint
  */
  class CALLABLE AGXPHYSICS_EXPORT BallJointFrame : public ConstraintFrame
  {
    public:
      BallJointFrame();

      /**
       \param center - Ball center position in world coordinates
       */
      BallJointFrame( const Vec3& center );

      /// Destructor
      ~BallJointFrame();

      void setCenter( const Vec3& center );
      const Vec3& getCenter() const;
  };

  /**
  Constraint that removes the three translation DOF between two bodies (or one and the world).
  */
  class CALLABLE AGXPHYSICS_EXPORT BallJoint : public Constraint
  {
    public:
      /**
      Constructor that creates a Ball Joint.
      \param bf - Frame specifying the geometry of the constraint.
      \param rb1 - The first body
      \param rb2 - If null, rb1 is attached to the world, if not, rb1 and rb2 are attached to each other.
      */
      BallJoint( const BallJointFrame& bf, RigidBody* rb1, RigidBody* rb2 = 0 );

      /**
      Create a Ball Joint given one or two bodies with their respective attachment frames. Given two bodies and
      two frames, the ball of the ball joint will be located at these two frames origins. For one body,
      rb2 == rb2AttachmentFrame == 0, the ball of the ball joint will be located at the world position of the
      attachment frame.
      \param rb1 - First rigid body (invalid if null)
      \param rb1AttachmentFrame - First rigid body attachment frame
      \param rb2 - Second rigid body (if null, first rigid body will be attached to world)
      \param rb2AttachmentFrame - Second rigid body attachment frame (invalid if rb2 != 0 and rb2AttachmentFrame == 0)
      \note Valid configurations:   rb1 != 0 && rb1AttachmentFrame != 0,
                                    rb1 != 0 && rb1AttachmentFrame != 0 && rb2 == 0 and
                                    rb1 != 0 && rb1AttachmentFrame != 0 && rb2 != 0 && rb2AttachmentFrame != 0.
      All other configurations are invalid.
      */
      BallJoint( RigidBody* rb1, Frame* rb1AttachmentFrame, RigidBody* rb2 = nullptr, Frame* rb2AttachmentFrame = nullptr );

      /**
      Create a Ball Joint given one or two bodies with relative attachment points. This is similar to attachment
      frames.
      \param rb1 - First rigid body (invalid if null)
      \param rb1LocalAttachPoint - Attach point given in first rigid body local frame
      \param rb2 - Second rigid body (if null, first rigid body will be attached to world)
      \param rb2LocalAttachPoint - Attach point given in second rigid body local frame
      */
      BallJoint( RigidBody* rb1, const Vec3& rb1LocalAttachPoint, RigidBody* rb2 = nullptr, const Vec3& rb2LocalAttachPoint = Vec3() );

      /**
      Enum used for specifying which Degree of Freedom (DOF) that should be accessed in calls to for example:
      constraint->getRegularizationParameters( dof ); constraint->setDamping( damping, dof );
      */
      enum DOF {
        ALL_DOF = -1,        /**< Select all degrees of freedom */
        TRANSLATIONAL_1 = 0, /**< Select DOF for the first translational axis */
        TRANSLATIONAL_2 = 1, /**< Select DOF for the second translational axis */
        TRANSLATIONAL_3 = 2, /**< Select DOF for the third translational axis */
        NUM_DOF = 3          /**< Number of DOF available for this constraint */
      };

      /**
      \return the number of DOF for this constraint, not including secondary constraints
      */
      virtual int getNumDOF() const override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::BallJoint);

    protected:
      BallJoint();
      BallJoint( class BallJointImplementation* impl );
      virtual ~BallJoint();

      virtual void render( agxRender::RenderManager* mgr, float scale ) const override;

    private:
      class BallJointImplementation* m_implementation;
  };

  typedef ref_ptr< BallJoint > BallJointRef;

  /**
  Special version of a ball joint where the spatial reference frame is world for
  the body/bodies. This translates to that BallJoint::DOF::TRANSLATIONAL_1 is
  world x axis, BallJoint::DOF::TRANSLATIONAL_2 world y etc..

  Since the reference frame isn't moving, this type of the ball joint tends to
  be more stable for fast moving/rotating objects.
  */
  class CALLABLE AGXPHYSICS_EXPORT WorldFrameBallJoint : public BallJoint
  {
    public:
      /**
      Create a Ball Joint given one or two bodies with their respective attachment frames. Given two bodies and
      two frames, the ball of the ball joint will be located at these two frames origins. For one body,
      rb2 == rb2AttachmentFrame == 0, the ball of the ball joint will be located at the world position of the
      attachment frame.
      \param rb1 - First rigid body (invalid if null)
      \param rb1AttachmentFrame - First rigid body attachment frame
      \param rb2 - Second rigid body (if null, first rigid body will be attached to world)
      \param rb2AttachmentFrame - Second rigid body attachment frame (invalid if rb2 != 0 and rb2AttachmentFrame == 0)
      \note Valid configurations:   rb1 != 0 && rb1AttachmentFrame != 0,
                                    rb1 != 0 && rb1AttachmentFrame != 0 && rb2 == 0 and
                                    rb1 != 0 && rb1AttachmentFrame != 0 && rb2 != 0 && rb2AttachmentFrame != 0.
      All other configurations are invalid.
      */
      WorldFrameBallJoint( RigidBody* rb1, Frame* rb1AttachmentFrame, RigidBody* rb2 = nullptr, Frame* rb2AttachmentFrame = nullptr );

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::WorldFrameBallJoint);

    protected:
      WorldFrameBallJoint();
      virtual ~WorldFrameBallJoint();

    private:
      class WorldFrameBallJointImplementation* m_implementation;
  };

  typedef ref_ptr< WorldFrameBallJoint > WorldFrameBallJointRef;

} // namespace agx

#endif // AGX_BALLJOINT_H
