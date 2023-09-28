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

#ifndef AGX_LOCKJOINT_H
#define AGX_LOCKJOINT_H

#include <agx/Constraint.h>

namespace agx
{
  /**
  Constraint that removes all 6 DOF between two bodies, or one body and the world.
  The constraint will lock the two bodies in their initial pose at the time of calling the constructor.
  */
  class CALLABLE AGXPHYSICS_EXPORT LockJoint : public Constraint
  {
    public:
      /**
      Create Lock joint given one or two rigid bodies. If rb2 is null, rb1 will be attached in world
      with the attachment point at model frame origin.
      \param rb1 - First rigid body (invalid if null)
      \param rb2 - Second rigid body (if null, first rigid body will be attached in world)
      */
      LockJoint( agx::RigidBody* rb1, RigidBody* rb2 = nullptr );

      /**
      Create Lock joint given one or two rigid bodies with corresponding attachment frames. The lock is
      defined to be fulfilled when the two attachment frames coincide in the world.
      \param rb1 - First rigid body (invalid if null)
      \param rb1AttachmentFrame - First rigid body attachment frame (invalid if null)
      \param rb2 - Second rigid body (if null, first rigid body will be attached in world)
      \param rb2AttachmentFrame - Second rigid body attachment frame (invalid if null and second rigid body is non-null)
      */
      LockJoint( agx::RigidBody* rb1, agx::Frame* rb1AttachmentFrame, agx::RigidBody* rb2 = nullptr, agx::Frame* rb2AttachmentFrame = nullptr );

      /**
      Create Lock joint given two rigid bodies and one attachment point (in world coordinates).
      \param rb1 - First rigid body (invalid if null)
      \param rb2 - Second rigid body (invalid if null)
      \param worldAnchorPoint - Anchor point in world coordinates
      */
      LockJoint( agx::RigidBody* rb1, agx::RigidBody* rb2, const agx::Vec3& worldAnchorPoint );

      /**
      Create Lock joint given one rigid body and a local anchor point.
      \param rb1 - First rigid body (invalid if null)
      \param localAnchorPoint - Anchor point given in first rigid body local model frame
      */
      LockJoint( agx::RigidBody* rb1, const agx::Vec3& localAnchorPoint );

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
        ROTATIONAL_3=5,    /**< Select DOF for rotation around Z-axis */
        NUM_DOF=6          /**< Number of DOF available for this constraint */
      };

      /**
      \return the number of DOF for this constraint, not including secondary constraints. -1 if num DOF is undefined for the constraint.
      */
      virtual int getNumDOF() const override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::LockJoint );

    protected:
      LockJoint();
      virtual ~LockJoint();

      /**
      Internal method used for debug rendering.
      */
      virtual void render( agxRender::RenderManager *mgr, float scale  ) const override;

    private:

      class LockJointImplementation *m_implementation;

  };

  typedef ref_ptr<LockJoint> LockJointRef;

}

#endif
