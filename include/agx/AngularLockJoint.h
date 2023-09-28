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

#ifndef AGX_ANGULARLOCKJOINT_H
#define AGX_ANGULARLOCKJOINT_H

#include <agx/Constraint.h>

namespace agx
{

  /**
  Constraint that removes 3 DOF (rotational degrees of freedom) between two bodies, or one body and the world.
  The constraint will lock the two bodies in their initial pose at the time of calling the constructor.
  */
  class AGXPHYSICS_EXPORT AngularLockJoint : public Constraint
  {
    public:
      /**
      Create Angular lock joint given one or two rigid bodies.
      \param rb1 - First rigid body (invalid if null)
      \param rb2 - Second rigid body (if null, first rigid body will keep its initial orientation)
      */
      AngularLockJoint( RigidBody* rb1, RigidBody* rb2 = 0);

      /**
      Create Angular lock joint given one or two rigid bodies with corresponding attachment frames.
      This angular lock is fulfilled when the axes in the attachment frames coincide.
      \param rb1 - First rigid body (invalid if null)
      \param rb1AttachmentFrame - Attachment frame of first rigid body (invalid if null)
      \param rb2 - Second rigid body
      \param rb2AttachmentFrame - Attachment frame of second rigid body
      */
      AngularLockJoint( RigidBody* rb1, Frame* rb1AttachmentFrame, RigidBody* rb2 = nullptr, Frame* rb2AttachmentFrame = nullptr );

      /**
      Enum used for specifying which Degree of Freedom (DOF) that should be accessed in calls to for example:
      constraint->getRegularizationParameters( dof ); constraint->setDamping( damping, dof );
      */
      enum DOF
      {
        ALL_DOF=-1,        /**< Select all degrees of freedom */
        ROTATIONAL_1=0,    /**< Select DOF corresponding to the first rotational axis */
        ROTATIONAL_2=1,    /**< Select DOF corresponding to the second rotational axis */
        ROTATIONAL_3=2,    /**< Select DOF for rotation around Z-axis */
        NUM_DOF=3          /**< Number of DOF available for this constraint */
      };

      /**
      \return the number of DOF for this constraint, not including secondary constraints. -1 if num DOF is undefined for the constraint.
      */
      virtual int getNumDOF() const override
      {
        return  NUM_DOF;
      }

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::AngularLockJoint);

      virtual void render(class agxRender::RenderManager *, float /*scale = 1.0f*/ ) const override
      {}

    protected:
      AngularLockJoint();
      virtual ~AngularLockJoint();

    private:
      class AngularLockJointImplementation *m_implementation;
  };

  typedef ref_ptr< AngularLockJoint > AngularLockJointRef;
}

#endif

