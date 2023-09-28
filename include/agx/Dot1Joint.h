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

#ifndef AGX_DOT1JOINT_H
#define AGX_DOT1JOINT_H

#include <agx/Constraint.h>

namespace agx
{
  /**
  Given two vectors or directions, this constraint will keep the two vectors orthogonal.
  */
  class AGXPHYSICS_EXPORT Dot1Joint : public Constraint
  {
    public:
      /**
      Create dot 1 constraint between one or two bodies. In world frame, \p rb1LocalAxis and \p rb2LocalAxis
      are constrained to be orthogonal.
      \param rb1 - first rigid body (invalid if 0)
      \param rb1LocalAxis - axis given in rb1 coordinate frame (invalid if rb1LocalAxis.length() = 0)
      \param rb2 - second rigid body, if 0, rb1 will be constrained with world
      \param rb2LocalAxis - axis given in rb2 coordinate frame. If rb2 is 0 this axis should be given in world frame (invalid if rb2LocalAxis.length() = 0)
      */
      Dot1Joint( RigidBody* rb1, const Vec3& rb1LocalAxis, RigidBody* rb2, const Vec3& rb2LocalAxis );

      /**
      Create dot 1 constraint between one or two bodies given attachment frames where the axis is defined
      to be the z-axis.
      \param rb1 - first rigid body (invalid if 0)
      \param rb1AttachmentFrame - attachment frame belonging to rb1 (invalid if 0)
      \param rb2 - second rigid body, if 0, rb1 will be constrained with world
      \param rb2AttachmentFrame - second rigid body attachment frame. If rb2 is 0 this attachment defines the world frame (invalid if 0)
      */
      Dot1Joint( RigidBody* rb1, Frame* rb1AttachmentFrame, RigidBody* rb2, Frame* rb2AttachmentFrame );

      /**
      Enum used for specifying which Degree of Freedom (DOF) that should be accessed in calls to for example:
      constraint->getRegularizationParameters( dof ); constraint->setDamping( damping, dof );
      */
      enum DOF
      {
        ALL_DOF=-1,        /**< Select all degrees of freedom */
        ROTATIONAL_1=0,    /**< Select DOF corresponding to the first rotational axis */
        NUM_DOF=1          /**< Number of DOF available for this constraint */
      };

      /**
      \return the number of DOF for this constraint, not including secondary constraints. -1 if num DOF is undefined for the constraint.
      */
      virtual int getNumDOF() const override { return  NUM_DOF; }

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::Dot1Joint);

    protected:
      Dot1Joint();
      virtual ~Dot1Joint();

      /**
      Used internally for debug rendering.
      */
      virtual void render( agxRender::RenderManager* mgr, float scale  ) const override;

    private:
      class Dot1JointImplementation* m_implementation;
  };

  typedef ref_ptr< Dot1Joint > Dot1JointRef;
} //namespace agx

#endif
