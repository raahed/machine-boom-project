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

#ifndef AGX_PLANEJOINT_H
#define AGX_PLANEJOINT_H

#include <agx/Constraint.h>

namespace agx
{
  class AGXPHYSICS_EXPORT PlaneJoint : public agx::Constraint
  {
    public:
      /**
      Create a plane joint between one body and world or between two bodies. The second frames z-axis
      defines the plane in which the two frames are constrained to live.
      \param rb1 - first rigid body (invalid if 0)
      \param rb1AttachmentFrame - attachment frame belonging to rb1 (invalid if 0)
      \param rb2 - second rigid body, if 0, rb1 will be constrained in world given rb2AttachmentFrame as world frame
      \param rb2AttachmentFrame - attachment frame belonging to rb2 or world frame if rb2 = 0
      */
      PlaneJoint( agx::RigidBody* rb1, agx::Frame* rb1AttachmentFrame, agx::RigidBody* rb2 = nullptr, agx::Frame* rb2AttachmentFrame = nullptr );

      enum DOF
      {
        ALL_DOF = -1,        /**< Select all degrees of freedom */
        TRANSLATIONAL_1 = 0, /**< Select DOF corresponding to the first rotational axis */
        NUM_DOF = 1          /**< Number of DOF available for this constraint */
      };

      virtual int getNumDOF() const override
      {
        return NUM_DOF;
      }

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::PlaneJoint );

      virtual void render(class agxRender::RenderManager *, float /*scale = 1.0f*/ ) const override
      {}


    protected:
      PlaneJoint();
      virtual ~PlaneJoint();

    private:
      class PlaneJointImplementation* m_implementation;
  };

  typedef agx::ref_ptr< agx::PlaneJoint > PlaneJointRef;
} //namespace agx

#endif
