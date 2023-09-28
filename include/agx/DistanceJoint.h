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

#ifndef AGX_DISTANCEJOINT_H
#define AGX_DISTANCEJOINT_H

#include <agx/Constraint.h>

namespace agx
{
  /**
  This joint will preserve the initial distance between a body and a point in world coordinate or between two bodies.
  */
  class CALLABLE AGXPHYSICS_EXPORT DistanceJoint : public Constraint1DOF
  {
    public:
      /**
      Create a Distance joint given an attachment frame on a rigid body and a fixed point (in world).
      \param rb1 - First rigid body (invalid if null)
      \param rb1AttachmentFrame - First rigid body attachment frame, the distance joint starts here (invalid if null)
      \param worldAttach - Attachment point in world, given i world coordinates. The distance joint ends here.
      */
      DistanceJoint( RigidBody* rb1, Frame* rb1AttachmentFrame, const Vec3& worldAttach );

      /**
      Create a Distance joint given attachment frames and two rigid bodies.
      \param rb1 - First rigid body (invalid if null)
      \param rb1AttachmentFrame - First rigid body attachment frame, the distance joint starts here (invalid if null)
      \param rb2 - Second rigid body (invalid if null)
      \param rb2AttachmentFrame - Second rigid body attachment frame, the distance joint ends here (invalid if null)
      */
      DistanceJoint( RigidBody* rb1, Frame* rb1AttachmentFrame, RigidBody* rb2, Frame* rb2AttachmentFrame );

      /**
      Create a Distance joint given two rigid bodies and local attachment points.
      \param rb1 - First rigid body (invalid if null)
      \param localAttachRB1 - First rigid body local attach point, the distance joint starts here
      \param rb2 - Second rigid body (invalid if null)
      \param localAttachRB2 - Second rigid body local attach point, the distance joint ends here
      */
      DistanceJoint( RigidBody* rb1, const Vec3& localAttachRB1, RigidBody* rb2, const Vec3& localAttachRB2 );

      /**
      Create a Distance joint given one rigid body, a local attach point and a world attach point.
      \param rb1 - First rigid body (invalid if null)
      \param localAttachRB1 - Local attach point on rb1, the distance joint starts here
      \param worldAttach - Attach point given in world coordinates, the distance joint ends here
      */
      DistanceJoint( RigidBody* rb1, const Vec3& localAttachRB1, const Vec3& worldAttach );

      /**
      This is a 1D constraint, i.e., only index \p i = 0 is valid. This call is equal to getLock1D()->getRegularizationParameters().
      \param i - is ignored because this distance joint is 1D
      \return the regularization parameter
      */
      virtual RegularizationParameters* getRegularizationParameters( agx::UInt i ) override;
      virtual const RegularizationParameters* getRegularizationParameters( agx::UInt i ) const override;

      /**
      Set compliance for this distance joint. This call is equal to getLock1D()->getRegularizationParameters()->setCompliance( compliance ).
      */
      virtual void setCompliance( Real compliance, agx::Int index = -1 ) override;

      /**
      \return the compliance of the 1D lock (i.e., equal to getLock1D()->getRegularizationParameters()->getCompliance()), \p index is ignored
      */
      virtual Real getCompliance( agx::UInt index ) const override;

      /**
      Set damping for this distance joint. This call is equal to getLock1D()->getRegularizationParameters()->setDamping( damping ).
      */
      virtual void setDamping( Real damping, agx::Int index = -1 ) override;

      /**
      \return the damping of the 1D lock (i.e., equal to getLock1D()->getRegularizationParameters()->getDamping())
      */
      virtual Real getDamping( agx::UInt index ) const override;

      /**
      \return the number of DOF for this constraint, not including secondary constraints. -1 if num DOF is undefined for the constraint.
      */
      virtual int getNumDOF() const override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agx::DistanceJoint);

    protected:
      DistanceJoint();
      virtual ~DistanceJoint();

      /**
      Internal method used for debug rendering
      */
      virtual void render(  agxRender::RenderManager *mgr, float scale   ) const override;

    private:
      class DistanceJointImplementation *m_implementation;
  };

  typedef ref_ptr<DistanceJoint> DistanceJointRef;
} // namespace agx

#endif
