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

#include <agx/Constraint.h>
#include <agx/Hinge.h>

namespace agx
{


  /**
  A constraint that is equal to the \sa Hinge but also has support for a slack range for the
  three translational DOF and for all rotational DOF.
  This can be used to simulate a gap in which two bodies can move freely.

  Each hinged body has an attachment frame with axes (U,V,N). With a regular Hinge, the N-axes
  for the bodies are equivalent. For a SlackHingeJoint, the N-axes can be different if rotational
  slack is used. Furthermore, if a motor is enabled to drive the hinge, the N-axis from body1s
  attachment frame will be used by the motor as rotational axis.
  Therefor a SlackHingeJoint is affected by the order of the bodies in the constructor unlike
  an ordinary Hinge.
  */
  class CALLABLE AGXPHYSICS_EXPORT SlackHingeJoint : public Constraint1DOF
  {
    public:
      /**
      Create hinge given a hinge frame (world coordinates) and one or two rigid bodies. The
      hinge axis and attachment point are given in world coordinates.
      \param hf - HingeFrame defining the hinge
      \param rb1 - First body (invalid if null)
      \param rb2 - Second body connected with the hinge (if null, first rigid body will be attached in world)
      \note If this constructor fails to create this constraint, a warning is printed and this
            constraint is cleared.
      */
      SlackHingeJoint( const HingeFrame& hf, RigidBody* rb1, RigidBody* rb2 = 0 );

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
      SlackHingeJoint( RigidBody* rb1, Frame* rb1AttachmentFrame, RigidBody* rb2 = nullptr, Frame* rb2AttachmentFrame = nullptr );


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


      /**
      Set slack range parameters. Positional slack parameter is in meters and specifies a range in which
      the constraint allow free movement for the three constrained translational DOF.

      Rotational parameter is in radians and specifies a range in which
      the constraint allows free rotation for all of the rotational DOF.

      The slack range parameters are symmetrical which means that the total range is twice the value used.

      \param pos_uvn Symmetrical positional slack range (m) in the first bodies attachment frame
      \param rot     Symmetrical rotational slack range (rad)
      \return True if parameters could be set.
      */
      bool setSlackParameters( agx::Vec3 pos_uvn, agx::Real rot );

      /**
      Read the current slack parameters.
      \return true if values are written
      */
      bool getSlackParameters( agx::Vec3& pos_uvn, agx::Real& rot ) const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::SlackHingeJoint );

    protected:
      /**
      Default constructor.
      */
      SlackHingeJoint();

      virtual ~SlackHingeJoint();

      /**
      Used internally for debug rendering.
      */
      virtual void render( class agxRender::RenderManager *mgr, float scale  ) const override;

    private:
      class SlackHingeJointImplementation *m_implementation;
  };

  typedef ref_ptr<SlackHingeJoint> SlackHingeJointRef;

  typedef SlackHingeJoint SlackHinge; // Hinge does not use Joint in its name, mimic naming

} //namespace agx

