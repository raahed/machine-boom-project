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

#include <agx/Prismatic.h>

namespace agx
{

  /**
  A constraint that is equal to the \sa Prismatic but also has support for a slack range for the
  two remaining translational DOF and for all rotational DOF.
  This can be used to simulate a gap in which two bodies can move freely.
  */
  class CALLABLE AGXPHYSICS_EXPORT SlackPrismaticJoint : public Constraint1DOF
  {
    public:
      /**
      Create a Prismatic joint given a prismatic frame (world coordinates) and one or
      two rigid bodies.
      \param pf - Prismatic frame, defining the prismatic axis and attach point
      \param rb1 - First rigid body (invalid if null)
      \param rb2 - Second rigid body (if null, first rigid body will be attached in world)
      */
      SlackPrismaticJoint( const PrismaticFrame& pf, RigidBody* rb1, RigidBody* rb2 = 0 );

      /**
      Create a Prismatic joint given attachment frames and one or two rigid bodies. An
      attachment is frame relative its body with the prismatic axis point in the z-direction.
      \param rb1 - First rigid body (invalid if null)
      \param rb1AttachmentFrame - First attachment frame (invalid if null)
      \param rb2 - Second rigid body (if null, first rigid body will be attached in world)
      \param rb2AttachmentFrame - Second attachment frame
      */
      SlackPrismaticJoint( RigidBody* rb1, Frame* rb1AttachmentFrame, RigidBody* rb2 = nullptr, Frame* rb2AttachmentFrame = nullptr );

      /**
      Create a Prismatic joint given the prismatic axis and one or two rigid bodies.
      \param axis - Prismatic axis given in world coordinates
      \param rb1 - Fist rigid body (invalid if null)
      \param rb2 - Second rigid body (if null, first rigid body will be attached in world)
      */
      SlackPrismaticJoint( const Vec3& axis, RigidBody* rb1, RigidBody* rb2 = nullptr );

      /**
      Enum used for specifying which Degree of Freedom (DOF) that should be accessed in calls to for example:
      constraint->getRegularizationParameters( dof ); constraint->setDamping( damping, dof );
      */
      enum DOF {
        ALL_DOF = -1,      /**< Select all degrees of freedom */
        ROTATIONAL_1 = 0,  /**< Select DOF corresponding to the first rotational axis */
        ROTATIONAL_2 = 1,  /**< Select DOF corresponding to the second rotational axis */
        ROTATIONAL_3 = 2,  /**< Select DOF for rotation around Z-axis */
        TRANSLATIONAL_1 = 3, /**< Select DOF for the first translational axis */
        TRANSLATIONAL_2 = 4, /**< Select DOF for the second translational axis */
        NUM_DOF = 5        /**< Number of DOF available for this constraint */
      };

      /**
      \return the number of DOF for this constraint, not including secondary constraints. -1 if num DOF is undefined for the constraint.
      */
      virtual int getNumDOF() const override;

      /**
      Set slack range parameters. Positional slack parameter is in meters and specifies a range in which
      the constraint allow free movement for the two constrained translational DOF.

      Rotational parameter is in radians and specifies a range in which
      the constraint allows free rotation for all of the rotational DOF.

      The slack range parameters are symmetrical which means that the total range is twice the value used.

      \param pos_uv Symmetrical positional slack range (m) in the first bodies attachment frame
      \param rot    Symmetrical rotational slack range (rad)
      \return True if parameters could be set.
      */
      bool setSlackParameters( agx::Vec2 pos_uv, agx::Real rot );


      /**
      Read the current slack parameters.
      \return true if values are written
      */
      bool getSlackParameters( agx::Vec2& pos_uv, agx::Real& rot ) const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::SlackPrismaticJoint );

    protected:
      SlackPrismaticJoint();
      virtual ~SlackPrismaticJoint();

      /**
      Used internally for debug rendering.
      */
      virtual void render(agxRender::RenderManager* mgr, float scale ) const override;

    private:
      class SlackPrismaticJointImplementation* m_implementation;
  };

  typedef SlackPrismaticJoint SlackPrismatic; // Normal Prismatic does not have Joint in name.
  typedef ref_ptr<SlackPrismaticJoint> SlackPrismaticJointRef;



} // namespace agx

