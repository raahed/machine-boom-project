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

#include <agx/AngularLockJoint.h>
#include <agx/ConstraintImplementation.h>
#include <agxWire/Node.h>


namespace agxWire
{
  /**
  AreaDefinition that specifies an area in which an eye node can move in an EyeNodeArea constraint. It also calculates
  if the eye node position violates that boundary.
  */
  class AGXPHYSICS_EXPORT AreaDefinition : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Callback from EyeNodeArea to fetch information about the boundary. If the point is inside the boundary,
      return false from this callback. If outside, return true. Mandatory info when active are \p pointOnBoundary,
      \p dirTowardsCenter and \p overlap. If length of \p tangent is zero, viscous friction like constraint for
      sliding along the boundary is ignored.
      \param pointOnBoundary - point on boundary surface given in world coordinates
      \param dirTowardsCenter - normalized vector pointing out the way out from the overlap
      \param overlap - overlap from boundary surface, negative value
      \param tangent - orthogonal vector to dirTowardsCenter (boundary tangent) to enable friction along boundary
                       surface
      \return true if the boundary is violated (i.e., overlap < 0) - otherwise false
      */
      virtual bool getOverlap( agx::Vec3& pointOnBoundary, agx::Vec3& dirTowardsCenter, agx::Real& overlap,
        agx::Vec3& tangent ) = 0;

      /**
      \return the center of the area in body coordinates.
      */
      agx::Vec3 getLocalCenter();

      /**
      \return the rigid body associated with the area.
      */
      agx::RigidBody* getRigidBody();

      /**
      \return the direction of the area that is perpendicular to the area plane in body coordinates.
      */
      agx::Vec3 getLocalDirection();

      /**
      \return the eye node associated with the area.
      */
      agxWire::EyeNode* getEyeNode();

      /**
      Optional debug rendering of this boundary.
      */
      virtual void render( agxRender::RenderManager *, float /*scale*/ ) const {}

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxWire::AreaDefinition );

    protected:
      AreaDefinition( agx::RigidBody* rb, agx::Vec3 localCenter, agx::Vec3 localDirection, agxWire::EyeNode* eyeNode );

      virtual ~AreaDefinition() {}

      virtual void store( agxStream::OutputArchive& out ) const override;

      virtual void restore( agxStream::InputArchive& in ) override;

    protected:
      agx::observer_ptr< agx::RigidBody >   m_body;
      agx::Vec3                             m_localCenter;
      agx::Vec3                             m_localDirection;
      agx::observer_ptr< agxWire::EyeNode > m_eyeNode;
  };

  typedef agx::ref_ptr< AreaDefinition > AreaDefinitionRef;

  /**
  Constraint that allows an eye node to move in a 2D area relative a rigid body.
  */
  class AGXPHYSICS_EXPORT EyeNodeArea : public agx::Constraint
  {
    public:
      /**
      Construct the two body joint given an area definition.
      \note This constraint is special, so do not expect general constraint interface to work.
      \param areaDefinition - callback
      */
      EyeNodeArea( AreaDefinition* areaDefinition );

      /**
      Assign compliance in the constraints, except boundary friction constraint which has explicit method.
      Default: 1E-10.
      Note that index is ignored.
      \param compliance - new compliance
      */
      virtual void setCompliance( agx::Real compliance, agx::Int /*i*/ = -1 ) override;

      /**
      Assign damping in the constraints, except boundary friction constraint which has explicit method.
      Default: 0.03.
      Note that index is ignored.
      \param damping - new damping
      */
      virtual void setDamping( agx::Real damping, agx::Int /*i*/ = -1 ) override;

      /**
      \return the compliance used for all constraints except the boundary friction constraint
      */
      virtual agx::Real getCompliance( agx::UInt /*i*/ ) const override;

      /**
      \return the damping used for all constraints except the boundary friction constraint
      */
      virtual agx::Real getDamping( agx::UInt /*i*/ ) const override;

      /**
      Assign compliance to the boundary friction constraint. Large values (~0.1) will make the extra body
      slide more easily on the boundary but may not be stable during high forces/torques.
      Default: 1E-3.
      \param compliance - new compliance
      */
      virtual void setBoundaryFrictionCompliance( agx::Real compliance );

      /**
      \return the compliance used for the boundary friction constraint.
      */
      virtual agx::Real getBoundaryFrictionCompliance() const;

      inline int getNumDOF() const override { return 4; }

      AGXSTREAM_DECLARE_SERIALIZABLE( agxWire::EyeNodeArea );

    protected:
      EyeNodeArea();
      virtual ~EyeNodeArea();

      virtual void render( agxRender::RenderManager* mgr, float scale ) const override;

    private:
      class EyeNodeAreaImplementation* m_implementation;
  };

  typedef agx::ref_ptr< EyeNodeArea > EyeNodeAreaRef;


  /**
  Class for defining a circular area in which the eye node is allowed to exist.
  */
  class AGXPHYSICS_EXPORT CircularAreaDefinition : public agxWire::AreaDefinition
  {
    public:
      /**
      Circular area definition given center of circle (localCenter), radius, body,
      line direction in body coordinate system (to calculate tangent of this boundary) and the extra body.
      */
      CircularAreaDefinition( const agx::Vec3& localCenter,
                                    agx::Real radius,
                                    agx::RigidBody* body,
                                    const agx::Vec3& localLineDir,
                                    agxWire::EyeNode* eyeNode );

      CircularAreaDefinition();

      AGXSTREAM_DECLARE_SERIALIZABLE( agxWire::CircularAreaDefinition );

      virtual ~CircularAreaDefinition()
      {
      }

    protected:

      virtual bool getOverlap( agx::Vec3& pointOnBoundary, agx::Vec3& dirTowardsCenter, agx::Real& overlap,
        agx::Vec3& tangent ) override;

    protected:
      agx::Real m_radius;
  };

  /**
  Class for defining a convex area in which the eye node is allowed to exist.
  */
  class AGXPHYSICS_EXPORT ConvexAreaDefinition : public agxWire::AreaDefinition
  {
  public:
    /**
    Convex area definition given center of area (localCenter), vector of local points that defines the convex area,
     body, line direction in body coordinate system (to calculate tangent of this boundary) and the extra body.
    The coordinates will be projected onto the area plane defined by the localCenter and localLineDir.
    */
    ConvexAreaDefinition( const agx::Vec3& localCenter,
                                const agx::Vec3Vector& areaCoordinates,
                                agx::RigidBody* body,
                                const agx::Vec3& localLineDir,
                                agxWire::EyeNode* eyeNode );

    AGXSTREAM_DECLARE_SERIALIZABLE( agxWire::ConvexAreaDefinition );

  protected:
    ConvexAreaDefinition();
    virtual ~ConvexAreaDefinition() {}

    virtual bool getOverlap( agx::Vec3& pointOnBoundary, agx::Vec3& dirTowardsCenter, agx::Real& overlap,
      agx::Vec3& tangent ) override;

    virtual void render( agxRender::RenderManager *mgr, float /*scale*/ ) const override;

  protected:
    agx::Vec3Vector m_areaCoordinates; // In body coordinates
  };
}

