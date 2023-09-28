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

#include <agx/agxPhysics_export.h>
#include <agxSDK/GuiEventListener.h>

#include <agxCollide/Geometry.h>
#include <agx/Constraint.h>
#include <agxCollide/LocalContactPoint.h>

namespace agx
{
  class DistanceJoint;
  class BallJoint;
  class LockJoint;
}

namespace agxSDK
{
  // class to handle events with a pick
  class AGXPHYSICS_EXPORT PickHandler : public agxSDK::GuiEventListener
  {
  public:
    // Old interface
    virtual PickResult pick( float /*screenX*/, float /*screenY*/ ) { return PickResult(); };

  public:
    PickHandler();

    virtual void addNotification();
    virtual void removeNotification();

    /**
    Attaches a constraint to the specified body
    \param worldPoint - Point to which a constraint will be attached to the body
    \param body - The body to be attached to the constraint
    \param useLock - Use lock instead of ball joint
    \param updateRotation - When useLock is set, should the rotation be updated instead of the position?
    \return true if a constraint was created
    */
    virtual bool attachBody(const agx::Vec3& worldPoint, agx::RigidBody *body, bool useLock = false, bool updateRotation = false);

    bool attachGeometry(const agx::Vec3& worldPoint, agxCollide::Geometry *geometry, bool useLock = false);

    /**
    Update the attachment point with the \p mousePositionWorld which is the point of the mouse in world coordinates
    \return true if attachment was updated successfully.
    */
    virtual bool updateAttachment(const agx::Vec3& mousePositionWorld);

    /// \return true if Simulation is in playback mode
    bool playbackMode() const;

    /// Will detach constraints and reset picked body/geometry
    virtual void reset();

    /// \return the current attached rigid body
    agx::RigidBody *getRigidBody();

    agxCollide::Geometry *getAttachedGeometry();


    /// \return the current point of the mouse in the world
    agx::Vec3 getCurrentWorldMousePoint() const;

    /// \return the current attachment point to the body
    agx::Vec3 getWorldAttachmentPoint() const;


    bool isActive() const;

  protected:
    virtual ~PickHandler();

    /// \return true if the given body and geometry are valid for being attached to a constraint
    bool validate(agx::RigidBody *body);

    /// \return the current position where the constraint is attached, relative to the picked geometry
    agx::Vec3 getLocalAttachmentPoint() const;

    void calculateComplianceAndDamping( agx::DistanceJoint* constraint, agx::RigidBody* rb );
    void calculateComplianceAndDamping( agx::BallJoint* constraint, agx::RigidBody* rb );
    void calculateComplianceAndDamping( agx::LockJoint* constraint, agx::RigidBody* rb );

  protected:

    agx::RigidBodyRef m_rigidBody;
    agxCollide::GeometryRef m_geometry;
    agx::ConstraintRef m_constraint;

    bool m_updateRotation;


    struct PickedParticle
    {
      PickedParticle() {}
      PickedParticle(agx::Physics::ParticlePtr p) : particle(p), material(p.material()), color(p.color()) {}

      agx::Physics::ParticlePtr particle;
      agx::Physics::MaterialPtr material;
      agx::Vec4f color;
    };
    agx::Physics::ParticlePtr m_particle;

    agx::Vector<PickedParticle> m_particleSet;
    agx::MaterialRef m_particleMaterial;

  private:


    agx::Vec3 m_initialMousePointWorld;
    agx::Vec3 m_currentMousePointWorld;
    agx::Vec3 m_worldAttachmentPoint;
    agx::Vec3 m_localAttachmentPoint;
    agx::Quat m_oldBodyRotation;

    agx::FrameRef m_constraintWorldFrame;
    agx::FrameRef m_constraintBodyFrame;

    agx::ComponentRef m_mouseComponent;
    agxData::ValueRef m_pickAttachmentPoint;
    agxData::ValueRef m_pickHandlePoint;
    agxData::ValueRef m_isPicking;
  };

  typedef agx::ref_ptr<PickHandler> PickHandlerRef;

}

