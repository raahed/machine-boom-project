/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
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

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/ClipNode>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.
#include <agx/Plane.h>

#include <agxOSG/export.h>
#include <agxOSG/Node.h>

// Forward declaration
namespace agx
{
  class ParticleSystem;
}

namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(ClipPlane);
  AGX_DECLARE_VECTOR_TYPES(ClipPlane);

  /**
   This class wraps the ClipPlane functionality of osg together with a clip node and a clip plane.
   This also extends the node with an update callback that culls particles against the clip plane.
   This node should be inserted at the top of the scene node in order to clip everything under it.
  */
  class AGXOSG_EXPORT ClipPlane : public osg::ClipNode
  {
  public:
    /**
    Create clip plane from agx::Plane.
    */
    ClipPlane( const agx::Plane& plane );

    /**
    Create clip plane from a position and a normal.
    */
    ClipPlane( const agx::Vec3& normal, const agx::Vec3& position );

    /**
    Create clip plane from plane constants.
    */
    ClipPlane( agx::Real a, agx::Real b, agx::Real c, agx::Real d );

    /**
    Set clip plane from an agx plane.
    \param plane - the plane that will be used to set the clip plane.
    */
    void setClipPlane( const agx::Plane& plane );

    /**
    Set position of clip plane.
    \param position - the position to be set to the plane.
    */
    void setPosition( const agx::Vec3& position );

    /**
    Set normal of clip plane.
    \param normal - the normal to be set to the plane.
    */
    void setNormal( const agx::Vec3& normal );

    /**
    Set the particle system that should be clipped against the clip plane.
    \param system - the particle system to should be clipped against the plane.
    */
    void setParticleSystemToClip( agx::ParticleSystem* system );

    /**
    Set enable of the clip plane.
    \param enable - true is clip plane should be enabled, false otherwise.
    */
    void setEnable(bool enable);

    /**
    Get a created agx::Plane representation of the clip plane.
    */
    agx::Plane getPlane() const;

    /**
    Get the position of the clip plane.
    */
    agx::Vec3 getPosition() const;

    /**
    Get the normal of the clip plane.
    */
    agx::Vec3 getNormal() const;

    /**
    Get if the clip plane is enabled.
    */
    bool getEnable() const;

    /**
    Performs the clipping of the set particle system.
    */
    void clipAgainstParticleSystem();

  private:
    void setDataFromPlane(const agx::Plane& plane);

  private:
    agx::Vec3                    m_position;
    agx::Vec3                    m_normal;
    osg::ref_ptr<osg::ClipPlane> m_clipPlane;
    agx::ParticleSystem*         m_system;
  };
}
