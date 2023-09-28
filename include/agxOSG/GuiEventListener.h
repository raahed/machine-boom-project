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

#include <agxSDK/GuiEventListener.h>
#include <agxSDK/PickHandler.h>
#include <agxOSG/export.h>

namespace osg
{
  class Camera;
}

namespace agxOSG
{

  /**
  Derive from this class to implement a listener for simulation GuiEvents.
  The events that can be listened to are:
  keyboard, mouseDragged, mouseMoved, mouse and update. See each method for detailed information.
  */
  class AGXOSG_EXPORT GuiEventListener : public agxSDK::GuiEventListener
  {
  public:

    /// Default constructor, sets the default activation mask to all (POST_STEP and PRE_STEP) events.
    GuiEventListener(ActivationMask mask = DEFAULT);

    /**
    Shoot a ray from the mouse position into the simulation and return the closest geometry
    \param x - x position in screen coordinates
    \param y - y position in screen coordinates
    param onlyDynamics - Return only geometries belonging to dynamic rigid bodies
    */
    agxSDK::PickResult intersect(float x, float y, const osg::Camera *camera, bool onlyDynamics=false);

  protected:

    size_t findIntersectGeometryIndex(const agxCollide::LocalGeometryContactVector& geometryContacts);

    void getNearFarPoints(const osg::Camera* camera, agx::Vec3& near, agx::Vec3& far, float x, float y) const;

    virtual ~GuiEventListener() {}
  };

  typedef agx::ref_ptr<GuiEventListener> GuiEventListenerRef;
}

