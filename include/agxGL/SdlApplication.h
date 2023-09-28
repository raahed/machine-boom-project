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
#ifndef AGXGL_SDLAPPLICATION_H
#define AGXGL_SDLAPPLICATION_H

#if AGX_USE_SDL()

#include <agx/config/AGX_USE_SDL.h>
#include <agx/Referenced.h>
#include <agx/Component.h>
#include <agx/Timer.h>
#include <agxGL/Camera.h>
#include <SDL/SDL.h>

namespace agxGL
{
  AGX_DECLARE_POINTER_TYPES(SdlApplication);
  class AGXPHYSICS_EXPORT SdlApplication : public agx::Referenced
  {
  public:
    SdlApplication(agx::System *system, agx::UInt width = 1024, agx::UInt height = 768);

    void run(bool blocking = true);

  protected:
    virtual ~SdlApplication();

  private:
    void initSDL(agx::UInt width, agx::UInt height);
    void handleEvents();
    void handleContinuousEvents(agx::Real dt);
    void handleEvent(const SDL_Event& event);
    void initComponent(agx::Component *component);

  private:
    bool m_running;
    bool m_autoStepping;
    agx::SystemRef m_system;
    CameraRef m_camera;

    bool m_mouseDragged;
    agx::Vec2i m_mouseDragStartPosition;
    agx::HashSet<int> m_continuousKeys;
    agx::Vec2i m_relMouse;
    agx::Real m_moveSpeed;
    agx::Real m_zoomSpeed;
    agx::Real m_mouseSensitivity;
    agx::Real m_moveSpeedDelta;
    agx::Timer m_eventTimer;
    agx::Timer m_stepTimer;
    agx::Real m_accumulatedTime;
  };

}

#endif

#endif /* _AGXGL_SDLAPPLICATION_H_ */
