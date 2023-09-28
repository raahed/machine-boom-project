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

#ifndef AGXSDK_GUIEVENT_H
#define AGXSDK_GUIEVENT_H

#include <agxSDK/GuiEventAdapter.h>
#include <agx/agxPhysics_export.h>

namespace agxSDK
{

  /// Class for storing an Event which will be executed later inside Simulation.
  class AGXPHYSICS_EXPORT GuiEvent
  {
  public:


    GuiEvent( GuiEventAdapter::EventType type,
      float x, float y );

    GuiEvent( GuiEventAdapter::EventType type,
      GuiEventAdapter::MouseButtonMask mouseButtonMask,
      float x, float y );

    GuiEvent( GuiEventAdapter::EventType type,
      int key, int unmodifiedKey,
      unsigned int modKeyMask,
      float x, float y);

    unsigned int getModKeyMask() const { return m_modKeyMask; }
    int getKey( ) const { return m_key; }
    int getUnmodifiedKey() const { return m_unmodifiedKey; }
    GuiEventAdapter::EventType getEventType() const { return m_eventType; }
    GuiEventAdapter::MouseButtonMask getMouseButtonMask() const { return m_mouseButtonMask; }

    float getX() const { return m_x; }
    float getY() const { return m_y; }

  private:

    unsigned int m_modKeyMask;
    int m_key;
    int m_unmodifiedKey;

    GuiEventAdapter::EventType m_eventType;
    GuiEventAdapter::MouseButtonMask m_mouseButtonMask;

    float m_x, m_y;
    // float m_time;
  };

}
#endif
