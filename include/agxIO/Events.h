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

#ifndef AGXIO_EVENTS_H
#define AGXIO_EVENTS_H

#include <agx/HashTable.h>
#include <agx/Event.h>
#include <agx/String.h>
#include <agx/agxCore_export.h>
#include <agx/Vec2.h>

namespace agxIO
{
  enum MouseButton
  {
    MOUSE_LEFT,
    MOUSE_RIGHT,
    MOUSE_MIDDLE,
    MOUSE_WHEELUP,
    MOUSE_WHEELDOWN
  };

  enum EventType
  {
    EVENT_DOWN,
    EVENT_UP
  };


  typedef agx::Event2<int, EventType> InputEvent;
  typedef agx::Event3<MouseButton, EventType, agx::Vec2> MouseEvent;

  void AGXCORE_EXPORT registerKeyboardCallback(int key, InputEvent::CallbackType *callback);
  // void AGXCORE_EXPORT unregisterKeyboardCallback(int key, InputEvent::CallbackType *callback);
  void AGXCORE_EXPORT triggerKeyboardEvent(int key, EventType type);
  void AGXCORE_EXPORT registerMouseCallback(MouseEvent::CallbackType *callback);
  void AGXCORE_EXPORT unregisterMouseCallback(MouseEvent::CallbackType *callback);
  void AGXCORE_EXPORT triggerMouseEvent(MouseButton button, EventType type, const agx::Vec2& pos);
}


#endif /* _AGXIO_EVENTS_H_ */
