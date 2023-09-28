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

#ifndef AGXSDK_GUIEVENTADAPTER_H
#define AGXSDK_GUIEVENTADAPTER_H


#include <agx/agxPhysics_export.h>

namespace agxSDK
{
  class GuiEvent;
  class Simulation;

  /// Base class for inserting mouse and keyboard event into agxSDK::Simulation
  class AGXPHYSICS_EXPORT GuiEventAdapter
  {

    public:

      enum MouseButtonMask {
        NO_MOUSE_BUTTON = 0,
        LEFT_MOUSE_BUTTON = 1,
        MIDDLE_MOUSE_BUTTON = 2,
        RIGHT_MOUSE_BUTTON = 4
      };



      enum EventType {
        NONE = 0,
        PUSH = 1,
        RELEASE = 2,
        DOUBLECLICK = 4,
        DRAG = 8,
        MOVE = 16,
        KEYDOWN = 32,
        KEYUP = 64,
        FRAME = 128,
        RESIZE = 256,
        SCROLL = 512,
        UPDATE = 1024,

        CLOSE_WINDOW = 8192,
        QUIT_APPLICATION = 16384,
        USER = 32768
      };

    public:

      /// Constructor
      GuiEventAdapter( Simulation* );
      virtual ~GuiEventAdapter();

      void setSimulation( Simulation*);
      Simulation* getSimulation( );

      /**
      Execute the event
      \return true if the event is handled by a listener. Otherwise false
      */
      bool triggerEvent( const GuiEvent& event );

      /**
      Update all added/removed GuiEvents.
      */
      void updateGuiEvents();

    protected:


    private:
      class GuiEventAdapterImplementation* m_implementation;
  };
}

#endif
