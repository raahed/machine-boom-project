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

#include <agxSDK/EventListener.h>
#include <agx/TimeStamp.h>
#include <agxSDK/agxSDK.h>
#include <agxSDK/GuiEventAdapter.h>
#include <agx/Vec3.h>
#include <agxCollide/Geometry.h>
#include <agxCollide/LocalContactPoint.h>


namespace agxSDK
{

  struct AGXPHYSICS_EXPORT PickResult
  {
    PickResult() : geometry(nullptr), distance(0), hit(false) {}

    agx::observer_ptr< agxCollide::Geometry > geometry;
    agx::Physics::ParticlePtr particle;
    agx::Real distance;
    agx::Vec3 worldPoint;
    agx::Vec3 localPoint;
    agx::Vec3 worldNormal;
    agx::Vec3 localNormal;
    bool hit;
  };


  /**
  Derive from this class to implement a listener for simulation GuiEvents.
  The events that can be listened to are:
  keyboard, mouseDragged, mouseMoved, mouse and update. See each method for detailed information.
  */
  class AGXPHYSICS_EXPORT GuiEventListener : public EventListener
  {
    public:
      friend class EventManager;

      enum MouseState {
        NO_MOUSE_STATE = 0,
        MOUSE_DOWN = 1,
        MOUSE_UP = 2,
        DOUBLECLICK = 4
      };

      enum MouseButtonMask {
        NO_MOUSE_BUTTON = GuiEventAdapter::NO_MOUSE_BUTTON,
        LEFT_MOUSE_BUTTON = GuiEventAdapter::LEFT_MOUSE_BUTTON,
        MIDDLE_MOUSE_BUTTON = GuiEventAdapter::MIDDLE_MOUSE_BUTTON,
        RIGHT_MOUSE_BUTTON = GuiEventAdapter::RIGHT_MOUSE_BUTTON
      };


      enum KeySymbol {
        KEY_Space           = 0x20,

        KEY_BackSpace       = 0xFF08,        /* back space, back char */
        KEY_Tab             = 0xFF09,
        KEY_Linefeed        = 0xFF0A,        /* Linefeed, LF */
        KEY_Clear           = 0xFF0B,
        KEY_Return          = 0xFF0D,        /* Return, enter */
        KEY_Pause           = 0xFF13,        /* Pause, hold */
        KEY_Scroll_Lock     = 0xFF14,
        KEY_Sys_Req         = 0xFF15,
        KEY_Escape          = 0xFF1B,
        KEY_Delete          = 0xFFFF,        /* Delete, rubout */


        /* Cursor control & motion */

        KEY_Home            = 0xFF50,
        KEY_Left            = 0xFF51,        /* Move left, left arrow */
        KEY_Up              = 0xFF52,        /* Move up, up arrow */
        KEY_Right           = 0xFF53,        /* Move right, right arrow */
        KEY_Down            = 0xFF54,        /* Move down, down arrow */
        KEY_Prior           = 0xFF55,        /* Prior, previous */
        KEY_Page_Up         = 0xFF55,
        KEY_Next            = 0xFF56,        /* Next */
        KEY_Page_Down       = 0xFF56,
        KEY_End             = 0xFF57,        /* EOL */
        KEY_Begin           = 0xFF58,        /* BOL */


        /* Misc Functions */

        KEY_Select          = 0xFF60,        /* Select, mark */
        KEY_Print           = 0xFF61,
        KEY_Execute         = 0xFF62,        /* Execute, run, do */
        KEY_Insert          = 0xFF63,        /* Insert, insert here */
        KEY_Undo            = 0xFF65,        /* Undo, oops */
        KEY_Redo            = 0xFF66,        /* redo, again */
        KEY_Menu            = 0xFF67,        /* On Windows, this is VK_APPS, the component-menu key */
        KEY_Find            = 0xFF68,        /* Find, search */
        KEY_Cancel          = 0xFF69,        /* Cancel, stop, abort, exit */
        KEY_Help            = 0xFF6A,        /* Help */
        KEY_Break           = 0xFF6B,
        KEY_Mode_switch     = 0xFF7E,        /* Character set switch */
        KEY_Script_switch   = 0xFF7E,        /* Alias for mode_switch */
        KEY_Num_Lock        = 0xFF7F,

        /* Keypad Functions, keypad numbers cleverly chosen to map to ascii */

        KEY_KP_Space        = 0xFF80,        /* space */
        KEY_KP_Tab          = 0xFF89,
        KEY_KP_Enter        = 0xFF8D,        /* enter */
        KEY_KP_F1           = 0xFF91,        /* PF1, KP_A, ... */
        KEY_KP_F2           = 0xFF92,
        KEY_KP_F3           = 0xFF93,
        KEY_KP_F4           = 0xFF94,
        KEY_KP_Home         = 0xFF95,
        KEY_KP_Left         = 0xFF96,
        KEY_KP_Up           = 0xFF97,
        KEY_KP_Right        = 0xFF98,
        KEY_KP_Down         = 0xFF99,
        KEY_KP_Prior        = 0xFF9A,
        KEY_KP_Page_Up      = 0xFF9A,
        KEY_KP_Next         = 0xFF9B,
        KEY_KP_Page_Down    = 0xFF9B,
        KEY_KP_End          = 0xFF9C,
        KEY_KP_Begin        = 0xFF9D,
        KEY_KP_Insert       = 0xFF9E,
        KEY_KP_Delete       = 0xFF9F,
        KEY_KP_Equal        = 0xFFBD,        /* equals */
        KEY_KP_Multiply     = 0xFFAA,
        KEY_KP_Add          = 0xFFAB,
        KEY_KP_Separator    = 0xFFAC,       /* separator, often comma */
        KEY_KP_Subtract     = 0xFFAD,
        KEY_KP_Decimal      = 0xFFAE,
        KEY_KP_Divide       = 0xFFAF,

        KEY_KP_0            = 0xFFB0,
        KEY_KP_1            = 0xFFB1,
        KEY_KP_2            = 0xFFB2,
        KEY_KP_3            = 0xFFB3,
        KEY_KP_4            = 0xFFB4,
        KEY_KP_5            = 0xFFB5,
        KEY_KP_6            = 0xFFB6,
        KEY_KP_7            = 0xFFB7,
        KEY_KP_8            = 0xFFB8,
        KEY_KP_9            = 0xFFB9,

        /*
        * Auxiliary Functions; note the duplicate definitions for left and right
        * function keys;  Sun keyboards and a few other manufactures have such
        * function key groups on the left and/or right sides of the keyboard.
        * We've not found a keyboard with more than 35 function keys total.
        */

        KEY_F1              = 0xFFBE,
        KEY_F2              = 0xFFBF,
        KEY_F3              = 0xFFC0,
        KEY_F4              = 0xFFC1,
        KEY_F5              = 0xFFC2,
        KEY_F6              = 0xFFC3,
        KEY_F7              = 0xFFC4,
        KEY_F8              = 0xFFC5,
        KEY_F9              = 0xFFC6,
        KEY_F10             = 0xFFC7,
        KEY_F11             = 0xFFC8,
        KEY_F12             = 0xFFC9,
        KEY_F13             = 0xFFCA,
        KEY_F14             = 0xFFCB,
        KEY_F15             = 0xFFCC,
        KEY_F16             = 0xFFCD,
        KEY_F17             = 0xFFCE,
        KEY_F18             = 0xFFCF,
        KEY_F19             = 0xFFD0,
        KEY_F20             = 0xFFD1,
        KEY_F21             = 0xFFD2,
        KEY_F22             = 0xFFD3,
        KEY_F23             = 0xFFD4,
        KEY_F24             = 0xFFD5,
        KEY_F25             = 0xFFD6,
        KEY_F26             = 0xFFD7,
        KEY_F27             = 0xFFD8,
        KEY_F28             = 0xFFD9,
        KEY_F29             = 0xFFDA,
        KEY_F30             = 0xFFDB,
        KEY_F31             = 0xFFDC,
        KEY_F32             = 0xFFDD,
        KEY_F33             = 0xFFDE,
        KEY_F34             = 0xFFDF,
        KEY_F35             = 0xFFE0,

        /* Modifiers */

        KEY_Shift_L         = 0xFFE1,        /* Left shift */
        KEY_Shift_R         = 0xFFE2,        /* Right shift */
        KEY_Control_L       = 0xFFE3,        /* Left control */
        KEY_Control_R       = 0xFFE4,        /* Right control */
        KEY_Caps_Lock       = 0xFFE5,        /* Caps lock */
        KEY_Shift_Lock      = 0xFFE6,        /* Shift lock */

        KEY_Meta_L          = 0xFFE7,        /* Left meta */
        KEY_Meta_R          = 0xFFE8,        /* Right meta */
        KEY_Alt_L           = 0xFFE9,        /* Left alt */
        KEY_Alt_R           = 0xFFEA,        /* Right alt */
        KEY_Super_L         = 0xFFEB,        /* Left super */
        KEY_Super_R         = 0xFFEC,        /* Right super */
        KEY_Hyper_L         = 0xFFED,        /* Left hyper */
        KEY_Hyper_R         = 0xFFEE         /* Right hyper */
      };

      enum ModKeyMask {
        MODKEY_LEFT_SHIFT  = 0x0001,
        MODKEY_RIGHT_SHIFT = 0x0002,
        MODKEY_LEFT_CTRL   = 0x0004,
        MODKEY_RIGHT_CTRL  = 0x0008,
        MODKEY_LEFT_ALT    = 0x0010,
        MODKEY_RIGHT_ALT   = 0x0020,
        MODKEY_LEFT_META   = 0x0040,
        MODKEY_RIGHT_META  = 0x0080,
        MODKEY_LEFT_SUPER  = 0x0100,
        MODKEY_RIGHT_SUPER = 0x0200,
        MODKEY_LEFT_HYPER  = 0x0400,
        MODKEY_RIGHT_HYPER = 0x0800,
        MODKEY_NUM_LOCK    = 0x1000,
        MODKEY_CAPS_LOCK   = 0x2000,
        MODKEY_CTRL        = (MODKEY_LEFT_CTRL | MODKEY_RIGHT_CTRL),
        MODKEY_SHIFT       = (MODKEY_LEFT_SHIFT | MODKEY_RIGHT_SHIFT),
        MODKEY_ALT         = (MODKEY_LEFT_ALT | MODKEY_RIGHT_ALT),
        MODKEY_META        = (MODKEY_LEFT_META | MODKEY_RIGHT_META),
        MODKEY_SUPER       = (MODKEY_LEFT_SUPER | MODKEY_RIGHT_SUPER),
        MODKEY_HYPER       = (MODKEY_LEFT_HYPER | MODKEY_RIGHT_HYPER)
      };

      enum MouseYOrientation {
        Y_INCREASING_UPWARDS,
        Y_INCREASING_DOWNWARDS
      };

      enum ScrollingMotion {
        SCROLL_NONE,
        SCROLL_LEFT,
        SCROLL_RIGHT,
        SCROLL_UP,
        SCROLL_DOWN,
        SCROLL_2D
      };


      /// Defines the event states for which a listener will be activated, a mask can be bitwise OR:ed from these members.
      enum ActivationMask {
        NONE = 0,

        MOUSE_BUTTON = 1,
        MOUSE_MOTION = 2,
        MOUSE_PASSIVE_MOTION = 4,
        KEYBOARD = 8,
        UPDATE   = 16,
        DEFAULT = MOUSE_BUTTON | KEYBOARD | UPDATE,
        ALL = MOUSE_BUTTON | MOUSE_MOTION | MOUSE_PASSIVE_MOTION | KEYBOARD | UPDATE

      };

      /// Default constructor, sets the default activation mask to all (POST_STEP and PRE_STEP) events.
      GuiEventListener( ActivationMask mask = DEFAULT );

      /// Specifies a bitmask which determines which event types that will activate this listener.
      virtual void setMask(int f);

      // ================ Override these two methods to catch events

      /**
      Called when one or more mouse buttons are pressed and moved
      \param buttonMask The current mouse key pressed/released
      \param x,y coordinate in normalized screen coordinates of the mouse pointer
      \return true if this Listener handled the event
      */
      virtual bool mouseDragged( MouseButtonMask buttonMask, float x, float y );

      /**
      \param x,y coordinate in normalized screen coordinates of the mouse pointer
      \return true if this Listener handled the event
      */
      virtual bool mouseMoved( float x, float y );

      /**
      \param button - The current mouse key pressed/released
      \param x,y coordinate in normalized screen coordinates of the mouse pointer
      \return true if this Listener handled the event
      */
      virtual bool mouse( MouseButtonMask button, MouseState state, float x, float y );

      /**
      \param key The keyboard key pressed/released
      \param modKeyMask - THe bitmask indicating the current pressed alt/ctrl/shift buttons
      \param x,y coordinate in normalized screen coordinates of the mouse pointer
      \param keydown - True if this is a keypress event, false if its a key release event
      \return true if this Listener handled the event
      */
      virtual bool keyboard( int key, unsigned int modKeyMask, float x, float y, bool keydown );

      /**
      Called once per simulation frame
      \param x,y coordinate in normalized screen coordinates of the mouse pointer
      */
      virtual void update( float x, float y );

      /**
      Shoot a ray from start to end into the space and calculate the closest geometry
      \param start - start of the ray
      \param end - end of the ray
      \param onlyDynamics - only return results including geometries which has dynamic rigid bodies
      */
      agxSDK::PickResult intersect(const agx::Vec3& start, const agx::Vec3& end, bool onlyDynamics);

      virtual void setInverseTransform(const agx::AffineMatrix4x4& inverseTransform);


    protected:

      size_t findIntersectGeometryIndex(const agxCollide::LocalGeometryContactVector& geometryContacts, bool onlyDynamic);

      bool _mouseDragged( MouseButtonMask buttonMask, float x, float y );


      bool _mouseMoved( float x, float y );

      bool _mouse( MouseButtonMask buttonMask, MouseState state, float x, float y );

      bool _keyboard( int key, unsigned int modKeyMask, float x, float y, bool keydown  );

      void _update( float x, float y );

      virtual ~GuiEventListener() {}

      agx::AffineMatrix4x4 m_directTransform;
      agx::AffineMatrix4x4 m_inverseTransform;

  };

  typedef agx::ref_ptr<GuiEventListener> GuiEventListenerRef;


  inline GuiEventListener::GuiEventListener( ActivationMask mask ) : EventListener( GUI_EVENT_LISTENER) {
    setMask( mask );
  }

  /// Specifies a bitmask which determines which event types that will activate this listener.
  inline void GuiEventListener::setMask(int f) {
    EventListener::setMask(f);
  }

  inline bool GuiEventListener::mouseDragged( MouseButtonMask /*buttonMask*/, float /*x*/, float /*y*/ ) {
    return false;
  }

  inline bool GuiEventListener::mouseMoved( float /*x*/, float /*y*/ ) {
    return false;
  }

  inline bool GuiEventListener::mouse( MouseButtonMask /*button*/, MouseState /*state*/, float /*x*/, float /*y*/ ) {
    return false;
  }

  inline bool GuiEventListener::keyboard( int /*key*/, unsigned int /*modKeyMask*/, float /*x*/, float /*y*/, bool /*keydown*/ ) {
    return false;
  }

  inline void GuiEventListener::update( float /*x*/, float /*y*/ ) {
    return;
  }

  inline bool GuiEventListener::_mouseDragged( MouseButtonMask buttonMask, float x, float y )  {
    if (getMask() & MOUSE_MOTION)
      return mouseDragged( buttonMask, x, y );
    return false;
  }


  inline bool GuiEventListener::_mouseMoved( float x, float y ) {
    if (getMask() & MOUSE_PASSIVE_MOTION)
      return mouseMoved( x, y );
    return false;
  }

  inline bool GuiEventListener::_mouse( MouseButtonMask buttonMask, MouseState state, float x, float y ) {
    if (getMask() & MOUSE_BUTTON) return mouse( buttonMask, state, x, y);
    return false;
  }

  inline bool GuiEventListener::_keyboard( int key, unsigned int modKeyMask, float x, float y, bool keydown  ) {
    if (getMask() & KEYBOARD)
      return keyboard( key, modKeyMask,  x,  y, keydown );
    return false;
  }

  inline void GuiEventListener::_update( float x, float y )  {
    if (getMask() & UPDATE)
      return update( x, y );
    return;
  }

}
