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

#ifndef AGXSENSOR_JOYSTICK_H
#define AGXSENSOR_JOYSTICK_H

#include <agxSDK/StepEventListener.h>
#include <agxSensor/export.h>
#include <agx/HashVector.h>
#include <agx/agx_vector_types.h>

DOXYGEN_START_INTERNAL_BLOCK()
namespace OIS
{
  class JoyStickState;
  class JoyStick;
}
DOXYGEN_END_INTERNAL_BLOCK()


namespace agxSensor
{

  /** Joystick interface for gamepads etc.
  derive from this class and implement the three virtual methods to catch event
  */
  class AGXSENSOR_EXPORT Joystick : public agxSDK::StepEventListener
  {
    public:

      Joystick();

      /// Class for storing a Joystick state
      class State
      {
        public:

          /// Bitmasks indicating the state of the POV button
          enum Pov {
            CENTERED  = 0x00000000,
            NORTH     = 0x00000001,
            SOUTH     = 0x00000010,
            EAST      = 0x00000100,
            WEST      = 0x00001000,
            NORTHEAST = 0x00000101,
            SOUTHEAST = 0x00000110,
            NORTHWEST = 0x00001001,
            SOUTHWEST = 0x00001010
          };

          /// Contains the state (in/out) of all buttons
          agx::BoolVector buttons;

          /// Contains the values for all axes
          agx::Int32Vector axes;

          /// Represents the value of POV button. Maximum of 4
          int pov[4];

          /// Contains the values for all sliders X coordinate. Maximum of 4.
          int slidersX[4];

          /// Contains the values for all sliders Y coordinate. Maximum of 4.
          int slidersY[4];

          /// Copies state from device driver
          State( const OIS::JoyStickState& oisState );

        protected:
          State() {}
      };

      /**
      Called whenever a button is pressed or released on a Joystick
      \param arg - Contains the complete state of the joystick
      \param button - which button was pressed
      \param down - true if button was pressed, false if it was released
      */
      virtual bool buttonChanged( const State &arg, int button, bool down ) =0;

      /**
      Called whenever an axis control is moved.
      \param arg - contains the complete state of the Joystick
      \param axis - Which of the axis was currently modified
      */
      virtual bool axisMoved( const State &arg, int axis ) =0;

      /**
      Called whenever an POV button is moved.
      \param arg - contains the complete state of the Joystick
      \param pov - Which of the pov was currently modified
      */
      virtual bool povMoved( const State &arg, int pov ) =0;

      /**
      Called whenever an slider is moved.
      \param arg - contains the complete state of the Joystick
      \param index - Which of the slider was currently modified
      */
      virtual bool sliderMoved( const State &arg, int index) =0;

      /**
      Initializes the joystick.
      \return true if initialization was successful.
      */
      bool init();

      /**
      Normalizes axis value [-1, 1].
      \param axisValue - raw axis value
      \return real value between -1 and 1
      */
      agx::Real normalize( int axisValue ) const;

      State getState() const;

    protected:
      virtual ~Joystick();
      virtual void addNotification();
      virtual void removeNotification();
      virtual void pre(const agx::TimeStamp& );

    private:
      OIS::JoyStick* m_oisJoystick;
      void *m_listener;
      bool m_initialized;
      bool m_forceFeedbackFound;
  };


  //////////////////////////////////////////////////////////////////////////////////////
  class JoystickManager;

  class AGXSENSOR_EXPORT JoystickState
  {
    public:
      /// Bit masks indicating the state of the POV button
      enum Pov {
        CENTERED  = 0x00000000,
        NORTH     = 0x00000001,
        SOUTH     = 0x00000010,
        EAST      = 0x00000100,
        WEST      = 0x00001000,
        NORTHEAST = 0x00000101,
        SOUTHEAST = 0x00000110,
        NORTHWEST = 0x00001001,
        SOUTHWEST = 0x00001010
      };

      /// Contains the state (in/out) of all buttons
      agx::BoolVector buttons;

      /// Contains the values for all axes
      agx::Int32Vector axes;

      /// Represents the value of POV button. Maximum of 4
      int pov[4];

      /// Contains the values for all sliders X coordinate. Maximum of 4.
      int slidersX[4];

      /// Contains the values for all sliders Y coordinate. Maximum of 4.
      int slidersY[4];

      /// Contains the values for all vector3s
      //agx::IntVector vector3s;

      /// Copies state from device driver
      JoystickState( const OIS::JoyStickState& oisState );

    protected:
      JoystickState() {}
  };


  //////////////////////////////////////////////////////////////////////////////////////
  class AGXSENSOR_EXPORT JoystickListener : public agx::Referenced
  {
    public:
      enum CallbackMask {
        BUTTON = (1<<0),
        AXIS = (1<<1),
        POV = (1<<2),
        SLIDER = (1<<3),
//        VECTOR3 = (1<<4),
        ALL = (
          BUTTON
          | AXIS
          | POV
          | SLIDER
          //| VECTOR3
          ) };

      /**
      Default constructor for a joystick listener. Default listens to all.
      */
      JoystickListener()
        : m_mask( ALL ), m_manager( nullptr ) {}

      /**
      Create joystick listener given a callback mask (default ALL).
      \param mask - callback mask (e.g., AXIS or (BUTTON | AXIS))
      */
      JoystickListener( int mask )
        : m_mask( mask ), m_manager( nullptr ) {}

      /**
      Called when button \p button changed state.
      \param state - current state of the joystick
      \param button - the button pressed or released
      \param down - true if button is down, otherwise false
      \return true if this listener used this button (no other listeners will receive this call) - otherwise false
      */
      virtual bool buttonChanged( const agxSensor::JoystickState& state, int button, bool down);

      /**
      Called when the manager has a new joystick state. This means that the value of the axis could be identical
      between different calls.
      \param state - current state of the joystick
      \param axis - axis to check
      \return true if this listener used this axis value (no other listeners will receive this call) - otherwise false
      */
      virtual bool axisUpdate( const agxSensor::JoystickState& state, int axis);

      /**
      Called when POV button \p pov has changed state.
      \param state - current joystick state
      \param pov - POV that has changed state (state.pov[pov])
      \return true if this listener used this POV value (no other listeners will receive this call) - otherwise false
      */
      virtual bool povMoved( const agxSensor::JoystickState& state, int pov );

      /**
      Called when slider \p slider has changed state.
      \param state - current joystick state
      \param slider - slider that has changed state (state.slider[pov])
      \return true if this listener used this slider value (no other listeners will receive this call) - otherwise false
      */
      virtual bool sliderMoved( const agxSensor::JoystickState& state, int slider);

      /*
      Called when vector3 \p vector3 has changed state.
      \param state - current joystick state
      \param vector3 - vector3 that has changed state (state.pov[pov])
      \return true if this listener used this slider value (no other listeners will receive this call) - otherwise false
      */
      //virtual bool vector3( const agxSensor::JoystickState& /*state*/, int /*vector3*/ ) { return false; }

      /**
      Called when this listener has been added to a JoystickManager.
      */
      virtual void addNotification() {}

      /**
      Called when this listener has been removed from a JoystickManager.
      */
      virtual void removeNotification() {}

      /**
      Assign new callback mask to this listener.
      \param mask - new callback mask
      */
      void setMask( int mask ) { m_mask = mask; }

      /**
      \return current callback mask
      */
      int getMask() const { return m_mask; }

    protected:
      virtual ~JoystickListener() {}

      /**
      \return the manager managing this listener
      */
      JoystickManager* getManager() { return m_manager; }
      const JoystickManager* getManager() const { return m_manager; }

    private:
      friend class JoystickManager;
      void setManager( JoystickManager* manager )
      {
        m_manager = manager;
      }

    private:
      int m_mask;
      JoystickManager* m_manager;
  };

  typedef agx::ref_ptr< JoystickListener > JoystickListenerRef;
  typedef agx::HashVector< JoystickListener*, JoystickListenerRef > JoystickListenerContainer;


  //////////////////////////////////////////////////////////////////////////////////////
  class AGXSENSOR_EXPORT JoystickManager : public agxSDK::StepEventListener
  {
    public:
      JoystickManager(intptr_t hwnd);

      /**
      Add new joystick listener.
      \param listener - joystick listener to add
      \return true if successfully added - otherwise false
      */
      bool add( JoystickListener* listener );

      /**
      Remove joystick listener.
      \param listener - listener to remove
      \return true if successfully removed - otherwise false
      */
      bool remove( JoystickListener* listener );

      /**
      \return true if initialized and valid
      */
      bool valid() const { return m_initialized; }

      /**
      \return true if this manager is in a simulation
      */
      bool inSimulation() const { return getSimulation() != nullptr; }

      /**
      \return the current state of the joystick without performing a new capture
      */
      JoystickState getState() const;

      /**
      Normalizes axis value [-1, 1].
      \param axisValue raw axis value
      \return real value between -1 and 1
      */
      agx::Real normalize( int axisValue ) const;

      /**
      Internal method.
      Callback to this manager when a button changed state.
      */
      virtual void buttonCallback( const agxSensor::JoystickState& state, int button, bool down );

      /**
      Internal method.
      Callback to this manager when POV has changed.
      */
      virtual void povCallback( const agxSensor::JoystickState& state, int pov );

    protected:
      virtual ~JoystickManager();

      /**
      Creates the device if not already created.
      */
      virtual void addNotification();

      /**
      Destroys the device and all listeners.
      */
      virtual void removeNotification();

      /**
      Makes a capture of the joystick and fires callbacks.
      */
      virtual void pre( const agx::TimeStamp& );

    private:
      JoystickListenerContainer m_listeners;
      JoystickListenerContainer m_listenersToBeRemoved;
      OIS::JoyStick* m_oisJoystick;
      bool m_initialized;
      intptr_t m_hwnd;
  };

  typedef agx::ref_ptr< JoystickManager > JoystickManagerRef;
}

#endif
